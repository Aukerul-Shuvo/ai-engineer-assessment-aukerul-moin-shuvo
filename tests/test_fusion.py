from app.retrieval.fusion import reciprocal_rank_fusion


def test_document_in_both_lists_beats_top_of_one_list() -> None:
    bm25 = [7, 3, 9]
    dense = [3, 1, 7]

    fused = reciprocal_rank_fusion([bm25, dense], k=60)

    order = [row for row, _ in fused]
    assert order[0] == 3, "rank 2 in one list and rank 1 in the other wins"
    assert order[1] == 7, "rank 1 and rank 3 comes next"
    assert set(order) == {7, 3, 9, 1}


def test_scores_follow_the_formula() -> None:
    fused = dict(reciprocal_rank_fusion([[5], [5]], k=60))

    assert fused[5] == 2 / 61


def test_empty_lists_are_fine() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    assert [row for row, _ in reciprocal_rank_fusion([[1, 2], []])] == [1, 2]


def test_ties_break_on_row_number_for_determinism() -> None:
    fused = reciprocal_rank_fusion([[8], [2]], k=60)

    assert [row for row, _ in fused] == [2, 8]
