"""Evaluation suite. Run as ``python -m evals.<name>`` from the repository root.

* ``retrieval``  recall@k and MRR per retrieval stage on the SQuAD gold questions.
* ``routing``    planner accuracy on a hand-written golden set. Needs a model key.
* ``answers``    end-to-end exact match, F1, citation precision and latency. Needs keys.

Results land in ``evals/reports/`` as JSON (ignored by git); ``evals/RESULTS.md`` is the
committed summary of the runs that decided the retrieval parameters.
"""
