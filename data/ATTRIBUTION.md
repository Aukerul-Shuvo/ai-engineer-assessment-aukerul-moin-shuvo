# Data attribution

## Text corpus and gold questions: SQuAD v1.1 development set

- Source file: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
- Paper: Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. (2016). *SQuAD: 100,000+ Questions
  for Machine Comprehension of Text.* Proceedings of EMNLP 2016.
- License: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0).
- The paragraphs are excerpts from English Wikipedia articles as they stood in 2016. Wikipedia
  text is itself CC BY-SA. Each paragraph in `paragraphs.jsonl` links to its article; the live
  article may have changed since that snapshot, so the paragraph id is the exact reference and
  the URL is the human-friendly one.

What this repository changes: nothing in the text. The build script assigns each paragraph a
stable id, attaches the article URL, and records for every question the paragraph it was
written from. The transformation is `app/retrieval/build.py`; `manifest.json` records the source
hash and counts of the build that produced these files.

## Superhero data

Fetched live from https://superheroapi.com at request time and never stored in this repository.
