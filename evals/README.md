# Trevor — Evaluation Framework

Evaluation approach not yet decided. This directory will contain:

- `questions.json` — test question/expected answer pairs
- `runner.py` — runs the pipeline against test questions and scores outputs
- `results/` — scored outputs from each eval run (gitignored raw data, committed summaries)

## Design decisions to make

- Scoring method: exact match / LLM-as-judge / human review?
- Test question sourcing: hand-written before data exists, or derived from data?
- Regression gate: do evals run in CI, or manually before releases?

Document the chosen approach here when decided.
