# Blind Trial: UCI Bank Marketing

## Classification task

Predict whether a client of a Portuguese banking institution will subscribe
to a term deposit after a direct-marketing contact.

- Target: `y`
- Negative class: `no` -> `0`
- Positive class: `yes` -> `1`
- Metric supplied to the SuperAgent: `accuracy`
- Problem type: imbalanced binary classification
- Features: 16 categorical/integer attributes about the client, contact, and
  previous marketing campaign
- Total samples: 45,211

## Source and license

- Provider: UCI Machine Learning Repository
- Dataset: Bank Marketing
- DOI: `10.24432/C5K306`
- License: CC BY 4.0
- Official page: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing
- Official archive: https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip

## Blind-test protocol

The prepared data uses a deterministic, stratified 70%/15%/15% split with
random seed 42. The SuperAgent may use train and validation results during its
autonomous loop. Test labels must remain unused until the solution has been
automatically selected and locked.

The task name and dataset identity are recorded for audit purposes, but the
LLM experiment planner receives only the data profile, validation history,
resource/budget state, and allow-listed adapter specifications.

## Prepared paths

- Configuration: `D:\python_project\unseen_trial_bank_marketing\unseen_config.json`
- Split manifest: `D:\python_project\unseen_trial_bank_marketing\split_manifest.json`
- NPY data: `D:\python_project\unseen_trial_bank_marketing\data`
- Future runs: `D:\python_project\unseen_trial_bank_marketing\runs`

No SuperAgent training or evaluation should be performed during data
preparation.
