# SuperAgent Blind Trial Record: UCI Bank Marketing

Recorded on: 2026-11-19 (Asia/Shanghai)

## 1. Trial purpose

This trial evaluates whether the frozen SuperAgent can complete an unfamiliar
classification task without dataset-specific code or human intervention during
the run. The trial is separate from the official Released Task results and is
marked as `development_run`.

The classification objective is to predict whether a bank client subscribes to
a term deposit (`no=0`, `yes=1`) from 16 client and marketing-campaign
attributes. The supplied evaluation metric is accuracy.

## 2. Dataset provenance and preparation

- Dataset: UCI Bank Marketing
- Provider: UCI Machine Learning Repository
- Official page: https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing
- DOI: `10.24432/C5K306`
- License: CC BY 4.0
- Total samples: `45,211`
- Feature count: `16`
- Data type: mixed categorical and integer attributes
- Split: deterministic stratified `70% / 15% / 15%`
- Split random seed: `42`
- Train samples: `31,647`
- Validation samples: `6,782`
- Test samples: `6,782`
- Raw archive SHA-256: `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`
- Raw CSV SHA-256: `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`
- Data preparation script: `D:\python_project\unseen_trial_bank_marketing\prepare_dataset.py`
- Split manifest: `D:\python_project\unseen_trial_bank_marketing\split_manifest.json`
- Agent configuration: `D:\python_project\unseen_trial_bank_marketing\unseen_config.json`

The preparation script performed only deterministic format conversion, label
encoding, and predefined data splitting. It did not select features, choose a
model, tune hyperparameters, run experiments, or inspect test metrics for model
selection.

## 3. Frozen SuperAgent version

- Git tag: `final-superagent-v1`
- Git commit: `3a817efe8c598feb56ca45a4795569b809add620`
- Detached worktree: `D:\python_project\superagent_unseen_test`
- Worktree status before and after run: clean
- Python executable: `D:\python_project\.venv\Scripts\python.exe`
- Python version: `3.14.6`
- Platform: `Windows-11-10.0.26200-SP0`
- NumPy: `2.5.5`
- scikit-learn: `1.9.1`
- joblib: `1.6.0`

## 4. Autonomous run identity

- Run type: `development_run`
- Planner mode: `llm`
- LLM model: `DeepSeek-V4.1-Flash`
- Run ID: `blind_bank_subscription_trial_20261119_012410_908468`
- Run directory: `D:\python_project\unseen_trial_bank_marketing\runs\blind_bank_subscription_trial_20261119_012410_908468`
- Started at: `2026-11-19T01:24:10.909264+08:00`
- Solution locked at: `2026-11-19T01:31:02.870578+08:00`
- Final evaluation at: `2026-11-19T01:33:03.133360+08:00`
- Autonomous-run elapsed time: `411.9448852 seconds`
- Completed experiments: `4`
- Successful experiments: `4`
- Failed experiments: `0`
- Autonomous stop reason: `no_validation_improvement`

## 5. Autonomous data inspection

After `main.py` started, the Agent independently detected:

- Modality: `tabular`
- Task type: `binary_classification`
- Format: `npy_split`
- Train samples: `31,647`
- Validation samples: `6,782`
- Features: `16`
- Features treated as categorical: `15`
- Missing values: none detected
- Constant columns: none detected
- Training class distribution: class 0 = `27,944`, class 1 = `3,703`
- Metric: `accuracy`

The test labels and test metrics were not supplied to the experiment planner.

## 6. Experiment history

| Experiment | Source | Model | Key configuration | Validation accuracy | Macro F1 | Class-1 recall | Training time |
|---|---|---|---|---:|---:|---:|---:|
| `exp_001` | Deterministic | One-Hot Logistic Regression | `C=1.0`, `max_iter=2000` | 0.896933 | 0.662328 | 0.271122 | 4.665 s |
| `exp_002` | Deterministic | One-Hot Extra Trees | 400 trees, balanced, leaf=2 | 0.828664 | 0.673415 | 0.595208 | 287.546 s |
| `exp_003` | LLM Planner | One-Hot Random Forest | 300 trees, depth=20, balanced | 0.758036 | 0.615528 | 0.638083 | 40.479 s |
| `exp_004` | LLM Planner | One-Hot Logistic Regression | `C=0.5`, `max_iter=3000` | **0.897375** | 0.651082 | 0.244641 | 4.724 s |

The LLM first selected Random Forest to test whether balanced class weights
could improve minority-class recall. After receiving the validation feedback,
it selected a more strongly regularized Logistic Regression configuration. The
final model was selected automatically by validation accuracy, as required by
the supplied metric.

## 7. Planner recovery behavior

The second LLM call consumed its output allowance but returned no usable
response. The SuperAgent:

1. recorded `PlannerResponseError` in `errors.jsonl`;
2. retained the failed call and its full token cost;
3. incremented the planner error counter to `1/2`;
4. retried the same next experiment ID without human intervention;
5. received and validated a usable plan on the next call.

No experiment records were deleted and no manual recovery decision was made.

## 8. LLM usage and cost

| Call | Input tokens | Output tokens | Cost (RMB) | Result |
|---|---:|---:|---:|---|
| 1 | 1,948 | 381 | 0.003472 | Proposed `exp_003` Random Forest |
| 2 | 2,476 | 800 | 0.005676 | Empty planner response; recorded and retried |
| 3 | 2,476 | 530 | 0.004596 | Proposed `exp_004` tuned Logistic Regression |
| **Total** | **6,900** | **1,711** | **0.013744** | Within RMB 20 task budget |

- Pricing snapshot date: `2026-10-21`
- Input price used: `RMB 1.00 / million tokens`
- Output price used: `RMB 4.00 / million tokens`
- Remaining budget: `RMB 19.986256`
- Budget ledger: `budget_ledger.jsonl`
- Full call log: `llm_calls.jsonl`

## 9. Automatic selection and test gate

- Selected experiment: `exp_004`
- Selected model: `one_hot_logistic_regression`
- Adapter: `tabular_one_hot_logistic`
- Selection metric: `accuracy`
- Validation accuracy: `0.8973754054851076`
- Automatically selected: `true`
- Test used for selection: `false`
- Solution locked before final evaluation: `true`
- Final test evaluation completed exactly once: `true`

The validation-to-test accuracy difference is approximately `0.003244`, which
indicates stable overall accuracy between the validation and test splits.

## 10. Final test results

| Metric | Value |
|---|---:|
| Accuracy | **0.8941315246240047** |
| Macro precision | 0.7686138541675105 |
| Macro recall / balanced accuracy | 0.6047269931823307 |
| Macro F1 | 0.6382236787948328 |
| Class-0 recall | 0.9824678577391885 |
| Class-1 recall | 0.22698612862547288 |
| Prediction time | 0.0865589 seconds |
| Test samples | 6,782 |

Confusion matrix, with rows as actual labels and columns as predictions:

```text
             predicted 0   predicted 1
actual 0          5884           105
actual 1           613           180
```

The majority-class-only accuracy for this test split is approximately 88.31%.
The selected model improves overall accuracy to 89.41%, but detects only 22.70%
of positive-class examples. This is consistent with optimizing accuracy on an
imbalanced dataset rather than optimizing macro F1 or minority recall.

## 11. Exported artifacts and integrity

- Final model: `D:\python_project\unseen_trial_bank_marketing\runs\blind_bank_subscription_trial_20261119_012410_908468\final\final_model.joblib`
- Test predictions CSV: `D:\python_project\unseen_trial_bank_marketing\runs\blind_bank_subscription_trial_20261119_012410_908468\final\test_predictions.csv`
- Test predictions NPY: `D:\python_project\unseen_trial_bank_marketing\runs\blind_bank_subscription_trial_20261119_012410_908468\final\test_predictions.npy`
- Test metrics: `D:\python_project\unseen_trial_bank_marketing\runs\blind_bank_subscription_trial_20261119_012410_908468\final\test_metrics.json`
- Model SHA-256: `d36b7f06a441422c68ba2fe7b3682b44307d08319c68457f85df7385354c9de7`
- Prediction CSV SHA-256: `b97081996ff6a8067d343cba1b01f6824b4cdcda19baaaf7065389d952d29a90`
- Prediction NPY SHA-256: `05c0bfbf6f034df3095cd2dec3dcafe99e723dd9839d7f6e053aad1576a98726`
- Metrics SHA-256: `7d1afbc9926b490c48b8fab9e76657f0ec877e7d33cf33dcc2d9706b0963f1d0`
- Final solution SHA-256: `3987168a8df9a92ce7b466ad4d9fdb9947fae38dc410b85dd11547a5417ad91e`
- Artifact verification: `passed`

Verification command:

```powershell
cd D:\python_project\superagent_unseen_test

D:\python_project\.venv\Scripts\python.exe `
  -m scripts.verify_final_artifacts `
  --run-directory D:\python_project\unseen_trial_bank_marketing\runs\blind_bank_subscription_trial_20261119_012410_908468
```

## 12. What this trial demonstrates

The frozen SuperAgent successfully completed the following on a dataset that
was not used during development:

1. modality and task-type detection;
2. autonomous data inspection and class-imbalance detection;
3. deterministic baseline experiments;
4. LLM-guided algorithm and hyperparameter selection;
5. validation-feedback-driven iteration;
6. automatic recovery from an unusable LLM response;
7. autonomous stopping;
8. automatic final-model selection;
9. test isolation until solution locking;
10. one-time final evaluation;
11. export of a reloadable prediction pipeline;
12. artifact and prediction integrity verification.

## 13. Limitations revealed by the trial

1. The current tabular pipeline converts values to categorical strings and
   applies One-Hot encoding. It does not fully exploit continuous numerical
   structure in high-cardinality integer features.
2. Only accuracy is currently accepted as the task metric. On this imbalanced
   dataset, the accuracy-optimal model has poor positive-class recall.
3. The algorithm search is autonomous but constrained to the allow-listed
   Logistic Regression, Extra Trees, and Random Forest adapters.
4. The system requires the data provider to supply a supported format and
   predefined train/validation/test splits.

## 14. Integrity statement

- No code or configuration was modified during the autonomous run.
- No model or hyperparameter was manually selected.
- No failed experiment or LLM call was removed.
- The same frozen Git commit was used throughout the trial.
- Test labels and metrics were not used for training, planning, tuning, or
  model selection.
- The final test set was evaluated only after automatic solution locking.
- The final model and result artifacts passed SHA-256 verification.
