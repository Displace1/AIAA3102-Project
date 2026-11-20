# AIAA3102 SuperAgent Official Run Record

Recorded on: 2026-11-19 (Asia/Shanghai)

## Frozen SuperAgent Version

- Git tag: `final-superagent-v1`
- Frozen commit: `3a817efe8c598feb56ca45a4795569b809add620`
- Tag target commit: `3a817efe8c598feb56ca45a4795569b809add620`
- Frozen commit message: `Freeze final SuperAgent implementation`
- Repository: `D:\python_project\superagent_project`
- Official Python executable: `D:\python_project\.venv\Scripts\python.exe`
- Python version: `3.14.6`
- Platform: `Windows-11-10.0.26200-SP0`
- NumPy: `2.5.5`
- scikit-learn: `1.9.1`
- joblib: `1.6.0`

Both Released Tasks were run using the same frozen Git commit and configurations
marked with `run_type: official_report_run`.

## Released Task 1

- Task name: `released_task_1`
- Modality: `tabular`
- Planner mode: `llm`
- Run type: `official_report_run`
- Run ID: `released_task_1_20261119_005217_427548`
- Run directory: `D:\python_project\superagent_project\runs\released_task_1_20261119_005217_427548`
- Started at: `2026-11-19T00:52:17.428271+08:00`
- Locked at: `2026-11-19T00:53:35.461055+08:00`
- Final evaluation at: `2026-11-19T00:58:58.407126+08:00`
- Configuration SHA-256: `5d65e3d0c644c17f273a430c4fade980129e54b3dd9ce5da676562286c355928`
- Completed experiments: `5`
- Successful experiments: `5`
- Failed experiments: `0`
- Autonomous stop reason: `no_validation_improvement`
- Autonomous-run elapsed time: `78.0192131 seconds`
- LLM cost: `RMB 0.01250400`
- Remaining LLM budget: `RMB 19.98749600`
- Selected experiment: `exp_002`
- Selected model: `one_hot_extra_trees`
- Adapter: `tabular_extra_trees`
- Selection metric: `accuracy`
- Validation accuracy: `0.7561576354679803`
- Test accuracy: `0.7611351990540007`
- Test macro precision: `0.7566336877738358`
- Test macro recall: `0.7558122597356538`
- Test macro F1: `0.7561988110461576`
- Test samples: `2537`
- Test confusion matrix: `[[785, 310], [296, 1146]]`
- Automatically selected: `true`
- Test used for model selection: `false`
- Final evaluation completed: `true`
- Final model: `D:\python_project\superagent_project\runs\released_task_1_20261119_005217_427548\final\final_model.joblib`
- Final model SHA-256: `fd842676bc23c83132a0f21dec883fb2db2e33581490401beca3a4b821ced6c2`
- Test predictions CSV: `D:\python_project\superagent_project\runs\released_task_1_20261119_005217_427548\final\test_predictions.csv`
- Test metrics: `D:\python_project\superagent_project\runs\released_task_1_20261119_005217_427548\final\test_metrics.json`
- Artifact verification: `passed`

## Released Task 2

- Task name: `released_task_2`
- Modality: `text_pair`
- Planner mode: `llm`
- Run type: `official_report_run`
- Run ID: `released_task_2_20261119_005516_301668`
- Run directory: `D:\python_project\superagent_project\runs\released_task_2_20261119_005516_301668`
- Started at: `2026-11-19T00:55:16.302398+08:00`
- Locked at: `2026-11-19T00:57:49.987486+08:00`
- Final evaluation at: `2026-11-19T00:59:09.051940+08:00`
- Configuration SHA-256: `5aca7278668645388139d260f333c82cb51f5033bdc154e9c35455f140808e4d`
- Completed experiments: `7`
- Successful experiments: `7`
- Failed experiments: `0`
- Autonomous stop reason: `no_validation_improvement`
- Autonomous-run elapsed time: `153.6710315 seconds`
- LLM cost: `RMB 0.02984800`
- Remaining LLM budget: `RMB 19.97015200`
- Selected experiment: `exp_005`
- Selected model: `pairwise_char_tfidf_logistic_regression`
- Adapter: `text_pair_char_tfidf_logistic`
- Selection metric: `accuracy`
- Validation accuracy: `0.7021276595744681`
- Test accuracy: `0.6810024752475248`
- Test macro precision: `0.6816756891123019`
- Test macro recall: `0.6814046432277292`
- Test macro F1: `0.6809511321321433`
- Test samples: `3232`
- Test confusion matrix: `[[1080, 563], [468, 1121]]`
- Automatically selected: `true`
- Test used for model selection: `false`
- Final evaluation completed: `true`
- Final model: `D:\python_project\superagent_project\runs\released_task_2_20261119_005516_301668\final\final_model.joblib`
- Final model SHA-256: `58336849521b81461b5a8761caed1bfdfd0c16e64b9b734acd61170615bb9ebc`
- Test predictions CSV: `D:\python_project\superagent_project\runs\released_task_2_20261119_005516_301668\final\test_predictions.csv`
- Test metrics: `D:\python_project\superagent_project\runs\released_task_2_20261119_005516_301668\final\test_metrics.json`
- Artifact verification: `passed`

## Budget Summary

- Task 1 LLM cost: `RMB 0.01250400`
- Task 2 LLM cost: `RMB 0.02984800`
- Combined LLM cost: `RMB 0.04235200`
- Per-task limit: `RMB 20.00`
- Task 1 within budget: `yes`
- Task 2 within budget: `yes`

## Reverification Commands

```powershell
cd D:\python_project\superagent_project

D:\python_project\.venv\Scripts\python.exe -m scripts.verify_final_artifacts `
  --run-directory D:\python_project\superagent_project\runs\released_task_1_20261119_005217_427548

D:\python_project\.venv\Scripts\python.exe -m scripts.verify_final_artifacts `
  --run-directory D:\python_project\superagent_project\runs\released_task_2_20261119_005516_301668
```

## Integrity Statement

- Both tasks used the same frozen SuperAgent version.
- Both solutions were selected automatically using validation accuracy.
- The designated test sets were not used for training, tuning, experiment
  planning, or final model selection.
- Each test set was evaluated only after the corresponding solution was locked.
- Both exported final models and result artifacts passed SHA-256 verification.
