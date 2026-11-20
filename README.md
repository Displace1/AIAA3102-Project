# AIAA3102 Final Project: Budgeted Autonomous Classification SuperAgent

This repository contains our final project for **AIAA3102 – Python Programming with Artificial Intelligence**. The project implements an auditable autonomous machine-learning agent that accepts a classification task, a pre-split dataset, and an evaluation metric, then completes the experiment workflow without human model selection.

Given an unfamiliar supported dataset, the SuperAgent can:

1. inspect the data and detect its modality;
2. establish deterministic baseline models;
3. ask a budget-limited LLM Experiment Planner for subsequent experiments;
4. validate every proposed adapter and hyperparameter before execution;
5. train and evaluate experiments in isolated worker processes;
6. use validation results to iterate or stop autonomously;
7. select and lock the best successful model automatically;
8. evaluate the locked model once on the held-out test set; and
9. export a reloadable model for inference on unlabeled data.

During an official autonomous run, students do not modify the code or configuration, tell the planner what to try next, remove failed results, or manually select the final model. Decisions and evidence are retained in machine-readable run artifacts.

> **Current scope:** classification with `accuracy`, `macro_f1`, or `balanced_accuracy` as the validation selection metric. The data provider must supply predefined training, validation, and test splits in one of the supported formats. Data inspection and modeling are performed by the Agent after the run starts; raw-data collection and split creation are outside the autonomous run.

## 1. Submission contents

The main implementation is in `superagent_project/`.

```text
python_project/
├── README.md                         # This submission guide
├── OFFICIAL_RUN_RECORD.md            # Frozen-version and official-result record
├── AIAA3102 - Final Project - Full Assignment Requirements.md
├── Released_Task1-In_Vehicle_Coupon_Recommendation/
├── Released_Task2-Headlines_CTR/
├── superagent_project/               # Main source repository
│   ├── main.py
│   ├── requirements.txt
│   ├── configs/
│   ├── adapters/
│   ├── superagent/
│   ├── scripts/
│   ├── tests/
│   └── runs/
└── unseen_trial_bank_marketing/
    └── UNSEEN_TRIAL_RECORD.md         # Independent unfamiliar-task trial
```

Local virtual environments, IDE settings, prediction smoke outputs, and API-key files are not project source and should not be included in a public submission archive.

## 2. System architecture

```text
Task JSON + pre-split dataset
              |
              v
      TaskConfig validation
              |
              v
 DataInspector / ImageInspector ------> data_profile.json
              |
              v
 deterministic baseline experiments
              |
              v
+-------------------- autonomous loop --------------------+
| validation history --> LLM Experiment Planner           |
|                               |                          |
|                               v                          |
|                         PlanValidator                    |
|                               |                          |
|                               v                          |
|                 isolated ExperimentExecutor              |
|          (timeout / CPU / RAM / GPU monitoring)          |
|                               |                          |
|                               v                          |
|                    validation metrics -------------------+
+----------------------------------------------------------+
              |
              v
 automatic best-model selection + solution_lock.json
              |
              v
 one-time FinalEvaluator --> exported model + test results
              |
              v
 FinalModelRunner --> predictions for new unlabeled data
```

### Design principles

- **Validation-driven decisions:** the planner and model selector receive validation results, not test metrics.
- **Held-out test protection:** final test evaluation is permitted only after `solution_lock.json` is created.
- **Allow-listed execution:** the LLM can choose only registered adapters and validated hyperparameters.
- **Complete audit trail:** successful runs, failures, planner errors, costs, and stop reasons are retained.
- **Resource protection:** each experiment runs in a child process with time, CPU-thread, and total-memory limits.
- **Recovery:** interrupted runs resume from `run_state.json` without repeating completed experiments.
- **Artifact integrity:** exported models and predictions are protected by SHA-256 hashes.

## 3. Supported task types

| Modality | Data format | Available model families |
| --- | --- | --- |
| Tabular classification | `npy_split` or `csv_split` | Type-aware Logistic Regression, Extra Trees, Random Forest |
| Single-text or text-pair classification | `csv_split` | Word TF-IDF + Logistic Regression, Character TF-IDF + Logistic Regression, Character TF-IDF + Linear SVM |
| Image classification | `image_directory_split` | Small CNN, ResNet18, MobileNetV3 Small |

For image tasks, the three split directories must contain identical class subdirectories:

```text
dataset/
├── train/
│   ├── cat/
│   └── dog/
├── validation/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

The image inspector records class distributions, dimensions, and color modes, rejects damaged images, and converts grayscale images to RGB during loading. Image training records train/validation accuracy, macro F1, generalization gap, epoch count, best epoch, training time, early stopping, and peak GPU memory. It falls back to CPU when CUDA is unavailable.

## 4. Frozen implementation and official results

The official Released Task runs used the same frozen implementation:

- Git tag: `final-superagent-v1`
- Git commit: `3a817efe8c598feb56ca45a4795569b809add620`
- Python: `3.14.6`
- Platform: Windows 11
- Selection metric: validation accuracy
- Per-task LLM budget: RMB 20.00

### Released Task results

| Task | Detected modality | Experiments | Automatically selected model | Validation accuracy | Test accuracy | LLM cost |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Task 1: In-Vehicle Coupon Recommendation | Tabular | 5 | One-Hot Extra Trees | 0.756158 | 0.761135 | RMB 0.012504 |
| Task 2: Headlines CTR | Text pair | 7 | Character TF-IDF Logistic Regression | 0.702128 | 0.681002 | RMB 0.029848 |

For both tasks:

- the autonomous stop reason was `no_validation_improvement`;
- all recorded experiments completed successfully;
- the final model was selected automatically from validation accuracy;
- the test set was not used for planning, tuning, or model selection;
- final test evaluation occurred only after solution locking; and
- exported artifacts passed SHA-256 verification.

The complete run IDs, timestamps, confusion matrices, hashes, and artifact paths are recorded in [`OFFICIAL_RUN_RECORD.md`](OFFICIAL_RUN_RECORD.md).

## 5. Environment setup

### 5.1 Tested environment

- Windows PowerShell
- Python 3.14.6
- NumPy 2.5.5
- scikit-learn 1.9.1
- PyTorch 2.11.0 with CUDA 12.8
- Torchvision 0.26.0 with CUDA 12.8
- NVIDIA GeForce RTX 5070 Ti for the tested CUDA runtime

### 5.2 Create the virtual environment

Open PowerShell in the submission root (the directory containing this README):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://pypi.org/simple `
  -r .\superagent_project\requirements.txt
```

The PyTorch CUDA wheel index is already declared in `requirements.txt`. A CUDA build can still run on CPU when no compatible GPU is available, although image training will be slower.

Do not write package commands as `https\://...`, and do not append `\` to a package name. In PowerShell, the line-continuation character is a backtick: `` ` ``.

### 5.3 Verify the installation

```powershell
python -m pip check
python -c "import sklearn, torch, torchvision; print('sklearn:', sklearn.__version__); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

The exact versions are pinned in [`superagent_project/requirements.txt`](superagent_project/requirements.txt).

## 6. Configure the LLM API

An API key is required only when a task uses `"planner_mode": "llm"`. Offline tests, final evaluation, artifact verification, and exported-model inference do not call the LLM.

Set the key only in the current PowerShell process:

```powershell
$env:AIGC_API_KEY = "YOUR_API_KEY"
```

If the key is stored in a private local file:

```powershell
$env:AIGC_API_KEY = (
  Get-Content -LiteralPath ".\api_key.txt" -Raw
).Trim()

if ([string]::IsNullOrWhiteSpace($env:AIGC_API_KEY)) {
  throw "AIGC_API_KEY is missing"
}
```

Never commit or submit `api_key.txt`, `.env`, or any source file containing a real key.

Test one API request from the project directory:

```powershell
cd .\superagent_project
python -m scripts.test_llm_connection
```

The endpoint and model are configured in `configs/llm.json`. The official price snapshot, accounting rules, and per-task budget are in `configs/model_pricing.json`.

## 7. Task configuration

Task files use the following structure:

```json
{
  "run_type": "development_run",
  "task": {
    "name": "my_unseen_classification_task",
    "metric": "accuracy"
  },
  "data": {},
  "execution": {
    "random_seed": 42,
    "evaluate_test": false,
    "planner_mode": "llm",
    "initial_deterministic_experiments": 2,
    "max_experiments": 8,
    "max_runtime_minutes": 180,
    "no_improvement_patience": 3,
    "minimum_improvement": 0.001,
    "maximum_planner_errors": 2,
    "maximum_same_experiment_errors": 2,
    "experiment_timeout_seconds": 1800,
    "maximum_memory_mb": 4096,
    "maximum_cpu_threads": 2,
    "resource_poll_interval_seconds": 0.2
  }
}
```

- `development_run` identifies development, smoke-test, and unfamiliar-task trials.
- `official_report_run` identifies a frozen-version run used in the final report.
- `evaluate_test` must remain `false`; the development controller is not permitted to evaluate the test set.
- `planner_mode` accepts `llm` or `deterministic`.
- patience and minimum improvement control autonomous stopping.
- timeout, memory, and CPU-thread values limit each individual experiment.

The included official Task 1 and Task 2 configuration files contain absolute paths from the machine used for the official runs. Keep those files unchanged when auditing the recorded run. To run the project after extracting it elsewhere, copy the relevant configuration and update only the copied data paths.

### 7.1 NPY tabular data

```json
"data": {
  "format": "npy_split",
  "train_features": "C:/data/task/C_train.npy",
  "train_labels": "C:/data/task/y_train.npy",
  "validation_features": "C:/data/task/C_val.npy",
  "validation_labels": "C:/data/task/y_val.npy",
  "test_features": "C:/data/task/C_test.npy",
  "test_labels": "C:/data/task/y_test.npy"
}
```

### 7.2 CSV tabular, single-text, or text-pair data

```json
"data": {
  "format": "csv_split",
  "train": "C:/data/task/train.csv",
  "validation": "C:/data/task/val.csv",
  "test": "C:/data/task/test.csv",
  "target_column": "label_pairwise",
  "columns": {
    "text": ["headline_a", "headline_b"]
  }
}
```

`columns` is optional but recommended. Declare one or two names under `text`, or declare every structured feature under `numeric` and `categorical`. Without a declaration, modality detection uses column content rather than a task or dataset name.

### 7.3 Image-directory data

```json
"data": {
  "format": "image_directory_split",
  "train_directory": "C:/data/images/train",
  "validation_directory": "C:/data/images/validation",
  "test_directory": "C:/data/images/test"
}
```

## 8. Run the autonomous pipeline

The examples below assume PowerShell is currently in `superagent_project/`:

```powershell
cd .\superagent_project
```

### 8.1 Run Task 1

```powershell
python .\main.py --config .\configs\task1.json
```

### 8.2 Run Task 2

```powershell
python .\main.py --config .\configs\task2.json
```

### 8.3 Store runs in a custom location

```powershell
python .\main.py `
  --config .\configs\task1.json `
  --runs-dir C:\experiment_results\task1
```

At completion, the program prints the exact run directory, for example:

```text
Run directory: C:\experiment_results\task1\released_task_1_20261119_005217_427548
```

Text such as `<run_id>` or `<timestamp>` in documentation is a placeholder. Do not type angle-bracket placeholders literally into PowerShell.

## 9. Inspect, evaluate, and verify a completed run

Store the actual printed directory in a PowerShell variable:

```powershell
$run = "C:\experiment_results\task1\released_task_1_ACTUAL_TIMESTAMP"
```

### 9.1 Inspect the automatic decision

```powershell
Get-Content -LiteralPath "$run\solution_lock.json"
Get-Content -LiteralPath "$run\stop_reason.json"
Get-Content -LiteralPath "$run\summary.json"
```

### 9.2 Perform the one-time final test evaluation

```powershell
python .\main.py `
  --config .\configs\task1.json `
  --final-evaluate-run $run
```

This mode does not call the LLM, execute new experiments, or reselect the model. It rejects an unlocked solution, a mismatched configuration, a replaced model artifact, or a run whose final evaluation has already been completed.

### 9.3 Verify exported artifacts

```powershell
python -m scripts.verify_final_artifacts `
  --run-directory $run
```

## 10. Resume an interrupted run

Each run atomically maintains `run_state.json`. Completed experiments are restored from their plan and result files. If a plan was saved but its experiment was interrupted, that experiment is rerun without requesting a replacement plan from the LLM.

```powershell
python .\main.py `
  --config .\configs\task1.json `
  --resume-run C:\experiment_results\task1\released_task_1_ACTUAL_TIMESTAMP
```

A run containing `solution_lock.json` is already complete and cannot be resumed. Resume events are appended to `resume_events.jsonl`, and interruption downtime is excluded from active runtime.

## 11. Run the exported model on new data

Inference does not require labels, a task configuration, an API key, or an LLM request. The run must first have a completed `final/` export.

### Tabular NPY input

```powershell
python .\main.py `
  --predict-run $run `
  --input-data C:\new_data\features.npy `
  --prediction-output-dir C:\predictions\task1_batch
```

### Text-pair CSV input

```powershell
python .\main.py `
  --predict-run $run `
  --input-data C:\new_data\headline_pairs.csv `
  --prediction-output-dir C:\predictions\task2_batch
```

The unlabeled CSV must contain the two feature columns stored in the exported text model.

### Image file or directory input

```powershell
python .\main.py `
  --predict-run $run `
  --input-data C:\new_images `
  --prediction-output-dir C:\predictions\images
```

Image inference writes:

```text
row_id,input_path,prediction,class_name
```

All inference modes create:

- `predictions.csv`
- `predictions.npy`
- `prediction_manifest.json`

The output directory must be empty or not yet exist. The runner verifies the final-model SHA-256 hash before loading it and deliberately refuses to overwrite previous predictions.

## 12. Run artifacts and audit evidence

Every autonomous run creates a separate directory containing the following evidence:

| File | Purpose |
| --- | --- |
| `run_manifest.json` | Task, configuration, run type, and run identity |
| `data_profile.json` | Agent-generated data inspection results |
| `plans.jsonl` | Validated experiment plans |
| `planner_decisions.jsonl` | LLM proposals and reasons |
| `experiments.jsonl` | Experiment states and validation results |
| `decisions.jsonl` | Controller loop decisions |
| `errors.jsonl` | Planner or experiment errors, when present |
| `llm_calls.jsonl` | Complete LLM call metadata and responses |
| `budget_ledger.jsonl` | Token usage and model cost |
| `run_state.json` | Recovery checkpoint |
| `stop_reason.json` | Autonomous stopping condition |
| `summary.json` | Automatically selected validation winner |
| `solution_lock.json` | Locked solution and selected artifact |
| `experiments/exp_*/` | Per-experiment plans, metrics, and models |

After final evaluation, the `final/` directory contains:

```text
final/
├── final_model.joblib      # Tabular or text model
├── final_model.pt          # Image model; mutually exclusive with joblib
├── test_predictions.csv
├── test_predictions.npy
├── test_metrics.json
├── final_solution.json
└── artifact_hashes.json
```

Failures are not deleted from the history. Timeout, memory-limit, worker-crash, and CUDA-out-of-memory errors are classified and returned to the planner. For image tasks, this allows later plans to reduce batch size, image size, or model scale.

## 13. Tests

Run the complete test suite from `superagent_project/`:

```powershell
python -m unittest discover -s tests -v
```

The Controller integration tests use Fake Planners and do not consume API quota. The suite covers planning, parameter validation, budget accounting, the autonomous feedback loop, resource limits, recovery, final evaluation, artifact verification, inference, image inspection, CPU image training, early stopping, CUDA fallback/OOM classification, and `.pt` model reload.

Run the offline Experiment Planner tests:

```powershell
python -m scripts.test_experiment_planner --task-config .\configs\task1.json
python -m scripts.test_experiment_planner --task-config .\configs\task2.json
```

Make one real budget-controlled planner request only when an API key is configured:

```powershell
python -m scripts.test_experiment_planner `
  --task-config .\configs\task1.json `
  --live
```

Create and run a self-contained image smoke task:

```powershell
python -m scripts.create_synthetic_image_dataset --force
python .\main.py --config .\configs\image_smoke.json
```

The image smoke configuration uses a deterministic planner and one Small CNN experiment, so it can validate the image pipeline without an external dataset or LLM request.

## 14. Independent unfamiliar-task trial

The frozen implementation was also evaluated on UCI Bank Marketing, a dataset not used to build the Released Task pipelines.

- Samples: 45,211
- Input features: 16
- Split: deterministic stratified 70/15/15, seed 42
- Autonomous experiments: 4
- Stop reason: `no_validation_improvement`
- Selected model: One-Hot Logistic Regression
- Validation accuracy: 0.897375
- One-time test accuracy: 0.894132
- Final model SHA-256: `d36b7f06a441422c68ba2fe7b3682b44307d08319c68457f85df7385354c9de7`
- Artifact verification: passed

The Agent completed inspection, experiment iteration, automatic selection, locking, final evaluation, and export without dataset-specific training code or human intervention during the run. This trial demonstrates supported unfamiliar-task operation; it does not claim that every possible unseen dataset will be optimally solved.

See [`unseen_trial_bank_marketing/UNSEEN_TRIAL_RECORD.md`](unseen_trial_bank_marketing/UNSEEN_TRIAL_RECORD.md) for provenance, split hashes, complete experiment history, cost accounting, final metrics, limitations, and integrity evidence.

## 15. Reproducing the official artifact checks

On the original submission machine, the official artifacts can be checked with:

```powershell
cd D:\python_project\superagent_project

D:\python_project\.venv\Scripts\python.exe -m scripts.verify_final_artifacts `
  --run-directory D:\python_project\superagent_project\runs\released_task_1_20261119_005217_427548

D:\python_project\.venv\Scripts\python.exe -m scripts.verify_final_artifacts `
  --run-directory D:\python_project\superagent_project\runs\released_task_2_20261119_005516_301668
```

These commands verify the locked selection state, prediction consistency, and stored SHA-256 hashes. The detailed expected values are in `OFFICIAL_RUN_RECORD.md`.

## 16. Official-run protocol

Before beginning an official report run, freeze one implementation version:

```powershell
git status --porcelain
git add .
git commit -m "Freeze final SuperAgent implementation"
git tag final-superagent-v1
git rev-parse HEAD
git rev-parse "final-superagent-v1^{commit}"
```

For Task 1 and Task 2:

- use the same frozen commit;
- set `run_type` to `official_report_run` before freezing;
- do not modify code or configuration after the run starts;
- do not tell the planner which model or hyperparameter to use;
- do not manually select the final model;
- do not delete failures or report only a preferred rerun; and
- evaluate the test split only after `solution_lock.json` is created.

## 17. Known limitations

- The supported primary metrics are `accuracy`, `macro_f1`, and `balanced_accuracy`.
- Datasets must be supplied as predefined train/validation/test splits in a supported format.
- The planner searches only within the registered adapter and hyperparameter allow-list; it does not execute arbitrary generated Python code.
- The tabular pipeline is strongest for categorical or string-valued features and does not yet include specialized continuous-feature engineering.
- For imbalanced data, accuracy-based selection may produce weak minority-class recall. Reports should also inspect macro F1, balanced accuracy, per-class recall, and the confusion matrix.
- `pretrained: true` for ResNet18 or MobileNet may download weights on first use. Offline runs should select a non-pretrained configuration.
- CPU memory is monitored over the experiment process tree. GPU usage is measured through PyTorch peak-memory statistics, and CUDA OOM errors are classified separately.

## 18. Troubleshooting

### `ModuleNotFoundError: No module named 'superagent'`

Change into the directory containing `main.py` and `superagent/` before running commands:

```powershell
cd .\superagent_project
python .\main.py --help
```

### `AIGC_API_KEY is not set`

The selected configuration uses `planner_mode: "llm"`. Set `AIGC_API_KEY` in the same PowerShell process that launches the run. Do not place the key in the repository.

### PowerShell reports illegal characters for `<run_id>`

`<run_id>` is a placeholder. Replace it with the directory printed by the program. To locate recent runs:

```powershell
Get-ChildItem .\runs -Directory |
  Sort-Object LastWriteTime -Descending
```

### `Prediction output directory is not empty`

Use a new output directory or move the old predictions elsewhere. Refusing to overwrite an existing batch is an intentional safety check.

### `No matching distribution found`

Confirm that URLs use `https://`, not `https\://`, and that package names do not end in `\`. Use the official index if a configured mirror is incomplete:

```powershell
python -m pip install --index-url https://pypi.org/simple `
  -r .\superagent_project\requirements.txt
```

## 19. Submission integrity statement

The reported official results were produced by the frozen SuperAgent version identified above. Both Released Tasks used autonomous validation-driven experimentation and automatic final-model selection. Test labels were not used by the planner or selector, and each test split was evaluated only after its solution was locked. The final artifacts for both tasks passed the provided integrity verifier.

For assessment, the README explains how to operate the system; the task configurations, run directories, official record, planner logs, cost ledger, solution locks, final metrics, and hashes provide the evidence that the system actually operated autonomously.
