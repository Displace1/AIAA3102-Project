# Budgeted Autonomous SuperAgent

This project is the first reproducible stage of the General Algorithm
SuperAgent. It detects the data modality from the input structure, runs a
compatible experiment plans, validates their parameter schemas, evaluates the
models only on the validation split, and records auditable artifacts.

The main controller deliberately does not read test labels. It currently uses
two deterministic initial plans, then a budgeted LLM planner proposes validated
experiments until a configured stopping condition is reached. Failed experiments
and detailed validation diagnostics are fed back to the next planner decision.

## Environment

From `D:\python_project`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\superagent_project\requirements.txt
```

## Run Task 1

```powershell
cd D:\python_project\superagent_project
python .\main.py --config .\configs\task1.json
```

The initial run compares one-hot logistic regression with one-hot Extra Trees.
The planner can also select the allow-listed Random Forest candidate.

## Run Task 2

```powershell
cd D:\python_project\superagent_project
python .\main.py --config .\configs\task2.json
```

The initial run compares pairwise word TF-IDF with pairwise character TF-IDF.
The planner can also select the allow-listed character TF-IDF Linear SVM.

## Run artifacts

Each execution creates a new directory under `runs` containing:

- `run_manifest.json`
- `data_profile.json`
- `experiments.jsonl`
- `plans.jsonl`
- `decisions.jsonl`
- `solution_lock.json`
- `stop_reason.json`
- `planner_decisions.jsonl`
- `errors.jsonl` (when errors occur)
- `summary.json`
- model and prediction artifacts under `experiments/exp_*/`

Task names are used only for run identification. Adapter selection is based on
the detected data modality, never on the task or dataset name.

Every experiment is represented by an `ExperimentPlan`. The plan validator
rejects unknown adapters, modality mismatches, missing or unknown parameters,
out-of-range values, invalid n-gram ranges, duplicate IDs, and duplicate
configurations before training starts.

Successful experiments record training and validation accuracy, macro precision,
recall and F1, confusion matrix, per-class recall, generalization gap, prediction
time, feature count, model size, and deterministic diagnostic warnings.

The controller stops on experiment/runtime/patience limits, planner request,
budget exhaustion, repeated planner errors, or repeated training-error classes.
API calls retry only transient network failures and HTTP 429/5xx responses with
bounded exponential backoff.

## Resume an interrupted run

Every run maintains an atomically written `run_state.json`. Completed experiments
are restored from their `plan.json` and `experiment.json` files. A plan without an
experiment result is treated as interrupted and is executed again without asking
the LLM to create another plan. The existing budget ledger remains authoritative.

```powershell
python .\main.py `
  --config .\configs\task1.json `
  --resume-run .\runs\released_task_1_<timestamp>
```

Runs containing `solution_lock.json` are already complete and cannot be resumed.
Resume events are appended to `resume_events.jsonl`; downtime is not counted as
active runtime.

## One-time final test evaluation

After an autonomous run has locked its selected model, evaluate that model once
on the designated test split:

```powershell
python .\main.py `
  --config .\configs\task1.json `
  --final-evaluate-run .\runs\released_task_1_<timestamp>
```

This mode does not call the LLM or run new experiments. It rejects unlocked,
manually selected, configuration-mismatched, or previously evaluated runs. The
locked model is exported under `final/` together with test predictions, metrics,
and SHA-256 integrity hashes. Test metrics are never returned to the planner.

## Run the exported model on new data

Final inference is self-contained and does not require a task configuration,
labels, an API key, or an LLM call:

```powershell
python .\main.py `
  --predict-run .\runs\<completed_run> `
  --input-data D:\new_data\features.npy `
  --prediction-output-dir .\predictions\new_batch
```

For text-pair models, `--input-data` is a CSV containing the two feature columns
stored in the exported model. The output directory receives `predictions.csv`,
`predictions.npy`, and `prediction_manifest.json`. The runner verifies the model
SHA-256 before loading it and refuses to overwrite a non-empty output directory.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The suite uses a Fake Planner for the Controller integration test and does not
consume API quota.

## Test the LLM Experiment Planner

The offline tests do not call an API or consume quota:

```powershell
python -m scripts.test_experiment_planner --task-config .\configs\task1.json
python -m scripts.test_experiment_planner --task-config .\configs\task2.json
```

To make one real budgeted planner call, first set `AIGC_API_KEY`, then run:

```powershell
python -m scripts.test_experiment_planner --task-config .\configs\task1.json --live
```

The planner receives only a data profile, validation experiment history,
remaining budget, and allow-listed adapter schemas. The program assigns the
experiment ID, parses the JSON response, rejects repeated configurations, and
runs `PlanValidator` before returning an executable plan. The smoke-test script
does not execute the proposed experiment.
