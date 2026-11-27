import json
import time
import os
from datetime import datetime
from pathlib import Path

from superagent.data_inspector import DataInspector
from superagent.executor import ExperimentExecutor
from superagent.experiment_registry import (
    ExperimentRegistry,
)
from superagent.llm.budget_manager import (
    BudgetError,
    BudgetExceededError,
)
from superagent.llm.client import LLMClientError
from superagent.llm.experiment_planner import (
    PlannerError,
)
from superagent.llm.factory import (
    build_experiment_planner,
)
from superagent.logger import RunLogger
from superagent.model_selector import (
    select_best_model,
)
from superagent.selection_labels import (
    LabelLoadError,
    load_validation_labels,
)
from superagent.cross_validation import cv_lock_blocks
from superagent.metrics import metric_direction, result_metric_value
from superagent.plan_validator import (
    PlanValidationError,
    PlanValidator,
)
from superagent.recovery import RunRecovery, RunState
from superagent.stop_policy import StopPolicy
from superagent.task_config import load_task_config
from superagent.throughput_cache import (
    ThroughputCache,
    attach_measured_throughput,
    measurement_from_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _format_metric_value(value):
    # A selection metric can legitimately be missing (log_loss on a model with
    # no predict_proba), and the run keeps going so the remaining experiments
    # still get a chance to report it.
    if value is None:
        return "not reported"
    return f"{float(value):.6f}"


class SuperAgentController:
    def __init__(
        self,
        config_path,
        runs_dir,
        planner_factory=None,
        resume_run_directory=None,
    ):
        self.config_path = config_path
        self.runs_dir = runs_dir
        self.resume_run_directory = resume_run_directory

        # Allows unit tests to inject a Fake Planner.
        self.planner_factory = (
            planner_factory
            or build_experiment_planner
        )

        # Measured per-epoch costs, shared across runs of the same task.  Filled
        # lazily by ``_throughput_cache()`` so tests can construct a controller
        # without touching the filesystem.
        self._throughput_cache_instance = None

    def run(self):
        session_started = time.perf_counter()

        print("[1/7] Loading task configuration")

        config = load_task_config(
            self.config_path
        )

        if (
            config.planner_mode == "llm"
            and not os.getenv("AIGC_API_KEY")
        ):
            raise RuntimeError(
                "planner_mode is 'llm', but "
                "AIGC_API_KEY is not set. Run: "
                "$env:AIGC_API_KEY = "
                "(Get-Content -LiteralPath "
                "'D:\\python_project\\api_key.txt' "
                "-Raw).Trim()"
            )

        recovery = RunRecovery()
        recovered = None
        if self.resume_run_directory is not None:
            recovered = recovery.load(
                run_directory=self.resume_run_directory,
                config=config,
            )

        logger = RunLogger(
            runs_dir=self.runs_dir,
            task_name=config.name,
            config_path=self.config_path,
            run_type=config.run_type,
            resume_run_directory=self.resume_run_directory,
        )

        accumulated_runtime = (
            recovered.accumulated_runtime_seconds
            if recovered is not None
            else 0.0
        )
        resume_count = (
            recovered.resume_count + 1
            if recovered is not None
            else 0
        )
        run_started = session_started - accumulated_runtime

        print("[2/7] Inspecting data")

        profile = DataInspector().inspect(
            config
        )

        # Measured throughput from earlier experiments (this run or a previous
        # one on the same task) travels inside the profile, so the plan
        # validator can gate on real numbers instead of calibrated constants.
        profile = attach_measured_throughput(
            profile,
            self._throughput_cache().entries(),
        )

        logger.write_json(
            "data_profile.json",
            profile,
        )

        print(
            "      Detected modality: "
            f"{profile['modality']}"
        )

        registry = ExperimentRegistry(
            random_seed=config.random_seed
        )

        validator = PlanValidator()
        executor = ExperimentExecutor()

        all_plans = list(recovered.plans) if recovered is not None else []
        results = list(recovered.results) if recovered is not None else []

        incomplete_plan = (
            recovered.incomplete_plan if recovered is not None else None
        )
        self._write_run_checkpoint(
            recovery=recovery,
            logger=logger,
            config=config,
            all_plans=all_plans,
            results=results,
            in_progress_experiment_id=(
                incomplete_plan.experiment_id
                if incomplete_plan is not None
                else None
            ),
            run_started=run_started,
            resume_count=resume_count,
            status="running",
        )

        if recovered is not None:
            print(
                "      Recovered "
                f"{len(results)} completed experiments; "
                f"resume count: {resume_count}"
            )

        if incomplete_plan is not None:
            validator.validate_against_history(
                plan=incomplete_plan,
                data_profile=profile,
                existing_plans=all_plans,
            )
            print(
                "      Resuming incomplete experiment: "
                f"{incomplete_plan.experiment_id}"
            )
            result = self._execute_and_record_plan(
                plan=incomplete_plan,
                plan_source=(
                    recovered.incomplete_plan_source or "recovered_plan"
                ),
                registry=registry,
                executor=executor,
                config=config,
                profile=profile,
                logger=logger,
                record_plan=False,
                allow_existing=True,
                checkpoint_before_execute=lambda experiment_id: (
                    self._write_run_checkpoint(
                        recovery=recovery,
                        logger=logger,
                        config=config,
                        all_plans=all_plans,
                        results=results,
                        in_progress_experiment_id=experiment_id,
                        run_started=run_started,
                        resume_count=resume_count,
                        status="running",
                    )
                ),
            )
            all_plans.append(incomplete_plan)
            results.append(result)
            self._write_run_checkpoint(
                recovery=recovery,
                logger=logger,
                config=config,
                all_plans=all_plans,
                results=results,
                in_progress_experiment_id=None,
                run_started=run_started,
                resume_count=resume_count,
                status="running",
            )

        print(
            "[3/7] Running initial "
            "deterministic experiments"
        )

        default_plans = (
            registry.default_plans_for(profile)
        )

        initial_count = (
            config.initial_deterministic_experiments
        )

        if initial_count > len(default_plans):
            raise ValueError(
                "initial_deterministic_experiments "
                f"is {initial_count}, but only "
                f"{len(default_plans)} default plans "
                "are available."
            )

        # A baseline that violates a hard limit (for example an image plan whose
        # estimated training time exceeds the experiment timeout) must not abort
        # the whole run: the LLM planner can still propose something feasible.
        # Skip it, record why, and continue with whatever survives.
        initial_plans = default_plans[
            :initial_count
        ]

        validated_initial_plans = []
        seen_signatures = set()
        for plan in initial_plans:
            try:
                validated_plan = validator.validate(
                    plan,
                    profile,
                )
            except PlanValidationError as error:
                print(
                    f"      {plan.experiment_id}: skipped; {error}"
                )
                logger.append_jsonl(
                    "decisions.jsonl",
                    {
                        "timestamp": (
                            datetime.now()
                            .astimezone()
                            .isoformat()
                        ),
                        "decision": (
                            "skip_deterministic_experiment"
                        ),
                        "details": {
                            "experiment_id": plan.experiment_id,
                            "adapter_name": plan.adapter_name,
                            "hyperparameters": (
                                plan.hyperparameters
                            ),
                            "reason": str(error),
                        },
                    },
                )
                continue

            signature = json.dumps(
                {
                    "adapter": validated_plan.adapter_name,
                    "hyperparameters": (
                        validated_plan.hyperparameters
                    ),
                },
                sort_keys=True,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            validated_initial_plans.append(validated_plan)

        initial_plans = validated_initial_plans

        if not initial_plans:
            message = (
                "No deterministic baseline plan fits the configured "
                "resource limits."
            )
            if config.planner_mode != "llm":
                raise ValueError(
                    f"{message} Switch planner_mode to llm or relax the "
                    "limits."
                )
            print(
                f"      Warning: {message} "
                "The planner will propose the first experiment."
            )

        for plan in initial_plans:
            if any(
                result.experiment_id == plan.experiment_id
                for result in results
            ):
                print(
                    f"      {plan.experiment_id}: already completed; skipping"
                )
                continue
            result = (
                self._execute_and_record_plan(
                    plan=plan,
                    plan_source="deterministic",
                    registry=registry,
                    executor=executor,
                    config=config,
                    profile=profile,
                    logger=logger,
                    checkpoint_before_execute=lambda experiment_id: (
                        self._write_run_checkpoint(
                            recovery=recovery,
                            logger=logger,
                            config=config,
                            all_plans=all_plans,
                            results=results,
                            in_progress_experiment_id=experiment_id,
                            run_started=run_started,
                            resume_count=resume_count,
                            status="running",
                        )
                    ),
                )
            )

            all_plans.append(plan)
            results.append(result)
            self._write_run_checkpoint(
                recovery=recovery,
                logger=logger,
                config=config,
                all_plans=all_plans,
                results=results,
                in_progress_experiment_id=None,
                run_started=run_started,
                resume_count=resume_count,
                status="running",
            )

        planner = None
        budget_manager = None
        stop_reason = None

        if config.planner_mode == "llm":
            print(
                "[4/7] Initializing budgeted "
                "LLM planner"
            )

            planner, budget_manager = (
                self.planner_factory(
                    project_root=PROJECT_ROOT,
                    run_directory=logger.run_dir,
                )
            )

            stop_policy = StopPolicy(
                max_experiments=(
                    config.max_experiments
                ),
                max_runtime_seconds=(
                    config.max_runtime_minutes
                    * 60
                ),
                no_improvement_patience=(
                    config.no_improvement_patience
                ),
                minimum_improvement=(
                    config.minimum_improvement
                ),
                metric=config.metric,
            )

            print(
                "[5/7] Starting autonomous "
                "experiment loop"
            )

            stop_reason = (
                self._run_autonomous_loop(
                    config=config,
                    profile=profile,
                    registry=registry,
                    validator=validator,
                    executor=executor,
                    planner=planner,
                    budget_manager=(
                        budget_manager
                    ),
                    stop_policy=stop_policy,
                    logger=logger,
                    all_plans=all_plans,
                    results=results,
                    run_started=run_started,
                    resume_count=resume_count,
                )
            )

        else:
            print(
                "[4/7] Deterministic planner "
                "mode selected"
            )

            stop_reason = (
                "deterministic_mode_completed"
            )

        print(
            "[6/7] Selecting best model "
            f"by validation {config.metric}"
        )

        elapsed_seconds = (
            time.perf_counter() - run_started
        )

        if budget_manager is not None:
            llm_cost_rmb = str(
                budget_manager.spent_rmb()
            )
            remaining_budget_rmb = str(
                budget_manager.remaining_rmb()
            )
        else:
            llm_cost_rmb = "0.00000000"
            remaining_budget_rmb = None

        stop_summary = {
            "run_type": config.run_type,
            "reason": stop_reason,
            "completed_experiments": len(
                results
            ),
            "successful_experiments": sum(
                result.status == "success"
                for result in results
            ),
            "failed_experiments": sum(
                result.status == "failed"
                for result in results
            ),
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "llm_cost_rmb": llm_cost_rmb,
            "remaining_budget_rmb": (
                remaining_budget_rmb
            ),
        }

        logger.write_json(
            "stop_reason.json",
            stop_summary,
        )

        successful_results = [
            result for result in results if result.status == "success"
        ]
        if not successful_results:
            failure_summary = {
                "task_name": config.name,
                "run_type": config.run_type,
                "status": "failed",
                "modality": profile["modality"],
                "planner_mode": config.planner_mode,
                "selection_metric": config.metric,
                "selected_experiment": None,
                "selected_model": None,
                "validation_metric_value": None,
                "completed_experiments": len(results),
                "successful_experiments": 0,
                "failed_experiments": len(results),
                "stop_reason": "all_experiments_failed",
                "loop_stop_reason": stop_reason,
                "failures": [
                    {
                        "experiment_id": result.experiment_id,
                        "model_name": result.model_name,
                        "error_type": result.error_type,
                        "raw_error_type": result.raw_error_type,
                        "error_message": result.error_message,
                    }
                    for result in results
                ],
                "llm_cost_rmb": llm_cost_rmb,
                "remaining_budget_rmb": remaining_budget_rmb,
                "run_directory": str(logger.run_dir),
                "test_evaluated": False,
            }
            logger.write_json("summary.json", failure_summary)
            logger.write_json(
                "stop_reason.json",
                {
                    **stop_summary,
                    "reason": "all_experiments_failed",
                    "loop_stop_reason": stop_reason,
                },
            )
            (logger.run_dir / "final").mkdir(parents=True, exist_ok=True)
            self._write_run_checkpoint(
                recovery=recovery,
                logger=logger,
                config=config,
                all_plans=all_plans,
                results=results,
                in_progress_experiment_id=None,
                run_started=run_started,
                resume_count=resume_count,
                status="failed",
                stop_reason="all_experiments_failed",
            )
            print("[7/7] Saving failed run summary")
            return failure_summary

        # Record where each experiment's validation predictions live so selection
        # can run paired tests; the path is derived from the run layout rather
        # than threaded through every adapter.
        for result in results:
            if getattr(result, "val_predictions_path", None):
                continue
            candidate = (logger.run_dir / "experiments" / result.experiment_id
                         / "val_predictions.npy")
            if candidate.is_file():
                result.val_predictions_path = str(candidate)

        # Selection's paired test needs the labels those predictions were scored
        # against. Missing labels degrade to the old behaviour (argmax only)
        # rather than failing the run.
        selection_labels = None
        label_skip = None
        try:
            selection_labels = load_validation_labels(config, profile)
        except LabelLoadError as error:
            label_skip = str(error)

        best = select_best_model(
            results, metric=config.metric, validation_labels=selection_labels)
        if label_skip and not getattr(best, "selection_test", None):
            best.selection_test = {
                "verdict": "no_comparison",
                "skipped_reason": label_skip,
            }
        selected_metric_value = result_metric_value(best, config.metric)
        selection_warnings = [
            warning
            for warning in (best.diagnosis or [])
            if warning != "no_obvious_metric_warning"
        ]

        selection = {
            "run_type": config.run_type,
            "selected_experiment": (
                best.experiment_id
            ),
            "selected_model": best.model_name,
            "selection_metric": (
                config.metric
            ),
            "validation_metric_value": selected_metric_value,
            "validation_accuracy": (
                best.validation_accuracy
            ),
            "validation_macro_f1": best.f1_macro,
            "validation_balanced_accuracy": best.balanced_accuracy,
            "validation_roc_auc": best.roc_auc,
            "validation_log_loss": best.log_loss,
            "decision_rule": getattr(best, "decision_rule", None),
            "selection_test": getattr(best, "selection_test", None),
            "artifact_path": (
                best.artifact_path
            ),
            "selected_automatically": True,
            "test_used_for_selection": False,
            "selection_rule": (
                f"{'Lowest' if metric_direction(config.metric) < 0 else 'Highest'} "
                f"validation {config.metric}; "
                "shorter training time breaks "
                "exact ties."
            ),
            "stop_reason": stop_reason,
            "total_experiments": len(
                results
            ),
            "llm_cost_rmb": llm_cost_rmb,
            "warnings": selection_warnings,
        }

        # CV is opt-in per experiment. cv_lock_blocks returns {} when nothing
        # computed CV, so a CV-free run writes a lock identical to before.
        selection.update(cv_lock_blocks(results, best))

        logger.append_jsonl(
            "decisions.jsonl",
            {
                "timestamp": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "decision": (
                    "select_final_candidate"
                ),
                "details": selection,
            },
        )

        if selection_warnings:
            logger.append_jsonl(
                "decisions.jsonl",
                {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "decision": "selected_model_warning",
                    "details": {
                        "experiment_id": best.experiment_id,
                        "warnings": selection_warnings,
                        "validation_accuracy": best.validation_accuracy,
                        "majority_class_accuracy": (
                            best.majority_class_accuracy
                        ),
                        "class_recall": best.class_recall,
                    },
                },
            )

        logger.write_json(
            "solution_lock.json",
            {
                **selection,
                "solution_locked": True,
                "locked_at": (
                    datetime.now()
                    .astimezone()
                    .isoformat()
                ),
                "test_evaluation_completed": (
                    False
                ),
            },
        )

        print("[7/7] Saving run summary")

        summary = {
            "task_name": config.name,
            "run_type": config.run_type,
            "modality": profile["modality"],
            "planner_mode": (
                config.planner_mode
            ),
            "selected_experiment": (
                best.experiment_id
            ),
            "selected_model": (
                best.model_name
            ),
            "selection_metric": config.metric,
            "validation_metric_value": selected_metric_value,
            "validation_accuracy": (
                best.validation_accuracy
            ),
            "validation_macro_f1": best.f1_macro,
            "validation_balanced_accuracy": best.balanced_accuracy,
            "validation_roc_auc": best.roc_auc,
            "validation_log_loss": best.log_loss,
            "completed_experiments": len(
                results
            ),
            "stop_reason": stop_reason,
            "llm_cost_rmb": llm_cost_rmb,
            "remaining_budget_rmb": (
                remaining_budget_rmb
            ),
            "run_directory": str(
                logger.run_dir
            ),
            "test_evaluated": False,
            "warnings": selection_warnings,
        }

        logger.write_json(
            "summary.json",
            summary,
        )

        self._write_run_checkpoint(
            recovery=recovery,
            logger=logger,
            config=config,
            all_plans=all_plans,
            results=results,
            in_progress_experiment_id=None,
            run_started=run_started,
            resume_count=resume_count,
            status="completed",
            stop_reason=stop_reason,
        )

        return summary

    def _run_autonomous_loop(
        self,
        config,
        profile,
        registry,
        validator,
        executor,
        planner,
        budget_manager,
        stop_policy,
        logger,
        all_plans,
        results,
        run_started,
        resume_count,
    ):
        planner_error_count = 0
        experiment_error_counts = {}
        for previous_result in results:
            if previous_result.status == "failed":
                error_key = self._experiment_error_key(
                    previous_result,
                    all_plans,
                )
                experiment_error_counts[error_key] = (
                    experiment_error_counts.get(error_key, 0) + 1
                )

        while True:
            if any(
                count >= config.maximum_same_experiment_errors
                for count in experiment_error_counts.values()
            ):
                return "maximum_same_experiment_errors_reached"

            elapsed_seconds = (
                time.perf_counter()
                - run_started
            )

            stop_decision = (
                stop_policy.evaluate(
                    results=results,
                    elapsed_seconds=(
                        elapsed_seconds
                    ),
                )
            )

            if stop_decision.should_stop:
                print(
                    "      Autonomous loop stopped: "
                    f"{stop_decision.reason}"
                )
                return stop_decision.reason

            next_experiment_id = self._next_experiment_id(all_plans)

            print(
                "      Planning "
                f"{next_experiment_id}; "
                "remaining budget: "
                f"{budget_manager.remaining_rmb()} RMB"
            )

            try:
                decision = planner.plan_next(
                    data_profile=profile,
                    experiment_history=results,
                    existing_plans=all_plans,
                    remaining_budget_rmb=(
                        budget_manager
                        .remaining_rmb()
                    ),
                    next_experiment_id=(
                        next_experiment_id
                    ),
                )

                logger.append_jsonl(
                    "planner_decisions.jsonl",
                    {
                        "timestamp": (
                            datetime.now()
                            .astimezone()
                            .isoformat()
                        ),
                        **decision.to_dict(),
                    },
                )

                planner_error_count = 0

                if decision.action == "stop":
                    logger.append_jsonl(
                        "decisions.jsonl",
                        {
                            "timestamp": (
                                datetime.now()
                                .astimezone()
                                .isoformat()
                            ),
                            "decision": (
                                "stop_exploration"
                            ),
                            "details": {
                                "source": (
                                    "llm_planner"
                                ),
                                "reason": (
                                    decision.reason
                                ),
                                "remaining_budget_rmb": str(
                                    budget_manager
                                    .remaining_rmb()
                                ),
                            },
                        },
                    )

                    print(
                        "      Planner requested stop: "
                        f"{decision.reason}"
                    )

                    return (
                        "planner_requested_stop"
                    )

                plan = decision.plan

                if plan is None:
                    raise PlannerError(
                        "Planner selected run_experiment "
                        "but did not return a plan."
                    )

                # LLMExperimentPlanner already
                # validates the plan. Validate again
                # at the execution boundary.
                validator.validate_against_history(
                    plan=plan,
                    data_profile=profile,
                    existing_plans=all_plans,
                )

                print(
                    "      Planner selected: "
                    f"{plan.adapter_name}"
                )
                print(
                    "      Reason: "
                    f"{plan.reason}"
                )

                result = (
                    self._execute_and_record_plan(
                        plan=plan,
                        plan_source=(
                            "llm_planner"
                        ),
                        registry=registry,
                        executor=executor,
                        config=config,
                        profile=profile,
                        logger=logger,
                        checkpoint_before_execute=lambda experiment_id: (
                            self._write_run_checkpoint(
                                recovery=RunRecovery(),
                                logger=logger,
                                config=config,
                                all_plans=all_plans,
                                results=results,
                                in_progress_experiment_id=experiment_id,
                                run_started=run_started,
                                resume_count=resume_count,
                                status="running",
                            )
                        ),
                    )
                )

                all_plans.append(plan)
                results.append(result)
                self._write_run_checkpoint(
                    recovery=RunRecovery(),
                    logger=logger,
                    config=config,
                    all_plans=all_plans,
                    results=results,
                    in_progress_experiment_id=None,
                    run_started=run_started,
                    resume_count=resume_count,
                    status="running",
                )

                if result.status == "success":
                    metric_value = result_metric_value(result, config.metric)
                    print(
                        "      Experiment completed; "
                        f"validation {config.metric}: "
                        f"{_format_metric_value(metric_value)}"
                    )
                else:
                    error_key = self._experiment_error_key(
                        result,
                        all_plans,
                    )
                    experiment_error_counts[error_key] = (
                        experiment_error_counts.get(error_key, 0) + 1
                    )
                    print(
                        "      Experiment failed; "
                        "the failure is now part of "
                        "the planner history."
                    )
                    if (
                        experiment_error_counts[error_key]
                        >= config.maximum_same_experiment_errors
                    ):
                        print(
                            "      Autonomous loop stopped: repeated "
                            "experiment error limit reached "
                            f"({error_key[0]} / {error_key[1]})"
                        )
                        return "maximum_same_experiment_errors_reached"

            except BudgetExceededError as error:
                self._record_error(
                    logger=logger,
                    stage="planner_budget",
                    error=error,
                    extra={
                        "next_experiment_id": (
                            next_experiment_id
                        ),
                        "remaining_budget_rmb": str(
                            budget_manager
                            .remaining_rmb()
                        ),
                    },
                )

                print(
                    "      Autonomous loop stopped: "
                    "LLM budget exhausted"
                )

                return "llm_budget_exhausted"

            except (
                PlannerError,
                PlanValidationError,
                LLMClientError,
                BudgetError,
            ) as error:
                planner_error_count += 1

                self._record_error(
                    logger=logger,
                    stage="planner",
                    error=error,
                    extra={
                        "next_experiment_id": (
                            next_experiment_id
                        ),
                        "consecutive_planner_errors": (
                            planner_error_count
                        ),
                    },
                )

                print(
                    "      Planner error "
                    f"{planner_error_count}/"
                    f"{config.maximum_planner_errors}: "
                    f"{error}"
                )

                if (
                    planner_error_count
                    >= config.maximum_planner_errors
                ):
                    print(
                        "      Autonomous loop stopped: "
                        "planner error limit reached"
                    )
                    return (
                        "maximum_planner_errors_"
                        "reached"
                    )

    def _execute_and_record_plan(
        self,
        plan,
        plan_source,
        registry,
        executor,
        config,
        profile,
        logger,
        record_plan=True,
        allow_existing=False,
        checkpoint_before_execute=None,
    ):
        experiment_id = plan.experiment_id

        adapter = registry.adapter_for(
            plan.adapter_name
        )

        # Give the adapter a per-run cache directory. The worker runs in a
        # spawned child, so this cannot be an in-memory attribute: the value
        # travels with the pickled adapter and the worker reads/writes the
        # directory itself. Only adapters that look for the key are affected,
        # and only the embedding adapter does.
        plan.hyperparameters.setdefault(
            "embedding_cache_dir",
            str(Path(logger.run_dir) / "embedding_cache"),
        )

        experiment_directory = (
            logger.experiment_dir(
                experiment_id,
                allow_existing=allow_existing,
            )
        )

        if record_plan:
            logger.write_json(
                (
                    f"experiments/"
                    f"{experiment_id}/plan.json"
                ),
                {
                    **plan.to_dict(),
                    "plan_source": plan_source,
                },
            )

            logger.append_jsonl(
                "plans.jsonl",
                {
                    **plan.to_dict(),
                    "plan_source": plan_source,
                },
            )

            logger.append_jsonl(
                "decisions.jsonl",
                {
                    "timestamp": (
                        datetime.now()
                        .astimezone()
                        .isoformat()
                    ),
                    "decision": (
                        "run_experiment"
                    ),
                    "details": {
                        **plan.to_dict(),
                        "plan_source": (
                            plan_source
                        ),
                    },
                },
            )

        if checkpoint_before_execute is not None:
            checkpoint_before_execute(experiment_id)

        print(
            f"      {experiment_id}: "
            f"{adapter.model_name} "
            f"({plan_source})"
        )

        result = executor.execute(
            experiment_id=experiment_id,
            adapter=adapter,
            plan=plan,
            config=config,
            data_profile=profile,
            experiment_dir=(
                experiment_directory
            ),
        )

        logger.write_json(
            (
                f"experiments/"
                f"{experiment_id}/"
                f"experiment.json"
            ),
            result.to_dict(),
        )

        logger.append_jsonl(
            "experiments.jsonl",
            result.to_dict(),
        )

        if result.status == "success":
            metric_value = result_metric_value(result, config.metric)
            print(
                f"      validation {config.metric}: "
                f"{_format_metric_value(metric_value)}"
            )
        else:
            print(
                "      experiment failed: "
                f"{result.error_message}"
            )
            self._record_error(
                logger=logger,
                stage="experiment",
                error=RuntimeError(result.error_message or "Experiment failed"),
                extra={
                    "experiment_id": experiment_id,
                    "error_type": result.error_type,
                    "raw_error_type": result.raw_error_type,
                },
            )

        self._record_measured_throughput(
            plan=plan,
            profile=profile,
            result=result,
        )

        return result

    def _throughput_cache(self):
        if self._throughput_cache_instance is None:
            self._throughput_cache_instance = ThroughputCache(
                Path(self.runs_dir) / "throughput_cache.json"
            )
        return self._throughput_cache_instance

    def _record_measured_throughput(self, plan, profile, result):
        """Feed this experiment's real per-epoch cost back into the gate.

        Only image experiments have a per-epoch cost model; tabular and text
        adapters finish in one shot and are bounded by memory, not time.
        """
        if (profile or {}).get("modality") != "image":
            return
        entry = measurement_from_result(plan, profile, result)
        if entry is None:
            return
        cache = self._throughput_cache()
        cache.record(entry)
        context = profile.get("execution_context")
        if isinstance(context, dict):
            context["measured_throughput"] = cache.entries()
        print(
            "      measured throughput: "
            f"{entry['max_epoch_seconds']:.1f}s/epoch "
            f"({entry['epochs_completed']} epochs, "
            f"{entry['adapter_name']}@{entry['image_size']}px)"
        )

    @staticmethod
    def _next_experiment_id(plans):
        numbers = []
        for plan in plans:
            try:
                prefix, number = plan.experiment_id.split("_", 1)
                if prefix != "exp":
                    raise ValueError
                numbers.append(int(number))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid experiment ID: {plan.experiment_id!r}"
                ) from error
        return f"exp_{max(numbers, default=0) + 1:03d}"

    @staticmethod
    def _experiment_error_key(result, plans):
        """Count repeat failures per adapter and stable error category."""
        plan = next(
            (
                candidate
                for candidate in reversed(plans)
                if candidate.experiment_id == result.experiment_id
            ),
            None,
        )
        adapter_name = (
            plan.adapter_name if plan is not None else result.model_name
        )
        return (adapter_name, result.error_type or "unknown")

    def _write_run_checkpoint(
        self,
        recovery,
        logger,
        config,
        all_plans,
        results,
        in_progress_experiment_id,
        run_started,
        resume_count,
        status,
        stop_reason=None,
    ):
        checkpoint_plans = list(all_plans)
        next_number = int(
            self._next_experiment_id(checkpoint_plans).split("_", 1)[1]
        )
        if in_progress_experiment_id is not None:
            try:
                in_progress_number = int(
                    in_progress_experiment_id.split("_", 1)[1]
                )
            except (AttributeError, IndexError, ValueError) as error:
                raise ValueError(
                    "Invalid in-progress experiment ID: "
                    f"{in_progress_experiment_id!r}"
                ) from error
            next_number = max(next_number, in_progress_number + 1)

        recovery.write_checkpoint(
            logger.run_dir,
            RunState(
                task_name=config.name,
                status=status,
                completed_experiment_ids=[
                    result.experiment_id for result in results
                ],
                in_progress_experiment_id=in_progress_experiment_id,
                next_experiment_number=next_number,
                accumulated_runtime_seconds=(
                    time.perf_counter() - run_started
                ),
                resume_count=resume_count,
                stop_reason=stop_reason,
            ),
        )

    @staticmethod
    def _record_error(
        logger,
        stage,
        error,
        extra=None,
    ):
        record = {
            "timestamp": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
            "stage": stage,
            "error_type": (
                type(error).__name__
            ),
            "error_message": str(error),
        }

        if extra:
            record.update(extra)

        logger.append_jsonl(
            "errors.jsonl",
            record,
        )
