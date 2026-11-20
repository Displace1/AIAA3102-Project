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
from superagent.plan_validator import (
    PlanValidationError,
    PlanValidator,
)
from superagent.recovery import RunRecovery, RunState
from superagent.stop_policy import StopPolicy
from superagent.task_config import load_task_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

        initial_plans = default_plans[
            :initial_count
        ]

        initial_plans = validator.validate_many(
            initial_plans,
            data_profile=profile,
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
            "by validation accuracy"
        )

        best = select_best_model(results)

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

        selection = {
            "run_type": config.run_type,
            "selected_experiment": (
                best.experiment_id
            ),
            "selected_model": best.model_name,
            "selection_metric": (
                config.metric
            ),
            "validation_accuracy": (
                best.validation_accuracy
            ),
            "artifact_path": (
                best.artifact_path
            ),
            "selected_automatically": True,
            "test_used_for_selection": False,
            "selection_rule": (
                "Highest validation accuracy; "
                "shorter training time breaks "
                "exact ties."
            ),
            "stop_reason": stop_reason,
            "total_experiments": len(
                results
            ),
            "llm_cost_rmb": llm_cost_rmb,
        }

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
            "validation_accuracy": (
                best.validation_accuracy
            ),
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
                error_key = previous_result.error_type or "unknown"
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
                    print(
                        "      Experiment completed; "
                        "validation accuracy: "
                        f"{result.validation_accuracy:.6f}"
                    )
                else:
                    error_key = result.error_type or "unknown"
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
                            f"experiment error limit reached ({error_key})"
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
            print(
                "      validation accuracy: "
                f"{result.validation_accuracy:.6f}"
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

        return result

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
