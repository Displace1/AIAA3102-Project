import argparse
from pathlib import Path

from superagent.controller import SuperAgentController
from superagent.final_evaluator import FinalEvaluator
from superagent.final_model_runner import FinalModelRunner
from superagent.task_config import load_task_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run or resume the budgeted autonomous SuperAgent pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Path to a task JSON configuration file.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Directory used to store run artifacts.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        default=None,
        help="Resume an interrupted run from an existing run directory.",
    )
    parser.add_argument(
        "--final-evaluate-run",
        type=Path,
        default=None,
        help="Evaluate the locked final model once on the designated test set.",
    )
    parser.add_argument(
        "--predict-run",
        type=Path,
        default=None,
        help="Run an exported final model on unlabeled input data.",
    )
    parser.add_argument(
        "--input-data",
        type=Path,
        default=None,
        help="NPY, CSV, image file, or image directory used with --predict-run.",
    )
    parser.add_argument(
        "--prediction-output-dir",
        type=Path,
        default=None,
        help="Empty output directory used with --predict-run.",
    )
    args = parser.parse_args()
    selected_modes = sum(
        value is not None
        for value in (args.resume_run, args.final_evaluate_run, args.predict_run)
    )
    if selected_modes > 1:
        parser.error(
            "--resume-run, --final-evaluate-run and --predict-run are "
            "mutually exclusive."
        )
    if args.predict_run is not None:
        if args.input_data is None or args.prediction_output_dir is None:
            parser.error(
                "--predict-run requires --input-data and "
                "--prediction-output-dir."
            )
    elif args.config is None:
        parser.error(
            "--config is required for training, recovery and final evaluation."
        )
    elif args.input_data is not None or args.prediction_output_dir is not None:
        parser.error(
            "--input-data and --prediction-output-dir require --predict-run."
        )
    return args


def main():
    args = parse_args()
    if args.predict_run is not None:
        result = FinalModelRunner().predict(
            run_directory=args.predict_run,
            input_path=args.input_data,
            output_directory=args.prediction_output_dir,
        )
        print("\nFinal model inference completed successfully.")
        print(f"Selected experiment: {result.selected_experiment}")
        print(f"Adapter: {result.adapter_name}")
        print(f"Predictions: {result.prediction_count}")
        print(f"CSV output: {result.predictions_csv_path}")
        print(f"NPY output: {result.predictions_npy_path}")
        print(f"Manifest: {result.manifest_path}")
        return

    if args.final_evaluate_run is not None:
        config = load_task_config(args.config)
        result = FinalEvaluator().evaluate(
            run_directory=args.final_evaluate_run,
            config=config,
        )
        print("\nFinal test evaluation completed successfully.")
        print(f"Selected experiment: {result.selected_experiment}")
        print(f"Validation accuracy: {result.validation_accuracy:.6f}")
        print(f"Test accuracy: {result.test_accuracy:.6f}")
        print(f"Test samples: {result.test_samples}")
        print(f"Final model: {result.final_model_path}")
        print(f"Predictions: {result.predictions_path}")
        return

    controller = SuperAgentController(
        config_path=args.config,
        runs_dir=args.runs_dir,
        resume_run_directory=args.resume_run,
    )
    summary = controller.run()

    print("\nRun completed successfully.")
    print(f"Detected modality: {summary['modality']}")
    print(f"Selected model: {summary['selected_model']}")
    print(
        "Validation accuracy: "
        f"{summary['validation_accuracy']:.6f}"
    )
    print(f"Run directory: {summary['run_directory']}")


if __name__ == "__main__":
    main()
