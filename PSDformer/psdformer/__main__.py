from __future__ import annotations
import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PSDformer reference: synthetic smoke demo, subject-disjoint training, evaluation and inference")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run synthetic train/evaluate/predict; NOT manuscript replication")
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--config", default=str(Path(__file__).parent/"configs"/"smoke.json"))
    synth = sub.add_parser("make-demo-data", help="Create explicitly synthetic data and regime labels")
    synth.add_argument("--output", default="data/synthetic.npz")
    synth.add_argument("--subjects", type=int, default=6)
    synth.add_argument("--windows-per-subject", type=int, default=5)
    synth.add_argument("--input-len", type=int, default=24)
    synth.add_argument("--pred-len", type=int, default=3)
    synth.add_argument("--input-dim", type=int, default=5)
    synth.add_argument("--output-dim", type=int, default=1)
    synth.add_argument("--num-classes", type=int, default=3)
    synth.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Fit train-only scalers; choose checkpoints on validation subjects")
    train.add_argument("--data", required=True)
    train.add_argument("--config", default=str(Path(__file__).parent/"configs"/"paper.json"))
    train.add_argument("--output", required=True)
    train.add_argument("--split-json")
    train.add_argument("--device", default="cpu")
    evaluate = sub.add_parser("evaluate", help="Evaluate original data using the checkpoint's saved split")
    predict = sub.add_parser("predict", help="Predict CSV/NPZ using frozen checkpoint preprocessing")
    for command in (evaluate, predict):
        command.add_argument("--data", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--threads", type=int, default=2)
    evaluate.add_argument("--partition", choices=("train", "val", "test"), default="test")
    predict.add_argument("--partition", choices=("all", "train", "val", "test"), default="all")
    prepare = sub.add_parser("prepare-csv", help="Chronological within-session input/future-target windows; explicit columns only")
    prepare.add_argument("--input", required=True, help="Input CSV")
    prepare.add_argument("--output", required=True, help="Output NPZ")
    prepare.add_argument("--feature-columns", nargs="+", required=True)
    prepare.add_argument("--target-columns", nargs="*", default=[])
    prepare.add_argument("--subject-column", required=True)
    prepare.add_argument("--session-column", required=True)
    prepare.add_argument("--time-column", required=True)
    prepare.add_argument("--label-column")
    prepare.add_argument("--num-classes", type=int)
    prepare.add_argument("--input-len", type=int, default=24)
    prepare.add_argument("--pred-len", type=int, default=3)
    prepare.add_argument("--stride", type=int, default=1)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "make-demo-data":
        from .data import make_demo_data
        print(make_demo_data(args.output, args.subjects, args.windows_per_subject, args.input_len,
                            args.pred_len, args.input_dim, args.output_dim, args.num_classes, args.seed))
    elif args.command == "prepare-csv":
        from .data import prepare_csv
        print(prepare_csv(args.input, args.output, args.feature_columns, args.target_columns,
            args.subject_column, args.session_column, args.time_column, args.input_len, args.pred_len,
            args.stride, args.label_column, args.num_classes))
    elif args.command == "train":
        from .train import train_model
        print(json.dumps(train_model(args.data, args.config, args.output, args.device, args.split_json), indent=2))
    elif args.command in ("evaluate", "predict"):
        import torch
        if args.threads < 1:
            raise ValueError("threads must be positive.")
        torch.set_num_threads(args.threads)
        if args.command == "evaluate":
            from .evaluate import evaluate_checkpoint
            print(json.dumps(evaluate_checkpoint(args.data, args.checkpoint, args.output, args.device, args.batch_size, args.partition), indent=2))
        else:
            from .predict import predict_file
            print(predict_file(args.data, args.checkpoint, args.output, args.device, args.batch_size, args.partition))
    elif args.command == "demo":
        from .data import make_demo_data
        from .evaluate import evaluate_checkpoint
        from .predict import predict_file
        from .train import train_model
        from .utils import read_config, write_json
        from .model import PSDformerConfig
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        config = read_config(args.config)
        model = PSDformerConfig(**config["model"])
        demo = config.get("demo", {})
        data_path = make_demo_data(output/"synthetic.npz", subjects=demo.get("subjects", 6),
            windows_per_subject=demo.get("windows_per_subject", 5), input_len=model.input_len,
            pred_len=model.pred_len, input_dim=model.input_dim, output_dim=model.output_dim,
            num_classes=model.num_classes, seed=config["training"].get("seed", 42))
        result = train_model(data_path, args.config, output, args.device, synthetic_demo=True)
        evaluate_checkpoint(data_path, result["checkpoint"], output/"metrics.json", args.device,
                            batch_size=config["training"].get("batch_size", 32), partition="test")
        predict_file(data_path, result["checkpoint"], output/"predictions.csv", args.device,
                     batch_size=config["training"].get("batch_size", 32), partition="test")
        write_json(output/"demo_summary.json", {"synthetic_demo": True, "training": result,
            "metrics": str(output/"metrics.json"), "predictions": str(output/"predictions.csv"),
            "warning": "Engineering smoke test only. These are NOT manuscript experiments or real stress labels."})
        print(f"Synthetic smoke test completed (NOT paper replication): {output.resolve()}")


if __name__ == "__main__":
    main()
