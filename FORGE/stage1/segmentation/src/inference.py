import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import stage1_pipeline as pipeline


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(PROJECT_ROOT / "outputs" / "stage1").as_posix(),
    )
    parser.add_argument("--run_name", type=str, default="vit_stage1_pred")
    parser.add_argument(
        "--weights_path",
        type=str,
        default=(
            PROJECT_ROOT / "outputs" / "stage1" / "vit_stage1_ft" / "best.pth"
        ).as_posix(),
    )
    parser.add_argument(
        "--answer_csv",
        type=str,
        default=(PROJECT_ROOT / "submission_ensemble.csv").as_posix(),
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--backbone", type=str, default="vit", choices=["vit", "hrnet"])
    parser.add_argument("--sam_checkpoint", type=str, default="")
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_l",
        choices=["default", "vit_b", "vit_l", "vit_h"],
    )
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--gt_ratio", type=int, default=16)
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--test_bs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=666666)
    parser.add_argument("--dict_size", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save_png", type=int, default=1)
    parser.add_argument("--positive_ratio_threshold", type=float, default=0.001)
    parser.add_argument("--min_component_area", type=int, default=32)
    parser.add_argument("--min_component_area_ratio", type=float, default=0.0001)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--rle_order", type=str, default="F", choices=["F", "C"])
    parser.add_argument("--train_only_forged", type=int, default=1)
    parser.add_argument("--score_with_cls", type=int, default=1)
    return parser


def build_pipeline_argv(args):
    argv = [
        "--mode",
        "predict",
        "--data_root",
        args.data_root,
        "--train_csv",
        args.train_csv,
        "--test_dir",
        args.input_path,
        "--output_root",
        args.output_root,
        "--run_name",
        args.run_name,
        "--weights_path",
        args.weights_path,
        "--backbone",
        args.backbone,
        "--sam_checkpoint",
        args.sam_checkpoint,
        "--sam_model_type",
        args.sam_model_type,
        "--input_size",
        str(args.input_size),
        "--gt_ratio",
        str(args.gt_ratio),
        "--train_bs",
        str(args.train_bs),
        "--test_bs",
        str(args.test_bs),
        "--num_workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--metric",
        args.metric,
        "--val_ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--dict_size",
        str(args.dict_size),
        "--gpu",
        args.gpu,
        "--save_png",
        str(args.save_png),
        "--positive_ratio_threshold",
        str(args.positive_ratio_threshold),
        "--min_component_area",
        str(args.min_component_area),
        "--min_component_area_ratio",
        str(args.min_component_area_ratio),
        "--morph_kernel",
        str(args.morph_kernel),
        "--rle_order",
        args.rle_order,
        "--train_only_forged",
        str(args.train_only_forged),
        "--score_with_cls",
        str(args.score_with_cls),
    ]
    answer_csv_path = Path(args.answer_csv)
    if args.answer_csv and answer_csv_path.exists():
        argv.extend(["--answer_csv", answer_csv_path.as_posix()])
    return argv


def main(argv=None):
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    pipeline_argv = build_pipeline_argv(args) + extras
    run_dir = Path(pipeline.main(pipeline_argv))
    answer_path = run_dir / "answer.csv"
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(answer_path, output_path)
    return run_dir


if __name__ == "__main__":
    main()
