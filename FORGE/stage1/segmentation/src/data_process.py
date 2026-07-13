import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import stage1_pipeline as pipeline


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument(
        "--output_path",
        type=str,
        default=(
            PROJECT_ROOT / "stage1_preliminary" / "data_process_summary.json"
        ).as_posix(),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pipeline_args = pipeline.parse_args(
        [
            "--mode",
            "train",
            "--data_root",
            args.data_root,
            "--train_csv",
            args.train_csv,
        ]
    )
    train_samples = pipeline.load_train_samples(pipeline_args)
    forged_count = int(sum(sample["label"] == 1 for sample in train_samples))
    real_count = int(sum(sample["label"] == 0 for sample in train_samples))

    test_dir = (
        args.test_dir
        or (Path(args.data_root) / "ForgeryAnalysis_Stage_1_Test" / "Image").as_posix()
    )
    test_summary = {
        "test_dir": test_dir,
        "num_images": None,
    }
    if Path(test_dir).exists():
        test_args = pipeline.parse_args(
            [
                "--mode",
                "predict",
                "--test_dir",
                test_dir,
            ]
        )
        test_samples = pipeline.build_test_samples(test_args)
        test_summary["num_images"] = len(test_samples)

    summary = {
        "data_root": args.data_root,
        "train_csv": args.train_csv,
        "train_total": len(train_samples),
        "train_forged": forged_count,
        "train_real": real_count,
        "test_summary": test_summary,
    }
    output_path = Path(args.output_path)
    pipeline.save_json(summary, output_path)
    return output_path


if __name__ == "__main__":
    main()
