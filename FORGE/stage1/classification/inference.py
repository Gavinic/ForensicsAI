import argparse
import json
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from adversary_detector import AdversaryDetector
from forensics_sam import ForensicsSAM
from mini_dataloader import BasicDataloader, build_stage1_test_records
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
from tqdm import tqdm

MODEL_TYPES = ["vit_b", "vit_l", "vit_h"]
DEFAULT_SAM_CHECKPOINTS = {
    "vit_b": "./weight/sam_vit_b_01ec64.pth",
    "vit_l": "./weight/sam_vit_l_0b3195.pth",
    "vit_h": "./weight/sam_vit_h_4b8939.pth",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference on ForgeryAnalysis_Stage_1_Test using a pretrained ForensicsSAM model."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to dataset root containing train.csv/submission.csv and Stage_1 folders.",
    )
    parser.add_argument(
        "--forensics-weights",
        type=str,
        required=True,
        help="Path to pretrained ForensicsSAM parameters (from save_all_parameters).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./submission_pred.csv",
        help="Output CSV path for predicted labels.",
    )

    parser.add_argument("--sam-type", type=str, default="vit_h", choices=MODEL_TYPES)
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="",
        help="SAM base checkpoint path; defaults to project preset for --sam-type.",
    )
    parser.add_argument(
        "--rank", type=int, default=8, help="LoRA rank r used by ForensicsSAM."
    )

    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--normalize-type", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-file",
        type=str,
        default="",
        help="Optional JSON file containing a tuned threshold (key: threshold). Overrides --threshold.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument(
        "--enable-adversary-experts",
        action="store_true",
        help="Enable adaptive adversarial experts at inference time.",
    )
    parser.add_argument(
        "--adversary-experts-path",
        type=str,
        default="",
        help="Path to adversary expert parameters when --enable-adversary-experts is set.",
    )
    parser.add_argument(
        "--use-adv-detector",
        action="store_true",
        help="Use adversary detector to decide adversarial expert activation.",
    )
    parser.add_argument(
        "--adv-detector-path",
        type=str,
        default="",
        help="Path to adversary detector checkpoint.",
    )
    return parser.parse_args()


def resolve_sam_checkpoint(sam_type: str, sam_checkpoint: str) -> str:
    if sam_checkpoint:
        return sam_checkpoint
    return DEFAULT_SAM_CHECKPOINTS[sam_type]


def resolve_cls_threshold(args: argparse.Namespace) -> float:
    if not args.threshold_file:
        return float(args.threshold)

    threshold_file = os.path.abspath(args.threshold_file)
    if not os.path.isfile(threshold_file):
        raise FileNotFoundError(f"Threshold metadata file not found: {threshold_file}")

    with open(threshold_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "threshold" not in payload:
        raise ValueError(
            f"{threshold_file} must be a JSON object containing key 'threshold'."
        )

    threshold = float(payload["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Threshold in {threshold_file} must be in [0.0, 1.0], got {threshold}"
        )

    print(f"Loaded threshold from metadata: {threshold:.4f} ({threshold_file})")
    return threshold


def build_forensics_model(
    args: argparse.Namespace, device: torch.device
) -> ForensicsSAM:
    sam_checkpoint = resolve_sam_checkpoint(args.sam_type, args.sam_checkpoint)
    sam, _ = sam_model_registry[args.sam_type](
        image_size=args.image_size, checkpoint=sam_checkpoint
    )

    if args.enable_adversary_experts and not args.adversary_experts_path:
        raise ValueError(
            "--adversary-experts-path is required when --enable-adversary-experts is set."
        )
    adversary_experts_path = (
        args.adversary_experts_path if args.enable_adversary_experts else None
    )
    model = ForensicsSAM(
        sam,
        r=args.rank,
        forgery_experts_path=args.forensics_weights,
        adversary_experts_path=adversary_experts_path,
        load_pretrained=True,
        freeze_shared_experts=True,
        freeze_detector=True,
        enable_adversary_experts=args.enable_adversary_experts,
    )
    model.to(device).eval()
    return model


def build_adv_detector(
    args: argparse.Namespace, device: torch.device
) -> AdversaryDetector:
    if not args.use_adv_detector:
        return None
    if not args.adv_detector_path:
        raise ValueError(
            "--adv-detector-path is required when --use-adv-detector is set."
        )
    detector = AdversaryDetector().to(device).eval()
    detector.load_detector(args.adv_detector_path)
    return detector


def build_submission_dataframe(
    template_path: str, image_names: List[str], labels: List[int]
) -> pd.DataFrame:
    if os.path.isfile(template_path):
        template_df = pd.read_csv(template_path)
        if "Image_name" in template_df.columns:
            mapping = {name: int(label) for name, label in zip(image_names, labels)}
            template_df["label"] = (
                template_df["Image_name"].map(mapping).fillna(0).astype(int)
            )
            return template_df
    return pd.DataFrame({"Image_name": image_names, "label": labels})


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    cls_threshold = resolve_cls_threshold(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    model = build_forensics_model(args, device)
    adv_detector = build_adv_detector(args, device)

    test_records = build_stage1_test_records(args.data_root, strict=True)
    test_dataset = BasicDataloader(
        sample_records=test_records,
        input_size=args.image_size,
        normalize_type=args.normalize_type,
        augment_prob=0.0,
        enable_aug_types=[],
        intensity={"rates": [0.8], "qfs": [75], "sds": [9], "ksizes": [9]},
        mode="val",
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_names: List[str] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
    cursor = 0

    with torch.no_grad():
        progress = tqdm(test_loader, desc="Inference")
        for images, _, _, _ in progress:
            images = images.to(device, non_blocking=True)

            if adv_detector is not None:
                adv_logits, _ = adv_detector(images)
                activate_adv = torch.argmax(adv_logits, dim=1)
            else:
                activate_adv = torch.zeros(
                    images.size(0), device=device, dtype=torch.long
                )

            _, cls_prediction = model(images, activate_adv)
            probs = torch.sigmoid(cls_prediction).view(-1).detach().cpu().numpy()
            labels = (probs > cls_threshold).astype(np.int32)

            batch_records = test_records[cursor : cursor + len(probs)]
            cursor += len(probs)

            all_names.extend([record["image_name"] for record in batch_records])
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

            progress.set_postfix(
                {"positive_rate": float(np.mean(labels)) if len(labels) > 0 else 0.0}
            )

    output_df = build_submission_dataframe(
        template_path=os.path.join(args.data_root, "submission.csv"),
        image_names=all_names,
        labels=all_labels,
    )

    output_csv = os.path.abspath(args.output_csv)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    prob_csv = os.path.splitext(output_csv)[0] + "_prob.csv"
    prob_df = pd.DataFrame(
        {"Image_name": all_names, "prob": all_probs, "label": all_labels}
    )
    prob_df.to_csv(prob_csv, index=False)

    print(f"Saved label predictions to: {output_csv}")
    print(f"Saved probability predictions to: {prob_csv}")
    print(f"Classification threshold used: {cls_threshold:.4f}")


if __name__ == "__main__":
    main()
