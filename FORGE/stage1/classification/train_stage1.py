import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from forensics_sam import ForensicsSAM
from mini_dataloader import (
    BasicDataloader,
    build_stage1_train_records,
    split_train_val_records,
    split_train_val_records_kfold,
)
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader

MODEL_TYPES = ["vit_b", "vit_l", "vit_h"]
DEFAULT_SAM_CHECKPOINTS = {
    "vit_b": "./weight/sam_vit_b_01ec64.pth",
    "vit_l": "./weight/sam_vit_l_0b3195.pth",
    "vit_h": "./weight/sam_vit_h_4b8939.pth",
}


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_list(value: str) -> List[int]:
    value = value.strip()
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    value = value.strip()
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ForensicsSAM on ForgeryAnalysis Stage-1 dataset format."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root path containing train.csv and Stage_1 folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./weight",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="train.csv",
        help="Training CSV filename under --data-root.",
    )
    parser.add_argument(
        "--best-model-name", type=str, default="forensics_stage1_best.pth"
    )
    parser.add_argument(
        "--last-model-name", type=str, default="forensics_stage1_last.pth"
    )

    parser.add_argument("--sam-type", type=str, default="vit_h", choices=MODEL_TYPES)
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="",
        help="SAM base checkpoint path; defaults to project preset for --sam-type.",
    )
    parser.add_argument("--rank", type=int, default=8)

    parser.add_argument(
        "--init-forensics-weights",
        type=str,
        default="",
        help="Optional initial ForensicsSAM parameter file from save_all_parameters.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint to resume training from.",
    )

    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--normalize-type", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--num-folds", type=int, default=1, help="If >1, use stratified k-fold split."
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Validation fold index used when --num-folds > 1.",
    )
    parser.add_argument(
        "--search-threshold-on-val",
        action="store_true",
        help="Search the best classification threshold on the validation set each epoch.",
    )
    parser.add_argument("--threshold-search-min", type=float, default=0.05)
    parser.add_argument("--threshold-search-max", type=float, default=0.95)
    parser.add_argument("--threshold-search-step", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--augment-prob", type=float, default=0.0)
    parser.add_argument(
        "--enable-aug-types",
        type=str,
        default="",
        help="Comma-separated augmentation ids, e.g. '3,4,5,6'.",
    )
    parser.add_argument("--rates", type=str, default="0.8")
    parser.add_argument("--qfs", type=str, default="75")
    parser.add_argument("--sds", type=str, default="9")
    parser.add_argument("--ksizes", type=str, default="9")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def resolve_sam_checkpoint(sam_type: str, sam_checkpoint: str) -> str:
    if sam_checkpoint:
        return sam_checkpoint
    return DEFAULT_SAM_CHECKPOINTS[sam_type]


def dice_loss_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def binary_f1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    tp = torch.sum(pred_flat * target_flat).item()
    fp = torch.sum(pred_flat * (1.0 - target_flat)).item()
    fn = torch.sum((1.0 - pred_flat) * target_flat).item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return float((2.0 * precision * recall) / (precision + recall + eps))


def binary_cls_metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same length")
    if probs.size == 0:
        return 0.0, 0.0

    pred = (probs > threshold).astype(np.float32)
    target = labels.astype(np.float32)

    acc = float(np.mean(pred == target))

    tp = float(np.sum(pred * target))
    fp = float(np.sum(pred * (1.0 - target)))
    fn = float(np.sum((1.0 - pred) * target))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = float((2.0 * precision * recall) / (precision + recall + eps))
    return acc, f1


def search_best_cls_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> Dict[str, float]:
    if probs.size == 0:
        raise ValueError("Cannot search threshold: no validation samples.")
    if not 0.0 <= threshold_min <= 1.0:
        raise ValueError("threshold_search_min must be in [0.0, 1.0]")
    if not 0.0 <= threshold_max <= 1.0:
        raise ValueError("threshold_search_max must be in [0.0, 1.0]")
    if threshold_min > threshold_max:
        raise ValueError("threshold_search_min must be <= threshold_search_max")
    if threshold_step <= 0:
        raise ValueError("threshold_search_step must be > 0")

    thresholds = np.arange(
        threshold_min,
        threshold_max + threshold_step * 0.5,
        threshold_step,
        dtype=np.float32,
    )
    if thresholds.size == 0:
        raise ValueError(
            "No thresholds generated for search. Check threshold search range/step."
        )

    best_threshold = float(thresholds[0])
    best_acc, best_f1 = binary_cls_metrics_from_probs(probs, labels, best_threshold)

    for threshold in thresholds[1:]:
        threshold = float(threshold)
        acc, f1 = binary_cls_metrics_from_probs(probs, labels, threshold)
        acc_improved = acc > best_acc + 1e-12
        same_acc_better_tie = abs(acc - best_acc) <= 1e-12 and abs(
            threshold - 0.5
        ) < abs(best_threshold - 0.5)
        if acc_improved or same_acc_better_tie:
            best_threshold = threshold
            best_acc = acc
            best_f1 = f1

    return {
        "threshold": best_threshold,
        "acc": best_acc,
        "f1": best_f1,
    }


def save_threshold_metadata(
    path: str,
    threshold: float,
    epoch: int,
    cls_acc: float,
    cls_f1: float,
    source: str,
) -> None:
    payload = {
        "threshold": float(threshold),
        "epoch": int(epoch),
        "cls_acc": float(cls_acc),
        "cls_f1": float(cls_f1),
        "source": source,
    }
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_model(args: argparse.Namespace, device: torch.device) -> ForensicsSAM:
    sam_checkpoint = resolve_sam_checkpoint(args.sam_type, args.sam_checkpoint)
    sam, _ = sam_model_registry[args.sam_type](
        image_size=args.image_size, checkpoint=sam_checkpoint
    )

    init_forensics = (
        args.init_forensics_weights if args.init_forensics_weights else None
    )
    model = ForensicsSAM(
        sam,
        r=args.rank,
        forgery_experts_path=init_forensics,
        adversary_experts_path=None,
        load_pretrained=bool(init_forensics),
        freeze_shared_experts=False,
        freeze_detector=False,
        enable_adversary_experts=False,
    )

    if args.resume:
        model.load_all_parameters(args.resume)

    model.to(device)
    return model


def evaluate(
    model: ForensicsSAM,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, object]:
    bce_criterion = torch.nn.BCEWithLogitsLoss()

    loss_meter = AverageMeter()
    cls_acc_meter = AverageMeter()
    mask_f1_meter = AverageMeter()
    cls_probs: List[float] = []
    cls_targets: List[float] = []

    model.eval()
    with torch.no_grad():
        for images, gt_masks, forged_label, _ in loader:
            images = images.to(device, non_blocking=True)
            gt_masks = gt_masks.to(device, non_blocking=True)
            forged_label = (
                forged_label.float().to(device, non_blocking=True).unsqueeze(1)
            )

            activate_adv = torch.zeros(images.size(0), device=device, dtype=torch.long)
            mask_logits, cls_logits = model(images, activate_adv)

            seg_bce = bce_criterion(mask_logits, gt_masks)
            seg_dice = dice_loss_from_logits(mask_logits, gt_masks)
            cls_bce = bce_criterion(cls_logits, forged_label)
            loss = seg_bce + seg_dice + cls_bce

            cls_prob = torch.sigmoid(cls_logits)
            cls_pred = (cls_prob > threshold).float()
            cls_acc = (cls_pred == forged_label).float().mean().item()
            cls_probs.extend(cls_prob.view(-1).detach().cpu().tolist())
            cls_targets.extend(forged_label.view(-1).detach().cpu().tolist())

            mask_prob = torch.sigmoid(mask_logits)
            mask_pred = (mask_prob > threshold).float()
            mask_f1 = binary_f1(mask_pred, gt_masks)

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            cls_acc_meter.update(cls_acc, batch_size)
            mask_f1_meter.update(mask_f1, batch_size)

    cls_acc_fixed, cls_f1_fixed = binary_cls_metrics_from_probs(
        np.asarray(cls_probs, dtype=np.float32),
        np.asarray(cls_targets, dtype=np.float32),
        threshold,
    )

    return {
        "loss": loss_meter.avg,
        "cls_acc": cls_acc_fixed,
        "cls_f1": cls_f1_fixed,
        "mask_f1": mask_f1_meter.avg,
        "score": cls_acc_fixed + mask_f1_meter.avg,
        "cls_probs": cls_probs,
        "cls_targets": cls_targets,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    records = build_stage1_train_records(
        data_root=args.data_root,
        train_csv_name=args.train_csv,
        strict=True,
    )
    if args.num_folds > 1:
        train_records, val_records = split_train_val_records_kfold(
            records,
            num_folds=args.num_folds,
            fold_index=args.fold_index,
            seed=args.seed,
        )
        split_mode = f"k-fold ({args.fold_index + 1}/{args.num_folds})"
    else:
        train_records, val_records = split_train_val_records(
            records,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        split_mode = f"holdout (val_ratio={args.val_ratio})"

    if len(train_records) == 0:
        raise ValueError("No training records available after split.")

    print(
        f"Split mode: {split_mode} | "
        f"train={len(train_records)} | val={len(val_records)}"
    )

    enable_aug_types = parse_int_list(args.enable_aug_types)
    intensity = {
        "rates": parse_float_list(args.rates),
        "qfs": [int(v) for v in parse_float_list(args.qfs)],
        "sds": parse_float_list(args.sds),
        "ksizes": [int(v) for v in parse_float_list(args.ksizes)],
    }

    train_dataset = BasicDataloader(
        sample_records=train_records,
        input_size=args.image_size,
        normalize_type=args.normalize_type,
        augment_prob=args.augment_prob,
        enable_aug_types=enable_aug_types,
        intensity=intensity,
        mode="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if len(val_records) > 0:
        val_dataset = BasicDataloader(
            sample_records=val_records,
            input_size=args.image_size,
            normalize_type=args.normalize_type,
            augment_prob=0.0,
            enable_aug_types=[],
            intensity=intensity,
            mode="val",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    model = build_model(args, device)

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    bce_criterion = torch.nn.BCEWithLogitsLoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_name = args.best_model_name
    last_model_name = args.last_model_name
    if args.num_folds > 1:
        if best_model_name == "forensics_stage1_best.pth":
            best_model_name = f"forensics_stage1_best_fold{args.fold_index}.pth"
        if last_model_name == "forensics_stage1_last.pth":
            last_model_name = f"forensics_stage1_last_fold{args.fold_index}.pth"

    best_path = os.path.join(args.output_dir, best_model_name)
    last_path = os.path.join(args.output_dir, last_model_name)
    best_threshold_path = f"{os.path.splitext(best_path)[0]}_threshold.json"
    last_threshold_path = f"{os.path.splitext(last_path)[0]}_threshold.json"

    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_meter = AverageMeter()
        train_cls_acc_meter = AverageMeter()

        for images, gt_masks, forged_label, _ in train_loader:
            images = images.to(device, non_blocking=True)
            gt_masks = gt_masks.to(device, non_blocking=True)
            forged_label = (
                forged_label.float().to(device, non_blocking=True).unsqueeze(1)
            )

            activate_adv = torch.zeros(images.size(0), device=device, dtype=torch.long)
            mask_logits, cls_logits = model(images, activate_adv)

            seg_bce = bce_criterion(mask_logits, gt_masks)
            seg_dice = dice_loss_from_logits(mask_logits, gt_masks)
            cls_bce = bce_criterion(cls_logits, forged_label)
            loss = seg_bce + seg_dice + cls_bce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cls_prob = torch.sigmoid(cls_logits)
            cls_pred = (cls_prob > args.threshold).float()
            cls_acc = (cls_pred == forged_label).float().mean().item()

            batch_size = images.size(0)
            train_loss_meter.update(loss.item(), batch_size)
            train_cls_acc_meter.update(cls_acc, batch_size)

        model.save_all_parameters(last_path)

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, args.threshold)
            eval_name = "val"
        else:
            metrics = {
                "loss": train_loss_meter.avg,
                "cls_acc": train_cls_acc_meter.avg,
                "cls_f1": 0.0,
                "mask_f1": 0.0,
                "score": train_cls_acc_meter.avg,
                "cls_probs": [],
                "cls_targets": [],
            }
            eval_name = "train"

        selected_threshold = args.threshold
        selected_cls_acc = float(metrics["cls_acc"])
        selected_cls_f1 = float(metrics["cls_f1"])
        threshold_source = "fixed"

        if val_loader is not None and args.search_threshold_on_val:
            threshold_result = search_best_cls_threshold(
                np.asarray(metrics["cls_probs"], dtype=np.float32),
                np.asarray(metrics["cls_targets"], dtype=np.float32),
                threshold_min=args.threshold_search_min,
                threshold_max=args.threshold_search_max,
                threshold_step=args.threshold_search_step,
            )
            selected_threshold = float(threshold_result["threshold"])
            selected_cls_acc = float(threshold_result["acc"])
            selected_cls_f1 = float(threshold_result["f1"])
            threshold_source = "searched_on_val"

        if val_loader is not None:
            score_for_selection = selected_cls_acc + float(metrics["mask_f1"])
        else:
            score_for_selection = float(metrics["score"])

        save_threshold_metadata(
            last_threshold_path,
            threshold=selected_threshold,
            epoch=epoch,
            cls_acc=selected_cls_acc,
            cls_f1=selected_cls_f1,
            source=threshold_source,
        )

        if score_for_selection > best_score:
            best_score = score_for_selection
            model.save_all_parameters(best_path)
            save_threshold_metadata(
                best_threshold_path,
                threshold=selected_threshold,
                epoch=epoch,
                cls_acc=selected_cls_acc,
                cls_f1=selected_cls_f1,
                source=threshold_source,
            )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss_meter.avg:.4f} "
            f"train_cls_acc={train_cls_acc_meter.avg:.4f} "
            f"{eval_name}_loss={metrics['loss']:.4f} "
            f"{eval_name}_cls_acc={metrics['cls_acc']:.4f} "
            f"{eval_name}_cls_f1={metrics['cls_f1']:.4f} "
            f"{eval_name}_mask_f1={metrics['mask_f1']:.4f} "
            f"selected_threshold={selected_threshold:.3f} "
            f"selected_cls_acc={selected_cls_acc:.4f} "
            f"best_score={best_score:.4f}"
        )

    print(f"Saved best model to: {best_path}")
    print(f"Saved last model to: {last_path}")
    print(f"Saved best threshold metadata to: {best_threshold_path}")
    print(f"Saved last threshold metadata to: {last_threshold_path}")


if __name__ == "__main__":
    main()
