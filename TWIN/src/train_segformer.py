"""
Final model training script - uses all data
- Training set: Forgery 800 images + Data 3383 images (mask>1%)
- No validation set

Usage:
    python train_final.py --model b3 --batch_size 8 --epochs 50
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# ==================== Early Stopping ====================


class EarlyStopping:
    """Early Stopping monitor"""

    def __init__(self, patience=5, mode="max", delta=0.001):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


# ==================== Data Augmentation ====================


def get_train_transforms(
    image_size: int = 512, aug_level: str = "geometric"
) -> A.Compose:
    """Get training data augmentation."""
    if aug_level == "none":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    elif aug_level == "geometric":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    elif aug_level == "light":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:  # full
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    translate_percent=(-0.1, 0.1),
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
                ),
                A.GaussNoise(std_range=(0.05, 0.2), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Get validation data augmentation."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


# ==================== Dataset ====================


class ForgeryDataset(Dataset):
    """Forgery dataset - uses only forged (Black) images, supports train/val split"""

    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        aug_level: str = "geometric",
        subset: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.subset = subset

        # Load only forged images (Black directory)
        self.black_image_dir = self.data_dir / "Black" / "Image"
        self.black_mask_dir = self.data_dir / "Black" / "Mask"
        all_paths = sorted(
            list(self.black_image_dir.glob("*.jpg"))
            + list(self.black_image_dir.glob("*.png"))
        )

        # Split train/val (when val_split=0, all data is used as the training set)
        if val_split > 0:
            random.seed(seed)
            indices = list(range(len(all_paths)))
            random.shuffle(indices)
            val_size = int(len(all_paths) * val_split)
            if subset == "train":
                split_indices = indices[val_size:]
                print(
                    f"[Forgery Dataset] Training set - using forged images only (split={1-val_split:.0%})"
                )
            else:
                split_indices = indices[:val_size]
                print(
                    f"[Forgery Dataset] Validation set - using forged images only (split={val_split:.0%})"
                )
            self.image_paths = [all_paths[i] for i in split_indices]
        else:
            # Use all data
            self.image_paths = all_paths
            print(
                f"[Forgery Dataset] Using all forged images (no validation set): {len(all_paths)} images"
            )

        print(f"  - Forged images (Black): {len(self.image_paths)} images")

        # Data augmentation
        if subset == "train":
            self.transform = get_train_transforms(image_size, aug_level)
        else:
            self.transform = get_val_transforms(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = img_path.stem

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Unable to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask_path = self.black_mask_dir / f"{img_name}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = (
            transformed["mask"].unsqueeze(0)
            if len(transformed["mask"].shape) == 2
            else transformed["mask"]
        )

        return {
            "image": image,
            "mask": mask,
            "image_name": img_name,
        }


class FilteredDataDataset(Dataset):
    """New dataset - uses only images with mask ratio > threshold, supports sampling ratio"""

    def __init__(
        self,
        data_dir: str,
        data_analysis: str,
        image_size: int = 512,
        aug_level: str = "geometric",
        min_mask_ratio: float = 0.01,
        sample_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        # Load analysis file
        with open(data_analysis) as f:
            all_data = json.load(f)

        # Filter images with mask ratio > threshold
        filtered_data = [d for d in all_data if d["ratio"] > min_mask_ratio]
        filtered_data = sorted(filtered_data, key=lambda x: x["ratio"], reverse=True)

        # Sample by sample_ratio
        random.seed(seed)
        sample_size = int(len(filtered_data) * sample_ratio)
        sampled_data = filtered_data[
            :sample_size
        ]  # Take the top N (already sorted by mask ratio)

        self.samples = sampled_data

        # Image and mask directories
        self.img_dir = self.data_dir / "img"
        self.mask_dir = self.data_dir / "mask"

        print(
            f"[Data Dataset] Filtered training set (mask>{min_mask_ratio*100:.0f}%, sample={sample_ratio:.0%})"
        )
        print(f"  - Original filtered: {len(filtered_data)} images")
        print(f"  - This sample: {len(self.samples)} images")

        # Data augmentation
        self.transform = get_train_transforms(image_size, aug_level)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample["name"]

        # Read image
        img_path = self.img_dir / f"{img_name}.jpg"
        if not img_path.exists():
            img_path = self.img_dir / f"{img_name}.png"

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Unable to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask_path = self.mask_dir / f"{img_name}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            else:
                mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = (
            transformed["mask"].unsqueeze(0)
            if len(transformed["mask"].shape) == 2
            else transformed["mask"]
        )

        return {
            "image": image,
            "mask": mask,
            "image_name": img_name,
        }


# ==================== Loss Functions ====================


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    """Dice Loss"""
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined BCE + Dice loss"""
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# ==================== Training Functions ====================


def train_one_epoch(model, dataloader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0
    total_dice = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=images)
        logits = outputs.logits

        # Upsample to mask size
        logits = F.interpolate(
            logits, size=masks.shape[2:], mode="bilinear", align_corners=False
        )

        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        # Compute Dice
        with torch.no_grad():
            pred = torch.sigmoid(logits)
            pred_binary = (pred > 0.5).float()
            intersection = (pred_binary * masks).sum()
            dice = (2.0 * intersection) / (pred_binary.sum() + masks.sum() + 1e-8)

        total_loss += loss.item()
        total_dice += dice.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice.item():.4f}"})

    return {"loss": total_loss / len(dataloader), "dice": total_dice / len(dataloader)}


@torch.no_grad()
def validate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    total_dice = 0

    for batch in tqdm(dataloader, desc="Validation"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(
            logits, size=masks.shape[2:], mode="bilinear", align_corners=False
        )

        loss = loss_fn(logits, masks)

        # Compute Dice
        pred = torch.sigmoid(logits)
        pred_binary = (pred > 0.5).float()
        intersection = (pred_binary * masks).sum()
        dice = (2.0 * intersection) / (pred_binary.sum() + masks.sum() + 1e-8)

        total_loss += loss.item()
        total_dice += dice.item()

    return {"loss": total_loss / len(dataloader), "dice": total_dice / len(dataloader)}


# ==================== Main ====================


def main():
    parser = argparse.ArgumentParser(
        description="Progressive training on the new dataset"
    )

    # Stage argument (final training is fixed at stage4=100%)
    parser.add_argument(
        "--stage",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Stage: 1=10%, 2=20%, 3=50%, 4=100%",
    )

    # Data arguments
    parser.add_argument(
        "--forgery_dir", type=str, default="../datasets/ForgeryAnalysis_Stage_1_Train"
    )
    parser.add_argument(
        "--data_dir", type=str, default="../datasets/tianchi_2022/train"
    )
    parser.add_argument(
        "--data_analysis",
        type=str,
        default="../datasets/tianchi_2022/data_mask_analysis.json",
    )
    parser.add_argument(
        "--min_mask_ratio",
        type=float,
        default=0.01,
        help="Minimum mask ratio threshold",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Forgery validation set ratio (0 means use all data for training)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image_size", type=int, default=512)

    # Model arguments
    parser.add_argument(
        "--model", type=str, default="b3", choices=["b0", "b1", "b2", "b3"]
    )
    parser.add_argument(
        "--pretrained", type=str, default="../models/segformer_b3_pretrained.pth"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Data augmentation
    parser.add_argument(
        "--aug_level",
        type=str,
        default="geometric",
        choices=["none", "geometric", "light", "full"],
    )

    # Early Stopping
    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)

    # Output
    parser.add_argument("--output_dir", type=str, default="models/data_progressive")
    parser.add_argument("--exp_name", type=str, default=None)

    args = parser.parse_args()

    # Set sampling ratio based on stage
    stage_ratio_map = {1: 0.10, 2: 0.20, 3: 0.50, 4: 1.00}
    sample_ratio = stage_ratio_map[args.stage]

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Experiment name
    if args.exp_name is None:
        args.exp_name = f"data_stage{args.stage}_{int(sample_ratio*100)}pct"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Output] {output_dir}")
    print(f"[Exp] {args.exp_name}")
    print(f"[Stage] {args.stage}/4 (sample_ratio={sample_ratio:.0%})")

    # Model name mapping
    model_map = {
        "b0": "nvidia/mit-b0",
        "b1": "nvidia/mit-b1",
        "b2": "nvidia/mit-b2",
        "b3": "nvidia/mit-b3",
    }
    model_name = model_map[args.model]

    # Load datasets
    print("\n[Data] Loading Forgery dataset...")
    forgery_train = ForgeryDataset(
        args.forgery_dir,
        image_size=args.image_size,
        aug_level=args.aug_level,
        subset="train",
        val_split=args.val_split,
        seed=args.seed,
    )
    # When val_split=0, no validation set is created

    print(
        f"\n[Data] Loading new dataset (Stage {args.stage}, {sample_ratio:.0%} sampling)..."
    )
    data_train = FilteredDataDataset(
        args.data_dir,
        args.data_analysis,
        image_size=args.image_size,
        aug_level=args.aug_level,
        min_mask_ratio=args.min_mask_ratio,
        sample_ratio=sample_ratio,
        seed=args.seed,
    )

    # Merge training sets
    train_dataset = ConcatDataset([forgery_train, data_train])
    print(f"\n[Data] Merged training set: {len(train_dataset)} images")
    print(f"  - Forgery: {len(forgery_train)} images")
    print(f"  - Data: {len(data_train)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(
        f"\n[Data] All used for training: {len(train_dataset)} images (no validation set)"
    )

    # Load model
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"\n[Model] Loading from local pretrained model: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        config = SegformerConfig.from_dict(checkpoint["config"])
        config.num_labels = 1
        model = SegformerForSemanticSegmentation(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("[Model] Local pretrained weights loaded (strict=False)")
    else:
        print(f"\n[Model] Loading from HuggingFace: {model_name}")
        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = 1
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params / 1e6:.2f}M")

    # Loss function
    loss_fn = bce_dice_loss
    print("[Loss] BCE + Dice")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"[Optimizer] lr={args.lr}, weight_decay={args.weight_decay}")

    # Early Stopping
    early_stop = None
    if args.use_early_stopping:
        early_stop = EarlyStopping(patience=args.patience, mode="max")
        print(f"[EarlyStop] patience={args.patience}")

    # Training
    best_dice = 0.0
    best_epoch = 0
    history = []

    print(f"\n{'='*60}")
    print(f"[Train] Starting training for {args.epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        scheduler.step()

        # Record
        epoch_info = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
        }
        history.append(epoch_info)

        print(
            f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}"
        )

        # Save best model (based on training Dice)
        if train_metrics["dice"] > best_dice:
            best_dice = train_metrics["dice"]
            best_epoch = epoch
            save_path = output_dir / f"segformer_{args.model}_{args.exp_name}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "history": history,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"[Save] Best model saved (Train Dice: {best_dice:.4f})")

        # Early Stopping (based on training Dice)
        if early_stop and early_stop(train_metrics["dice"]):
            print(
                f"\n[EarlyStop] Early stopping triggered ( patience={args.patience} )"
            )
            break

    # Save final model
    final_path = output_dir / f"segformer_{args.model}_{args.exp_name}_final.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "history": history,
            "args": vars(args),
        },
        final_path,
    )

    # Save training history
    history_path = output_dir / f"history_{args.exp_name}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("[Done] Training complete!")
    print(f"[Best] Train Dice: {best_dice:.4f} (Epoch {best_epoch})")
    print(
        f"[Save] Best model: {output_dir}/segformer_{args.model}_{args.exp_name}_best.pth"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
