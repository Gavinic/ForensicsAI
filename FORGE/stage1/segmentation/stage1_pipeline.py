import argparse
import csv
import datetime
import json
import logging as logger
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from losses import MyInfoNCE
from models.hrnet import FOCAL_HRNet
from models.vit import FOCAL_ViT
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger.basicConfig(
    level=logger.INFO,
    format="%(levelname)s %(asctime)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

try:
    import albumentations as A
except ImportError:
    A = None

try:
    from torch_kmeans import KMeans
    from torch_kmeans.utils.distances import CosineSimilarity
except ImportError:
    KMeans = None
    CosineSimilarity = None

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "val", "predict"]
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument(
        "--test_dir", type=str, default="data/ForgeryAnalysis_Stage_1_Test/Image"
    )
    parser.add_argument("--answer_csv", type=str, default="")
    parser.add_argument("--output_root", type=str, default="outputs/stage1")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--sam_checkpoint", type=str, default="")
    parser.add_argument("--backbone", type=str, default="vit", choices=["vit", "hrnet"])
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_l",
        choices=["default", "vit_b", "vit_l", "vit_h"],
    )
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--gt_ratio", type=int, default=16)
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--test_bs", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
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
    args = parser.parse_args(argv)
    if args.backbone == "hrnet" and args.gt_ratio == 16:
        args.gt_ratio = 4
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def thresholding(x, thres=0.5):
    y = x.copy()
    y[y <= int(thres * 255)] = 0
    y[y > int(thres * 255)] = 255
    return y.astype(np.uint8)


def resolve_weight_path(path):
    candidate = Path(path)
    for item in [candidate, Path("weights") / path]:
        if item.exists():
            return item
    raise FileNotFoundError(f"Weights not found: {path}")


def row_value(row, keys):
    for key in keys:
        if key in row and row[key] not in [None, ""]:
            return row[key]
    raise KeyError(f"Missing columns: {keys}")


def resolve_mask_path(mask_dir, image_name):
    stem = Path(image_name).stem
    for name in [
        image_name,
        f"{stem}.png",
        f"{stem}.jpg",
        f"{stem}.jpeg",
        f"{stem}.bmp",
    ]:
        candidate = mask_dir / name
        if candidate.exists():
            return candidate
    extra = sorted(mask_dir.glob(f"{stem}.*"))
    return extra[0] if extra else None


def mask_to_coco_rle(mask):
    if mask_utils is None:
        raise ImportError(
            "pycocotools is required to export compressed RLE masks. Please install pycocotools before running val/predict export."
        )
    binary_mask = (mask > 0).astype(np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(binary_mask))
    counts = (
        encoded["counts"].decode("utf-8")
        if isinstance(encoded["counts"], bytes)
        else str(encoded["counts"])
    )
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts,
    }


def build_location_value(height, width, counts):
    return json.dumps({"size": [int(height), int(width)], "counts": counts})


def build_location_from_rle(label, height, width, rle=None):
    if int(label) == 0:
        return build_location_value(height, width, "")
    if rle is None:
        raise ValueError("Compressed RLE is required when label=1.")
    return json.dumps({"size": rle["size"], "counts": rle["counts"]})


def postprocess_mask(mask, args):
    mask = thresholding(mask)
    kernel_size = max(1, int(args.morph_kernel))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    cleaned = np.zeros_like(mask)
    min_area = max(
        int(args.min_component_area),
        int(mask.shape[0] * mask.shape[1] * args.min_component_area_ratio),
    )
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == idx] = 255
    return cleaned


def fallback_cluster_batch(features, max_iter=20):
    batch_labels = []
    for sample in features:
        sample = F.normalize(sample, dim=1)
        num_vectors = sample.shape[0]
        if num_vectors == 0:
            batch_labels.append(torch.zeros(0, dtype=torch.long, device=sample.device))
            continue
        if num_vectors == 1:
            batch_labels.append(torch.zeros(1, dtype=torch.long, device=sample.device))
            continue

        first_idx = torch.randint(0, num_vectors, (1,), device=sample.device).item()
        first_center = sample[first_idx : first_idx + 1]
        similarity = torch.matmul(sample, first_center.t()).squeeze(1)
        second_idx = int(torch.argmin(similarity).item())
        centers = torch.stack([sample[first_idx], sample[second_idx]], dim=0)
        prev_labels = None

        for _ in range(max_iter):
            similarity = torch.matmul(sample, centers.t())
            labels = torch.argmax(similarity, dim=1)
            if prev_labels is not None and torch.equal(labels, prev_labels):
                break
            prev_labels = labels
            updated_centers = []
            for cluster_id in range(2):
                cluster_vectors = sample[labels == cluster_id]
                if cluster_vectors.numel() == 0:
                    updated_centers.append(centers[cluster_id])
                else:
                    center = cluster_vectors.mean(dim=0, keepdim=True)
                    center = F.normalize(center, dim=1).squeeze(0)
                    updated_centers.append(center)
            centers = torch.stack(updated_centers, dim=0)

        batch_labels.append(prev_labels if prev_labels is not None else labels)

    return torch.stack(batch_labels, dim=0)


class Stage1Dataset(Dataset):
    def __init__(self, samples, input_size=1024, gt_ratio=16, choice="train"):
        self.samples = samples
        self.input_size = input_size
        self.gt_ratio = gt_ratio
        self.choice = choice
        self.transform = transforms.Compose([np.float32, transforms.ToTensor()])
        self.albu = (
            A.Compose(
                [
                    A.RandomScale(scale_limit=(-0.5, 0.0), p=0.75),
                    A.PadIfNeeded(
                        min_height=self.input_size, min_width=self.input_size, p=1.0
                    ),
                    A.OneOf(
                        [
                            A.HorizontalFlip(p=1),
                            A.VerticalFlip(p=1),
                            A.RandomRotate90(p=1),
                            A.Transpose(p=1),
                        ],
                        p=0.75,
                    ),
                    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.75),
                ]
            )
            if choice == "train" and A is not None
            else None
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if sample.get("mask_path", ""):
            mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(sample["mask_path"])
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        mask = thresholding(mask)
        if self.albu is not None and random.random() < 0.75:
            aug = self.albu(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        image = cv2.resize(
            image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask,
            (self.input_size // self.gt_ratio, self.input_size // self.gt_ratio),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (thresholding(mask).astype(np.float32) / 255.0)[..., None]
        image = image.astype(np.float32) / 255.0
        label = int(sample.get("label", -1))
        return (
            self.transform(image),
            torch.from_numpy(mask).float().permute(2, 0, 1),
            torch.tensor(height),
            torch.tensor(width),
            sample["image_name"],
            torch.tensor(label),
        )


def load_train_samples(args):
    data_root = Path(args.data_root)
    train_csv = Path(args.train_csv)
    black_image_dir = data_root / "ForgeryAnalysis_Stage_1_Train" / "Black" / "Image"
    black_mask_dir = data_root / "ForgeryAnalysis_Stage_1_Train" / "Black" / "Mask"
    white_image_dir = data_root / "ForgeryAnalysis_Stage_1_Train" / "White" / "Image"

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv.as_posix()}")
    if not black_image_dir.exists():
        raise FileNotFoundError(
            f"Black image directory not found: {black_image_dir.as_posix()}"
        )
    if not black_mask_dir.exists():
        raise FileNotFoundError(
            f"Black mask directory not found: {black_mask_dir.as_posix()}"
        )
    if not white_image_dir.exists():
        raise FileNotFoundError(
            f"White image directory not found: {white_image_dir.as_posix()}"
        )

    samples = []
    with train_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row_value(row, ["Image_name", "image_name"])
            label = int(row_value(row, ["label", "Label"]))
            if label == 1:
                image_path = black_image_dir / image_name
                mask_path = resolve_mask_path(black_mask_dir, image_name)
                if mask_path is None:
                    raise FileNotFoundError(f"Mask not found for {image_name}")
            else:
                image_path = white_image_dir / image_name
                mask_path = ""
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path.as_posix()}")
            samples.append(
                {
                    "image_name": image_name,
                    "label": label,
                    "image_path": image_path.as_posix(),
                    "mask_path": mask_path.as_posix() if mask_path else "",
                }
            )
    return samples


def stratified_split(samples, val_ratio, seed):
    if val_ratio <= 0:
        return list(samples), []

    rng = random.Random(seed)
    grouped = {}
    for sample in samples:
        grouped.setdefault(int(sample["label"]), []).append(sample)

    train_samples, val_samples = [], []
    for group in grouped.values():
        rng.shuffle(group)
        if len(group) <= 1:
            val_count = 0
        else:
            val_count = int(round(len(group) * val_ratio))
            if val_ratio > 0:
                val_count = max(1, val_count)
            val_count = min(len(group) - 1, val_count)
        val_samples.extend(group[:val_count])
        train_samples.extend(group[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def load_answer_labels(path):
    if not path:
        return {}

    answer_path = Path(path)
    if not answer_path.exists():
        raise FileNotFoundError(f"answer.csv not found: {answer_path.as_posix()}")

    labels = {}
    with answer_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_name = row_value(row, ["Image_name", "image_name"])
            label = int(row_value(row, ["label", "Label"]))
            if label not in (0, 1):
                raise ValueError(
                    f"Invalid label for {image_name}: {label}. Expected 0 or 1."
                )
            if image_name in labels:
                raise ValueError(
                    f"Duplicate image_name found in answer.csv: {image_name}"
                )
            labels[image_name] = label
    return labels


def validate_forced_labels(test_samples, forced_labels):
    test_names = [sample["image_name"] for sample in test_samples]
    missing = [name for name in test_names if name not in forced_labels]
    extra = [name for name in forced_labels if name not in set(test_names)]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(
            f"answer.csv is missing labels for {len(missing)} test images: {preview}{suffix}"
        )
    if extra:
        logger.warning(
            "answer.csv contains %d image names not present in test_dir. They will be ignored.",
            len(extra),
        )


def build_test_samples(args):
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir.as_posix()}")

    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [
        path
        for path in sorted(test_dir.iterdir())
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    if not files:
        raise FileNotFoundError(f"No test images found in {test_dir.as_posix()}")

    return [
        {
            "image_name": path.name,
            "label": -1,
            "image_path": path.as_posix(),
            "mask_path": "",
        }
        for path in files
    ]


def pixel_f1(prediction, groundtruth):
    pred = (prediction > 0).astype(np.uint8)
    gt = (groundtruth > 0).astype(np.uint8)
    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def metric_iou(prediction, groundtruth):
    pred = prediction > 0
    gt = groundtruth > 0
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    if intersection == 0 and union == 0:
        return 1.0
    return intersection / (union + 1e-6)


def classification_summary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    if tp == 0 and fp == 0 and fn == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = (
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
    acc = float((tp + tn) / max(1, len(y_true)))
    return {
        "image_acc": acc,
        "image_precision": float(precision),
        "image_recall": float(recall),
        "image_f1": float(f1),
    }


def prepare_run_dir(args, prefix):
    run_name = (
        args.run_name or f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_dataloader(samples, args, choice, batch_size, shuffle):
    dataset = Stage1Dataset(
        samples, input_size=args.input_size, gt_ratio=args.gt_ratio, choice=choice
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class Stage1FOCAL:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.network = self.build_network()
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.loss_fn = MyInfoNCE(metric=args.metric)
        self.clustering = (
            KMeans(verbose=False, n_clusters=2, distance=CosineSimilarity)
            if KMeans is not None
            else None
        )
        if args.weights_path:
            self.load(args.weights_path)

    def build_network(self):
        if self.args.backbone == "vit":
            network = FOCAL_ViT(
                checkpoint=self.args.sam_checkpoint or None,
                model_type=self.args.sam_model_type,
            )
        elif self.args.backbone == "hrnet":
            network = FOCAL_HRNet()
        else:
            raise ValueError(f"Unsupported backbone: {self.args.backbone}")

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            network = nn.DataParallel(network)
        return network.to(self.device)

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train_step(self, images, masks):
        self.optimizer.zero_grad()
        features = self.network(images)
        features = features.permute(0, 2, 3, 1)
        if tuple(masks.shape[-2:]) != tuple(features.shape[1:3]):
            masks = F.interpolate(masks, size=features.shape[1:3], mode="nearest")
        features = F.normalize(features, dim=3)

        losses = []
        for idx in range(features.shape[0]):
            feature_map = features[idx]
            mask = masks[idx][0] > 0.5
            pristine = feature_map[~mask]
            forged = feature_map[mask]
            if pristine.size(0) == 0 or forged.size(0) == 0:
                continue
            pristine_index = torch.randperm(pristine.size(0), device=pristine.device)[
                : min(self.args.dict_size, pristine.size(0))
            ]
            forged_index = torch.randperm(forged.size(0), device=forged.device)[
                : min(self.args.dict_size, forged.size(0))
            ]
            pristine_sample = pristine[pristine_index]
            forged_sample = forged[forged_index]
            losses.append(self.loss_fn(pristine_sample, pristine_sample, forged_sample))

        if not losses:
            return None

        loss = torch.mean(torch.stack(losses))
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def predict_step(self, images):
        with torch.no_grad():
            features = self.network(images)
            features = features.permute(0, 2, 3, 1)
            batch_size, height, width, _ = features.shape
            features = F.normalize(features, dim=3)
            features = torch.flatten(features, start_dim=1, end_dim=2)
            if self.clustering is not None:
                result = self.clustering(x=features, k=2)
                label_batch = result.labels
            else:
                label_batch = fallback_cluster_batch(features)
            masks = []
            for idx in range(batch_size):
                labels = label_batch[idx]
                if torch.sum(labels) > torch.sum(1 - labels):
                    labels = 1 - labels
                masks.append(labels.view(height, width).float().unsqueeze(0))
            return torch.stack(masks, dim=0)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(unwrap_model(self.network).state_dict(), path)
        logger.info("Saved weights to %s", path.as_posix())

    def load(self, path):
        resolved = resolve_weight_path(path)
        state = torch.load(resolved, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if (
            isinstance(state, dict)
            and "model_state_dict" in state
            and isinstance(state["model_state_dict"], dict)
        ):
            state = state["model_state_dict"]
        current_state = unwrap_model(self.network).state_dict()
        loaded_keys = 0
        for key, value in state.items():
            clean_key = key[7:] if key.startswith("module.") else key
            if (
                clean_key in current_state
                and current_state[clean_key].shape == value.shape
            ):
                current_state[clean_key] = value
                loaded_keys += 1
        unwrap_model(self.network).load_state_dict(current_state)
        logger.info("Loaded %d tensors from %s", loaded_keys, resolved.as_posix())


def run_inference(
    model, dataloader, sample_lookup, args, device, save_dir=None, forced_labels=None
):
    model.eval_mode()
    rows = []
    image_gt = []
    image_pred = []
    pixel_f1_scores = []
    pixel_iou_scores = []
    mask_dir = None

    if save_dir is not None and int(args.save_png) == 1:
        mask_dir = Path(save_dir) / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

    for batch in dataloader:
        images, _, heights, widths, image_names, labels = batch
        images = images.to(device, non_blocking=True)
        pred_masks = model.predict_step(images).squeeze(1).cpu().numpy() * 255.0

        for idx, image_name in enumerate(image_names):
            height = int(heights[idx].item())
            width = int(widths[idx].item())
            pred_resized = cv2.resize(
                pred_masks[idx], (width, height), interpolation=cv2.INTER_NEAREST
            )
            pred_mask = postprocess_mask(pred_resized, args)
            positive_ratio = float((pred_mask > 0).mean())
            forced_label = None
            if forced_labels is not None and image_name in forced_labels:
                forced_label = int(forced_labels[image_name])
            pred_label = int(positive_ratio > args.positive_ratio_threshold)
            if forced_label is not None:
                pred_label = forced_label
            if pred_label == 0:
                pred_mask = np.zeros_like(pred_mask)
            rle = None
            if pred_label == 1:
                rle = mask_to_coco_rle(pred_mask)
            rle_mask = rle["counts"] if rle is not None else ""
            location = build_location_from_rle(pred_label, height, width, rle=rle)

            if mask_dir is not None:
                cv2.imwrite(
                    (mask_dir / f"{Path(image_name).stem}.png").as_posix(), pred_mask
                )

            rows.append(
                {
                    "image_name": image_name,
                    "label": int(pred_label),
                    "location": location,
                    "height": height,
                    "width": width,
                    "rle_mask": rle_mask,
                    "positive_ratio": positive_ratio,
                }
            )

            gt_label = int(labels[idx].item())
            if gt_label >= 0:
                image_gt.append(gt_label)
                image_pred.append(int(pred_label))
                if gt_label == 1:
                    sample = sample_lookup[image_name]
                    gt_mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
                    if gt_mask is None:
                        raise FileNotFoundError(sample["mask_path"])
                    gt_mask = thresholding(gt_mask)
                    pixel_f1_scores.append(pixel_f1(pred_mask, gt_mask))
                    pixel_iou_scores.append(metric_iou(pred_mask, gt_mask))

    metrics = {
        "num_images": len(rows),
        "predicted_positive_images": int(sum(row["label"] for row in rows)),
        "mean_positive_ratio": (
            float(np.mean([row["positive_ratio"] for row in rows])) if rows else 0.0
        ),
    }
    if image_gt:
        metrics.update(classification_summary(image_gt, image_pred))
    if pixel_f1_scores:
        metrics["pixel_f1"] = float(np.mean(pixel_f1_scores))
    if pixel_iou_scores:
        metrics["pixel_iou"] = float(np.mean(pixel_iou_scores))

    score_parts = []
    if "pixel_f1" in metrics:
        score_parts.append(metrics["pixel_f1"])
    if "pixel_iou" in metrics:
        score_parts.append(metrics["pixel_iou"])
    if int(args.score_with_cls) == 1 and "image_f1" in metrics:
        score_parts.append(metrics["image_f1"])
    metrics["score"] = float(np.mean(score_parts)) if score_parts else 0.0
    return metrics, rows


def save_prediction_outputs(rows, output_dir):
    answer_rows = [
        {
            "image_name": row["image_name"],
            "label": row["label"],
            "location": row["location"],
        }
        for row in rows
    ]
    write_csv_rows(
        Path(output_dir) / "answer.csv",
        ["image_name", "label", "location"],
        answer_rows,
    )
    write_csv_rows(
        Path(output_dir) / "detailed_predictions.csv",
        [
            "image_name",
            "label",
            "location",
            "rle_mask",
            "positive_ratio",
            "height",
            "width",
        ],
        rows,
    )


def train_pipeline(args, device):
    all_samples = load_train_samples(args)
    train_samples_all, val_samples = stratified_split(
        all_samples, args.val_ratio, args.seed
    )
    train_samples = (
        [sample for sample in train_samples_all if sample["label"] == 1]
        if int(args.train_only_forged) == 1
        else train_samples_all
    )
    if not train_samples:
        raise ValueError(
            "No training samples available after filtering. Check train.csv and train_only_forged."
        )

    run_dir = prepare_run_dir(args, "train")
    save_json(vars(args), run_dir / "config.json")
    save_json(
        {
            "train_total": len(train_samples_all),
            "train_effective": len(train_samples),
            "train_forged": int(
                sum(sample["label"] == 1 for sample in train_samples_all)
            ),
            "train_real": int(
                sum(sample["label"] == 0 for sample in train_samples_all)
            ),
            "val_total": len(val_samples),
            "val_forged": int(sum(sample["label"] == 1 for sample in val_samples)),
            "val_real": int(sum(sample["label"] == 0 for sample in val_samples)),
        },
        run_dir / "split_summary.json",
    )

    train_loader = build_dataloader(train_samples, args, "train", args.train_bs, True)
    val_loader = (
        build_dataloader(val_samples, args, "val", args.test_bs, False)
        if val_samples
        else None
    )
    val_lookup = {sample["image_name"]: sample for sample in val_samples}

    model = Stage1FOCAL(args, device)
    scheduler = ReduceLROnPlateau(
        model.optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-8
    )
    best_score = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train_mode()
        epoch_losses = []
        for batch in train_loader:
            images = batch[0].to(device, non_blocking=True)
            masks = batch[1].to(device, non_blocking=True)
            loss = model.train_step(images, masks)
            if loss is not None:
                epoch_losses.append(loss)

        if not epoch_losses:
            raise RuntimeError(
                "No valid contrastive pairs were found during training. Verify the forged masks and gt_ratio setting."
            )

        train_loss = float(np.mean(epoch_losses))
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss}
        score = -train_loss

        if val_loader is not None:
            val_metrics, _ = run_inference(model, val_loader, val_lookup, args, device)
            for key, value in val_metrics.items():
                epoch_metrics[f"val_{key}"] = value
            score = float(val_metrics["score"])

        scheduler.step(score)
        model.save(run_dir / "latest.pth")
        if score > best_score:
            best_score = score
            model.save(run_dir / "best.pth")
            epoch_metrics["is_best"] = 1
        else:
            epoch_metrics["is_best"] = 0

        epoch_metrics["best_score"] = float(best_score)
        history.append(epoch_metrics)
        save_json(history, run_dir / "history.json")
        logger.info(
            "Epoch %03d/%03d train_loss=%.6f score=%.6f best=%.6f",
            epoch,
            args.epochs,
            train_loss,
            score,
            best_score,
        )

    if val_loader is not None and (run_dir / "best.pth").exists():
        best_model = Stage1FOCAL(args, device)
        best_model.load((run_dir / "best.pth").as_posix())
        best_val_dir = run_dir / "best_val"
        best_metrics, best_rows = run_inference(
            best_model, val_loader, val_lookup, args, device, save_dir=best_val_dir
        )
        save_json(best_metrics, best_val_dir / "metrics.json")
        save_prediction_outputs(best_rows, best_val_dir)

    logger.info("Training finished. Outputs saved to %s", run_dir.as_posix())
    return run_dir


def val_pipeline(args, device):
    if not args.weights_path:
        raise ValueError("weights_path is required for val mode.")

    all_samples = load_train_samples(args)
    _, val_samples = stratified_split(all_samples, args.val_ratio, args.seed)
    eval_samples = val_samples if val_samples else all_samples
    eval_loader = build_dataloader(eval_samples, args, "val", args.test_bs, False)
    eval_lookup = {sample["image_name"]: sample for sample in eval_samples}
    run_dir = prepare_run_dir(args, "val")

    save_json(vars(args), run_dir / "config.json")
    model = Stage1FOCAL(args, device)
    metrics, rows = run_inference(
        model, eval_loader, eval_lookup, args, device, save_dir=run_dir
    )
    save_json(metrics, run_dir / "metrics.json")
    save_prediction_outputs(rows, run_dir)
    logger.info("Validation finished. Outputs saved to %s", run_dir.as_posix())
    return run_dir


def predict_pipeline(args, device):
    if not args.weights_path and not args.sam_checkpoint:
        logger.warning(
            "No pretrained weights were provided. Prediction quality will be poor unless weights_path or sam_checkpoint is set."
        )

    test_samples = build_test_samples(args)
    forced_labels = load_answer_labels(args.answer_csv) if args.answer_csv else None
    if forced_labels is not None:
        validate_forced_labels(test_samples, forced_labels)
    test_loader = build_dataloader(test_samples, args, "test", args.test_bs, False)
    test_lookup = {sample["image_name"]: sample for sample in test_samples}
    run_dir = prepare_run_dir(args, "predict")

    save_json(vars(args), run_dir / "config.json")
    model = Stage1FOCAL(args, device)
    metrics, rows = run_inference(
        model,
        test_loader,
        test_lookup,
        args,
        device,
        save_dir=run_dir,
        forced_labels=forced_labels,
    )
    save_json(metrics, run_dir / "metrics.json")
    save_prediction_outputs(rows, run_dir)
    logger.info("Prediction finished. Outputs saved to %s", run_dir.as_posix())
    return run_dir


def run_mode(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(args)
    logger.info("Using device: %s", device)

    if args.mode == "train":
        return train_pipeline(args, device)
    elif args.mode == "val":
        return val_pipeline(args, device)
    elif args.mode == "predict":
        return predict_pipeline(args, device)
    raise ValueError(f"Unsupported mode: {args.mode}")


def main(argv=None):
    args = parse_args(argv)
    return run_mode(args)


if __name__ == "__main__":
    main()
