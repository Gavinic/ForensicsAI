import csv
import os
import random
from typing import Dict, List, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _safe_join(root: str, *parts: str) -> str:
    """Join path parts and guarantee the result stays inside root."""
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, *parts))
    if os.path.commonpath([root_abs, target]) != root_abs:
        raise ValueError(f"Path escapes root directory: {target}")
    return target


def _resolve_mask_path(mask_dir: str, image_name: str, strict: bool) -> str:
    """Resolve mask path for forged image using the same stem as image name."""
    stem, ext = os.path.splitext(image_name)
    candidates = [
        _safe_join(mask_dir, image_name),
        _safe_join(mask_dir, f"{stem}.png"),
        _safe_join(mask_dir, f"{stem}{ext}"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    if strict:
        raise FileNotFoundError(f"Mask file not found for {image_name} in {mask_dir}")
    return candidates[1]


def _validate_file(path: str, strict: bool, desc: str) -> None:
    if strict and not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def build_stage1_train_records(
    data_root: str,
    train_csv_name: str = "train.csv",
    adv_label: int = 0,
    strict: bool = True,
) -> List[Dict[str, object]]:
    """
    Build records for BasicDataloader from ForgeryAnalysis_Stage_1_Train, auto-scan images without csv.
    Label 1: Black/Image (forged images, with masks under Black/Mask)
    Label 0: White/Image (real images, no masks)
    """
    black_image_dir = _safe_join(
        data_root, "ForgeryAnalysis_Stage_1_Train", "Black", "Image"
    )
    black_mask_dir = _safe_join(
        data_root, "ForgeryAnalysis_Stage_1_Train", "Black", "Mask"
    )
    white_image_dir = _safe_join(
        data_root, "ForgeryAnalysis_Stage_1_Train", "White", "Image"
    )

    if not os.path.isdir(black_image_dir):
        raise FileNotFoundError(f"Black image directory not found: {black_image_dir}")
    if not os.path.isdir(white_image_dir):
        raise FileNotFoundError(f"White image directory not found: {white_image_dir}")

    records: List[Dict[str, object]] = []

    # Read forged images (label=1)
    for image_name in os.listdir(black_image_dir):
        ext = os.path.splitext(image_name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        forgery_path = _safe_join(black_image_dir, image_name)
        gt_mask_path = _resolve_mask_path(black_mask_dir, image_name, strict=strict)
        records.append(
            {
                "forgery_path": forgery_path,
                "gt_mask_path": gt_mask_path,
                "forged_label": 1,
                "adv_label": adv_label,
                "image_name": image_name,
            }
        )

    # Read real images (label=0)
    for image_name in os.listdir(white_image_dir):
        ext = os.path.splitext(image_name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        forgery_path = _safe_join(white_image_dir, image_name)
        records.append(
            {
                "forgery_path": forgery_path,
                "gt_mask_path": "",
                "forged_label": 0,
                "adv_label": adv_label,
                "image_name": image_name,
            }
        )

    if strict and len(records) == 0:
        raise ValueError(
            f"No training images found in {black_image_dir} or {white_image_dir}"
        )

    print(
        f"Auto-scan of training set complete: {len(records)} images total, {sum(1 for r in records if r['forged_label'] == 1)} positive samples, {sum(1 for r in records if r['forged_label'] == 0)} negative samples"
    )
    return records


def build_stage1_test_records(
    data_root: str,
    adv_label: int = 0,
    strict: bool = True,
) -> List[Dict[str, object]]:
    """
    Build records for BasicDataloader from ForgeryAnalysis_Stage_1_Test/Image.
    forged_label is set to 0 as a placeholder for inference-only usage.
    """
    test_image_dir = _safe_join(data_root, "ForgeryAnalysis_Stage_1_Test", "Image")
    if strict and not os.path.isdir(test_image_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_image_dir}")

    image_names = []
    if os.path.isdir(test_image_dir):
        for name in os.listdir(test_image_dir):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_names.append(name)
    image_names.sort()

    if strict and len(image_names) == 0:
        raise ValueError(f"No test images found in {test_image_dir}")

    records = [
        {
            "forgery_path": _safe_join(test_image_dir, image_name),
            "gt_mask_path": "",
            "forged_label": 0,
            "adv_label": adv_label,
            "image_name": image_name,
        }
        for image_name in image_names
    ]
    return records


def split_train_val_records(
    records: List[Dict[str, object]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Stratified split by forged_label, preserving old training/inference behavior.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    if val_ratio == 0.0 or len(records) <= 1:
        return records, []

    by_label: Dict[int, List[Dict[str, object]]] = {0: [], 1: []}
    for record in records:
        label = int(record["forged_label"])
        by_label.setdefault(label, []).append(record)

    rng = random.Random(seed)
    train_records: List[Dict[str, object]] = []
    val_records: List[Dict[str, object]] = []

    for label_records in by_label.values():
        if len(label_records) == 0:
            continue
        rng.shuffle(label_records)
        val_count = int(len(label_records) * val_ratio)
        if val_count == 0 and len(label_records) > 1:
            val_count = 1
        val_records.extend(label_records[:val_count])
        train_records.extend(label_records[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def split_train_val_records_kfold(
    records: List[Dict[str, object]],
    num_folds: int = 5,
    fold_index: int = 0,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Stratified k-fold split by forged_label.
    """
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2 for k-fold splitting")
    if not 0 <= fold_index < num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds - 1}]")

    by_label: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        label = int(record["forged_label"])
        by_label.setdefault(label, []).append(record)

    rng = random.Random(seed)
    folds: List[List[Dict[str, object]]] = [[] for _ in range(num_folds)]

    for label_records in by_label.values():
        rng.shuffle(label_records)
        for idx, record in enumerate(label_records):
            folds[idx % num_folds].append(record)

    val_records = list(folds[fold_index])
    train_records: List[Dict[str, object]] = []
    for idx, fold_records in enumerate(folds):
        if idx == fold_index:
            continue
        train_records.extend(fold_records)

    if len(train_records) == 0 or len(val_records) == 0:
        raise ValueError(
            "Insufficient samples for k-fold split. "
            "Reduce num_folds or provide more data."
        )

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records
