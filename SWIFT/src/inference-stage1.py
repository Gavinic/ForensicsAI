from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BOOL_TRUE = {"1", "true", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "no", "n", "off"}

DEFAULT_USER_QUERY = """ Perform forgery detection on this image and output a forensic report. Output specification and format requirements:
• You must output only one JSON object containing the three fields label, bbox, and explanation.
• Coordinate specification: in this task all coordinates must be normalized to the [0, 1000] range.
• "label": 0 = real, 1 = forged.
• "bbox": a list of normalized coordinates for all tampered regions, in the format [[x1, y1, x2, y2], ...].
• "explanation": a rigorous forensic statement. It must be a single coherent natural-language paragraph; lists are strictly forbidden.
Format reference: {"label": 0 or 1, "bbox":[[x1, y1, x2, y2], ...], "explanation": "..."}
Do not output any additional text. Do not insert line breaks.
If the image is a bill or invoice, typical digital-tampering signals include: three decimal places appearing, mismatches in item quantities or total amounts, fonts that clearly differ from the surrounding text, and times or dates that defy common sense. Some calculation hints: for a bill with a 6% GST rate, taxed_total ÷ 1.06 gives the pre-tax amount, and pre-tax_amount × 0.06 gives the tax amount. As long as the quantity, amount, tax-rate, and time calculations show no anomalies, the bill is real; do not be overly strict about local font details.
If the image contains a person or object, it is likely AIGC-generated. For people, first check whether the fingers are deformed, then check the face, eyes, nose, ears, and mouth for AI-generation artifacts, and finally inspect details such as jewelry. For cars, focus on the license plate and windows. For race cars, focus on the text on the car and anomalies in the tires.
If the image is mainly text with no people or objects in the surroundings, typical digital-tampering signals are: some words altered into meaningless tokens, with visible font inconsistency or semantic incoherence; typical AI-generation signals are: cartoonish fonts and unnatural lighting/shadows in the environment.
If the image is a Chinese street-scene photo, it is most likely real; do not be too strict."""

torch = None
Image = None
InferRequest = None
RequestConfig = None
TransformersEngine = None


def parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in BOOL_TRUE:
        return True
    if text in BOOL_FALSE:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL inference and save raw predictions for the submission normalizer.",
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Test-set root directory or the image directory itself.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Raw inference CSV path. The output is designed to be consumed by norm-checkpoint2submission.py.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=os.getenv("CHECKPOINT_PATH", str(pick_default_checkpoint())),
        help="LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--base_model_path",
        default=os.getenv(
            "BASE_MODEL_PATH", str(PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct")
        ),
        help="Base model directory.",
    )
    parser.add_argument(
        "--engine_max_batch_size",
        type=int,
        default=int(os.getenv("ENGINE_MAX_BATCH_SIZE", "60")),
    )
    parser.add_argument(
        "--initial_batch_size",
        type=int,
        default=int(os.getenv("INITIAL_BATCH_SIZE", "60")),
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=int(os.getenv("MIN_BATCH_SIZE", "32")),
    )
    parser.add_argument(
        "--batch_growth_step",
        type=int,
        default=int(os.getenv("BATCH_GROWTH_STEP", "8")),
    )
    parser.add_argument(
        "--growth_reserved_ratio",
        type=float,
        default=float(os.getenv("GROWTH_RESERVED_RATIO", "0.72")),
    )
    parser.add_argument(
        "--shrink_reserved_ratio",
        type=float,
        default=float(os.getenv("SHRINK_RESERVED_RATIO", "0.90")),
    )
    parser.add_argument(
        "--oom_shrink_ratio",
        type=float,
        default=float(os.getenv("OOM_SHRINK_RATIO", "0.8")),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.4")),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=float(os.getenv("TOP_P", "0.8")),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=int(os.getenv("TOP_K", "20")),
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=int(os.getenv("MAX_TOKENS", "2048")),
    )
    parser.add_argument(
        "--retry_max_tokens",
        type=int,
        default=int(os.getenv("RETRY_MAX_TOKENS", "3072")),
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=int(os.getenv("MAX_PIXELS", "1003520")),
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument(
        "--attn_impl",
        default=os.getenv("ATTN_IMPL", "sdpa"),
    )
    parser.add_argument(
        "--sort_by_effective_pixels",
        type=parse_bool,
        default=parse_bool(os.getenv("SORT_BY_EFFECTIVE_PIXELS", "true")),
    )
    parser.add_argument(
        "--limit_images",
        type=int,
        default=int(os.getenv("LIMIT_IMAGES", "0")),
    )
    parser.add_argument(
        "--sleep_after_oom_seconds",
        type=int,
        default=int(os.getenv("SLEEP_AFTER_OOM_SECONDS", "3")),
    )
    parser.add_argument(
        "--user_query_path",
        default=os.getenv("USER_QUERY_PATH", ""),
        help="Optional external prompt file. If omitted, the built-in prompt is used.",
    )
    parser.add_argument(
        "--pytorch_cuda_alloc_conf",
        default=os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    )
    return parser.parse_args()


def pick_default_checkpoint() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "adapters" / "stage2_final" / "checkpoint-4200",
        PROJECT_ROOT / "models" / "adapters" / "stage1_preliminary" / "checkpoint-2600",
    ]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file() and (
            candidate / "adapter_model.safetensors"
        ).is_file():
            return candidate
    return candidates[0]


def resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def has_image_files(directory: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in directory.iterdir()
    )


def resolve_image_dir(input_path: str) -> Path:
    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return path.parent
        raise ValueError(
            f"input_path must be a test-set directory or an image directory, got file: {path}"
        )

    if has_image_files(path):
        return path

    for candidate_name in ("Image", "image", "images", "Images"):
        candidate = path / candidate_name
        if candidate.is_dir() and has_image_files(candidate):
            return candidate

    raise FileNotFoundError(
        "No image directory found under input_path. Expected images directly inside the path "
        "or in a child directory named Image/image/images."
    )


def load_runtime_dependencies() -> None:
    global torch, Image, InferRequest, RequestConfig, TransformersEngine

    import torch as torch_module
    from PIL import Image as pil_image_module
    from swift.infer_engine import InferRequest as infer_request_cls
    from swift.infer_engine import RequestConfig as request_config_cls
    from swift.infer_engine import TransformersEngine as transformers_engine_cls

    torch = torch_module
    Image = pil_image_module
    InferRequest = infer_request_cls
    RequestConfig = request_config_cls
    TransformersEngine = transformers_engine_cls

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_user_query(user_query_path: str) -> str:
    if user_query_path:
        return resolve_path(user_query_path).read_text(encoding="utf-8").strip()
    return DEFAULT_USER_QUERY


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return "1" if int(value) == 1 else "0"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "fake", "forged", "tampered", "manipulated"}:
            return "1"
        if text in {"0", "real", "authentic", "true"}:
            return "0"
        try:
            return "1" if int(float(text)) == 1 else "0"
        except Exception:
            return "0"
    return "0"


def normalize_bbox_list(raw_bbox: Any) -> List[List[int]]:
    if isinstance(raw_bbox, str):
        raw_bbox = raw_bbox.strip()
        if not raw_bbox:
            return []
        try:
            raw_bbox = json.loads(raw_bbox)
        except Exception:
            try:
                raw_bbox = ast.literal_eval(raw_bbox)
            except Exception:
                return []

    boxes: List[List[int]] = []

    def walk(value: Any) -> None:
        if (
            isinstance(value, (list, tuple))
            and len(value) == 4
            and all(isinstance(item, (int, float, str)) for item in value)
        ):
            try:
                x1, y1, x2, y2 = (int(float(item)) for item in value)
            except Exception:
                return
            boxes.append(
                [
                    max(0, min(1000, x1)),
                    max(0, min(1000, y1)),
                    max(0, min(1000, x2)),
                    max(0, min(1000, y2)),
                ]
            )
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(raw_bbox)
    return boxes


def parse_response_json(raw_text: str) -> Tuple[str, str, str, bool]:
    text = (raw_text or "").strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    candidate: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            candidate = parsed
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    candidate = parsed
            except Exception:
                candidate = None

    if candidate is None:
        return "0", "[]", text.replace("\r", " ").replace("\n", " ").strip(), False

    label = normalize_label(candidate.get("label", 0))
    bbox = normalize_bbox_list(candidate.get("bbox", []))
    explanation = (
        str(candidate.get("explanation", ""))
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )
    return label, json.dumps(bbox, ensure_ascii=False), explanation, True


def load_done(output_csv: Path) -> set[str]:
    done: set[str] = set()
    if not output_csv.exists():
        return done

    try:
        with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "file" not in reader.fieldnames:
                return done
            for row in reader:
                file_name = (row.get("file") or "").strip()
                if file_name:
                    done.add(file_name)
    except Exception:
        return set()

    return done


def get_effective_pixels(width: int, height: int, max_pixels: int) -> int:
    if width <= 0 or height <= 0:
        return max_pixels
    return min(width * height, max_pixels)


def iter_image_paths(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def build_pending_records(
    image_dir: Path,
    done: set[str],
    max_pixels: int,
    sort_by_effective_pixels: bool,
    limit_images: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for image_path in iter_image_paths(image_dir):
        if image_path.name in done:
            continue

        width = 0
        height = 0
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            pass

        records.append(
            {
                "file": image_path.name,
                "path": str(image_path),
                "width": width,
                "height": height,
                "effective_pixels": get_effective_pixels(width, height, max_pixels),
            }
        )

    if sort_by_effective_pixels:
        records.sort(
            key=lambda item: (
                item["effective_pixels"],
                item["height"],
                item["width"],
                item["file"],
            )
        )

    if limit_images > 0:
        records = records[:limit_images]

    return records


def build_request_config(
    config: Dict[str, Any], max_tokens: Optional[int] = None
) -> Any:
    return RequestConfig(
        max_tokens=max_tokens if max_tokens is not None else config["max_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
    )


def is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def cleanup_cuda(delay_seconds: int) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def get_memory_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "free_gb": free_bytes / 1024**3,
        "total_gb": total_bytes / 1024**3,
        "peak_allocated_gb": peak_allocated / 1024**3,
        "peak_reserved_gb": peak_reserved / 1024**3,
        "peak_reserved_ratio": peak_reserved / max(total_bytes, 1),
    }


def infer_batch(
    engine: Any,
    batch_records: Sequence[Dict[str, Any]],
    user_query: str,
    request_config: Any,
):
    requests = [
        InferRequest(
            messages=[{"role": "user", "content": user_query}], images=[item["path"]]
        )
        for item in batch_records
    ]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    responses = engine.infer(requests, request_config=request_config, use_tqdm=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return responses, get_memory_stats()


def extract_raw_text(response: Any) -> str:
    return (
        response.choices[0].message.content
        if hasattr(response, "choices")
        else str(response)
    )


def retry_single_if_needed(
    engine: Any,
    record: Dict[str, Any],
    user_query: str,
    retry_request_config: Optional[Any],
    label: str,
    bbox: str,
    explanation: str,
    json_ok: bool,
) -> Tuple[str, str, str, bool]:
    if json_ok or retry_request_config is None:
        return label, bbox, explanation, json_ok

    print(
        f"[retry] {record['file']} json_ok=0, retry with max_tokens={retry_request_config.max_tokens}"
    )
    retry_response, _ = infer_batch(engine, [record], user_query, retry_request_config)
    return parse_response_json(extract_raw_text(retry_response[0]))


def adjust_batch_size(
    config: Dict[str, Any],
    current_batch_size: int,
    batch_size: int,
    memory_stats: Dict[str, float],
) -> int:
    next_batch_size = current_batch_size
    peak_reserved_ratio = memory_stats.get("peak_reserved_ratio")
    if peak_reserved_ratio is None:
        return next_batch_size

    if (
        batch_size == current_batch_size
        and peak_reserved_ratio < config["growth_reserved_ratio"]
    ):
        next_batch_size = min(
            config["engine_max_batch_size"],
            current_batch_size + config["batch_growth_step"],
        )
    elif peak_reserved_ratio > config["shrink_reserved_ratio"]:
        next_batch_size = max(
            config["min_batch_size"], current_batch_size - config["batch_growth_step"]
        )
    return next_batch_size


def ensure_output_header(output_csv: Path) -> bool:
    return (not output_csv.exists()) or output_csv.stat().st_size == 0


def validate_runtime_paths(config: Dict[str, Any]) -> None:
    required_paths = {
        "base_model_path": Path(config["base_model"]),
        "checkpoint_path": Path(config["ckpt"]),
        "image_dir": Path(config["image_dir"]),
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    image_dir = resolve_image_dir(args.input_path)
    output_csv = resolve_path(args.output_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    return {
        "ckpt": str(resolve_path(args.checkpoint_path)),
        "base_model": str(resolve_path(args.base_model_path)),
        "image_dir": str(image_dir),
        "output_csv": str(output_csv),
        "engine_max_batch_size": args.engine_max_batch_size,
        "initial_batch_size": args.initial_batch_size,
        "min_batch_size": args.min_batch_size,
        "batch_growth_step": args.batch_growth_step,
        "growth_reserved_ratio": args.growth_reserved_ratio,
        "shrink_reserved_ratio": args.shrink_reserved_ratio,
        "oom_shrink_ratio": args.oom_shrink_ratio,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "retry_max_tokens": args.retry_max_tokens,
        "max_pixels": args.max_pixels,
        "cuda_visible_devices": args.cuda_visible_devices,
        "attn_impl": args.attn_impl,
        "sort_by_effective_pixels": args.sort_by_effective_pixels,
        "limit_images": args.limit_images,
        "sleep_after_oom_seconds": args.sleep_after_oom_seconds,
        "user_query_path": args.user_query_path,
        "pytorch_cuda_alloc_conf": args.pytorch_cuda_alloc_conf,
    }


def write_empty_header_if_needed(output_csv: Path) -> None:
    if ensure_output_header(output_csv):
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "label", "bbox", "explanation", "json_ok"])


def main() -> None:
    args = parse_args()
    config = build_config(args)

    os.environ["MAX_PIXELS"] = str(config["max_pixels"])
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", config["pytorch_cuda_alloc_conf"])

    load_runtime_dependencies()
    validate_runtime_paths(config)
    user_query = load_user_query(config["user_query_path"])

    output_csv = Path(config["output_csv"])
    done = load_done(output_csv)
    pending_records = build_pending_records(
        image_dir=Path(config["image_dir"]),
        done=done,
        max_pixels=config["max_pixels"],
        sort_by_effective_pixels=config["sort_by_effective_pixels"],
        limit_images=config["limit_images"],
    )

    if not pending_records:
        write_empty_header_if_needed(output_csv)
        print(f"[resume] done={len(done)}, pending=0")
        print("Done.")
        return

    engine = TransformersEngine(
        model=config["base_model"],
        adapters=[config["ckpt"]],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_batch_size=config["engine_max_batch_size"],
        attn_impl=config["attn_impl"],
    )

    request_config = build_request_config(config, config["max_tokens"])
    retry_request_config = None
    if config["retry_max_tokens"] > config["max_tokens"]:
        retry_request_config = build_request_config(config, config["retry_max_tokens"])

    current_batch_size = max(config["min_batch_size"], config["initial_batch_size"])
    current_batch_size = min(config["engine_max_batch_size"], current_batch_size)
    need_header = ensure_output_header(output_csv)

    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if need_header:
            writer.writerow(["file", "label", "bbox", "explanation", "json_ok"])

        print(
            "[resume] "
            f"done={len(done)}, pending={len(pending_records)}, "
            f"engine_max_batch_size={config['engine_max_batch_size']}, "
            f"initial_batch_size={current_batch_size}, "
            f"max_tokens={config['max_tokens']}"
        )

        progress = tqdm(total=len(pending_records), dynamic_ncols=True)
        cursor = 0

        while cursor < len(pending_records):
            batch_size = min(current_batch_size, len(pending_records) - cursor)
            batch_records = pending_records[cursor : cursor + batch_size]

            try:
                responses, memory_stats = infer_batch(
                    engine, batch_records, user_query, request_config
                )
            except Exception as exc:
                if is_oom_error(exc) and batch_size > config["min_batch_size"]:
                    new_batch_size = max(
                        config["min_batch_size"],
                        int(batch_size * config["oom_shrink_ratio"]),
                    )
                    if new_batch_size >= batch_size:
                        new_batch_size = batch_size - 1
                    print(
                        f"[oom] batch_size={batch_size} -> {new_batch_size}, clear cache and retry"
                    )
                    current_batch_size = min(
                        config["engine_max_batch_size"],
                        max(config["min_batch_size"], new_batch_size),
                    )
                    cleanup_cuda(config["sleep_after_oom_seconds"])
                    continue
                raise

            for record, response in zip(batch_records, responses):
                raw_text = extract_raw_text(response)
                label, bbox, explanation, json_ok = parse_response_json(raw_text)
                label, bbox, explanation, json_ok = retry_single_if_needed(
                    engine=engine,
                    record=record,
                    user_query=user_query,
                    retry_request_config=retry_request_config,
                    label=label,
                    bbox=bbox,
                    explanation=explanation,
                    json_ok=json_ok,
                )
                writer.writerow(
                    [record["file"], label, bbox, explanation, int(json_ok)]
                )
                done.add(record["file"])

            handle.flush()
            cursor += len(batch_records)
            progress.update(len(batch_records))

            max_pixels_in_batch = max(
                item["effective_pixels"] for item in batch_records
            )
            print(
                "[batch] "
                f"size={len(batch_records)}, "
                f"max_effective_pixels={max_pixels_in_batch}, "
                f"peak_reserved={memory_stats.get('peak_reserved_gb', 0):.2f}GB, "
                f"free_after={memory_stats.get('free_gb', 0):.2f}GB"
            )

            current_batch_size = adjust_batch_size(
                config, current_batch_size, len(batch_records), memory_stats
            )
            current_batch_size = min(
                config["engine_max_batch_size"], current_batch_size
            )

        progress.close()

    print("Done.")


if __name__ == "__main__":
    main()
