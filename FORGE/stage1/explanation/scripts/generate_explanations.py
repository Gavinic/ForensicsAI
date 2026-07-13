from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.explainer.runtime_env import sanitize_thread_env

sanitize_thread_env()

from src.explainer.prompting import (
    SYSTEM_PROMPT,
    build_forgery_instruction,
    build_grounded_rewrite_instruction,
    build_real_instruction,
    fallback_explanation,
)
from src.explainer.rle_utils import decode_location_to_mask
from src.explainer.vision_features import (
    extract_forgery_evidence,
    extract_real_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chinese image-forgery-detection explanations from answer.csv"
    )
    default_workers = max(1, min(8, os.cpu_count() or 1))

    parser.add_argument(
        "--answer-csv",
        type=str,
        default=str(PROJECT_ROOT / "answer-P50-P51.csv"),
        help="Input answer.csv path (must contain image_name,label,location)",
    )
    parser.add_argument(
        "--test-image-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "ForgeryAnalysis_Stage_1_Test" / "Image"),
        help="Test set image directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(PROJECT_ROOT / "answer_with_explanation.csv"),
        help="Output CSV path (adds an explanation column)",
    )

    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Base model name or local path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "explainer_qlora" / "adapter"),
        help="LoRA adapter path; if it does not exist, only the base model is loaded",
    )

    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Do not call the LLM; use rule-based fallback text directly",
    )
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Load in 4-bit quantization",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit",
    )
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load in 8-bit quantization (mutually exclusive with 4-bit)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=["float16", "bf16", "bfloat16"],
    )
    parser.add_argument(
        "--enable-vl",
        action="store_true",
        help="Call a VL model during inference to add global and local observations",
    )
    parser.add_argument(
        "--vl-model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VL model name or local path",
    )
    parser.add_argument(
        "--vl-load-in-4bit",
        dest="vl_load_in_4bit",
        action="store_true",
        help="Load VL model in 4-bit",
    )
    parser.add_argument(
        "--no-vl-load-in-4bit",
        dest="vl_load_in_4bit",
        action="store_false",
        help="Disable VL model 4-bit",
    )
    parser.set_defaults(vl_load_in_4bit=True)
    parser.add_argument(
        "--vl-load-in-8bit",
        action="store_true",
        help="Load VL model in 8-bit (mutually exclusive with 4-bit)",
    )
    parser.add_argument(
        "--vl-torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bf16", "bfloat16"],
    )
    parser.add_argument(
        "--vl-batch-size",
        type=int,
        default=4,
        help="VL batch inference size, used to improve GPU utilization",
    )
    parser.add_argument(
        "--vl-trust-remote-code",
        action="store_true",
        help="Allow the VL model to load remote custom code",
    )
    parser.add_argument(
        "--vl-max-regions",
        type=int,
        default=3,
        help="Maximum number of suspicious regions to observe per image",
    )
    parser.add_argument(
        "--vl-context-ratio",
        type=float,
        default=0.35,
        help="Context expansion ratio when cropping local regions",
    )
    parser.add_argument(
        "--vl-min-crop-size",
        type=int,
        default=224,
        help="Minimum side length for local crops",
    )

    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=1,
        help="LLM batch inference size; 1 means generate one at a time",
    )
    parser.add_argument(
        "--grounded-rewrite",
        dest="grounded_rewrite",
        action="store_true",
        help="Perform a second grounded rewrite on the draft",
    )
    parser.add_argument(
        "--no-grounded-rewrite",
        dest="grounded_rewrite",
        action="store_false",
        help="Disable grounded rewrite",
    )
    parser.set_defaults(grounded_rewrite=True)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=default_workers,
        help="Number of parallel threads for evidence extraction (defaults to CPU auto-detection)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Process only the first N samples when debugging",
    )

    parser.add_argument(
        "--evidence-jsonl",
        type=str,
        default="",
        help="Optional: write each sample's evidence and instruction to a jsonl for auditing",
    )

    return parser.parse_args()


def _resolve_image_path(test_image_dir: Path, image_name: str) -> Path:
    image_path = test_image_dir / image_name
    if image_path.exists():
        return image_path

    stem = Path(image_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = test_image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return image_path


def _load_generator(args: argparse.Namespace):
    if args.disable_llm:
        return None, None

    from src.explainer.modeling import load_explainer_model

    adapter_path = Path(args.adapter_path)
    use_adapter = str(adapter_path) if adapter_path.exists() else None

    tokenizer, model = load_explainer_model(
        model_name_or_path=args.model_name_or_path,
        adapter_path=use_adapter,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=args.torch_dtype,
        device_map="auto",
        trust_remote_code=False,
    )
    return tokenizer, model


def _load_vl_observer(args: argparse.Namespace):
    if not args.enable_vl:
        return None

    from src.explainer.vision_language import VisionLanguageObserver

    return VisionLanguageObserver(
        model_name_or_path=args.vl_model_name_or_path,
        load_in_4bit=args.vl_load_in_4bit,
        load_in_8bit=args.vl_load_in_8bit,
        torch_dtype=args.vl_torch_dtype,
        device_map="auto",
        trust_remote_code=args.vl_trust_remote_code,
        batch_size=args.vl_batch_size,
    )


def _generate_text(
    tokenizer: Any,
    model: Any,
    instruction: str,
    args: argparse.Namespace,
) -> str:
    if tokenizer is None or model is None:
        return ""

    from src.explainer.modeling import generate_with_model

    return generate_with_model(
        tokenizer=tokenizer,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        instruction=instruction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


def _generate_text_batch(
    tokenizer: Any,
    model: Any,
    instructions: Sequence[str],
    args: argparse.Namespace,
) -> List[str]:
    if tokenizer is None or model is None:
        return ["" for _ in instructions]

    from src.explainer.modeling import generate_with_model_batch

    return generate_with_model_batch(
        tokenizer=tokenizer,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        instructions=instructions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def _finalize_text(text: str, sample: Dict[str, Any]) -> str:
    output = _normalize_text(text)
    evidence = sample.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}

    if int(sample.get("label", 0)) == 1:
        boxes = evidence.get("boxes", [])
        if boxes and ("coordinates" not in output and "[" not in output):
            output = f"{output} Suspicious region coordinates: {boxes}.".strip()
    return output


def _prepare_sample(
    row: Dict[str, Any],
    test_image_dir: Path,
    vl_observer: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    image_name = str(row.get("image_name", "")).strip()
    label = int(row.get("label", 0))
    image_path = _resolve_image_path(test_image_dir, image_name)

    if not image_path.exists():
        evidence = {
            "image_size": [0, 0],
            "tamper_ratio": 0.0,
            "boxes": [],
            "cues": ["Test image is missing; cannot extract evidence"],
            "metrics": {},
        }
        return {
            "image_name": image_name,
            "label": label,
            "instruction": "",
            "evidence": evidence,
            "vl_observation": {},
            "preset_explanation": fallback_explanation(
                label=label, image_name=image_name, evidence=evidence
            ),
        }

    try:
        if label == 1:
            mask = decode_location_to_mask(row.get("location", ""))
            evidence = extract_forgery_evidence(str(image_path), mask_array=mask)
        else:
            evidence = extract_real_evidence(str(image_path))

        try:
            vl_observation = (
                vl_observer.observe(
                    image_path=str(image_path),
                    evidence=evidence,
                    max_regions=args.vl_max_regions,
                    context_ratio=args.vl_context_ratio,
                    min_crop_size=args.vl_min_crop_size,
                )
                if vl_observer is not None
                else {}
            )
        except Exception:
            vl_observation = {}

        if vl_observation:
            evidence = dict(evidence)
            evidence["vl_observation"] = vl_observation

        if label == 1:
            instruction = build_forgery_instruction(
                image_name, evidence, vl_observation=vl_observation
            )
        else:
            instruction = build_real_instruction(
                image_name, evidence, vl_observation=vl_observation
            )

        return {
            "image_name": image_name,
            "label": label,
            "instruction": instruction,
            "evidence": evidence,
            "vl_observation": vl_observation,
            "preset_explanation": "",
        }
    except Exception as exc:
        evidence = {
            "image_size": [0, 0],
            "tamper_ratio": 0.0,
            "boxes": [],
            "cues": [f"Explanation generation error: {exc}"],
            "metrics": {},
        }
        return {
            "image_name": image_name,
            "label": label,
            "instruction": "",
            "evidence": evidence,
            "vl_observation": {},
            "preset_explanation": fallback_explanation(
                label=label, image_name=image_name, evidence=evidence
            ),
        }


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _run_generation_pass(
    tokenizer: Any,
    model: Any,
    instructions: Sequence[str],
    args: argparse.Namespace,
    desc: str,
) -> List[str]:
    if tokenizer is None or model is None:
        return ["" for _ in instructions]
    if not instructions:
        return []

    generated_texts = ["" for _ in instructions]
    dynamic_batch_size = max(1, int(args.gen_batch_size))
    progress = tqdm(total=len(instructions), desc=desc)
    cursor = 0

    while cursor < len(instructions):
        current_bs = min(dynamic_batch_size, len(instructions) - cursor)
        batch_instructions = list(instructions[cursor : cursor + current_bs])

        try:
            if current_bs == 1:
                batch_outputs = [
                    _generate_text(tokenizer, model, batch_instructions[0], args)
                ]
            else:
                batch_outputs = _generate_text_batch(
                    tokenizer, model, batch_instructions, args
                )

            for local_idx, text in enumerate(batch_outputs):
                generated_texts[cursor + local_idx] = text

            cursor += current_bs
            progress.update(current_bs)

        except Exception as exc:
            if _is_oom_error(exc) and current_bs > 1:
                dynamic_batch_size = max(1, current_bs // 2)
                print(
                    f"[Warning] Batch inference triggered an out-of-memory error; automatically reducing batch_size to {dynamic_batch_size} and retrying."
                )
                try:
                    import torch as _torch

                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            for local_idx, instruction in enumerate(batch_instructions):
                try:
                    generated_texts[cursor + local_idx] = _generate_text(
                        tokenizer, model, instruction, args
                    )
                except Exception:
                    generated_texts[cursor + local_idx] = ""
                progress.update(1)

            cursor += current_bs

    progress.close()
    return generated_texts


def main() -> None:
    args = parse_args()

    answer_path = Path(args.answer_csv)
    test_image_dir = Path(args.test_image_dir)
    output_path = Path(args.output_csv)

    if not answer_path.exists():
        raise FileNotFoundError(f"answer.csv does not exist: {answer_path}")
    if not test_image_dir.exists():
        raise FileNotFoundError(
            f"Test image directory does not exist: {test_image_dir}"
        )

    df = pd.read_csv(answer_path)
    required_cols = {"image_name", "label", "location"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Input file is missing required columns: {required_cols}")

    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    tokenizer = model = None
    if not args.disable_llm:
        try:
            tokenizer, model = _load_generator(args)
            print("Explanation generation model loaded.")
        except Exception as exc:
            print(
                f"[Warning] Model loading failed; falling back to rule-based explanation: {exc}"
            )
            tokenizer = model = None

    vl_observer = None
    if args.enable_vl:
        try:
            vl_observer = _load_vl_observer(args)
            print("VL observation model loaded.")
        except Exception as exc:
            print(
                f"[Warning] VL model loading failed; only statistical evidence will be used: {exc}"
            )
            vl_observer = None

    evidence_writer = None
    if args.evidence_jsonl:
        evidence_path = Path(args.evidence_jsonl)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_writer = evidence_path.open("w", encoding="utf-8")

    rows = df.to_dict(orient="records")
    preprocess_workers = max(1, int(args.preprocess_workers))
    if vl_observer is not None and preprocess_workers > 1:
        print(
            "[Note] VL observation is enabled; to avoid instability from concurrent thread reuse of the model, evidence extraction is automatically switched to single-threaded."
        )
        preprocess_workers = 1

    if preprocess_workers > 1:
        with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
            prepared_samples = list(
                tqdm(
                    executor.map(
                        lambda item: _prepare_sample(
                            item, test_image_dir, vl_observer, args
                        ),
                        rows,
                    ),
                    total=len(rows),
                    desc="Extract evidence",
                )
            )
    else:
        prepared_samples = [
            _prepare_sample(item, test_image_dir, vl_observer, args)
            for item in tqdm(rows, desc="Extract evidence")
        ]

    generated_texts = [""] * len(prepared_samples)
    pending_indices = [
        idx
        for idx, sample in enumerate(prepared_samples)
        if not sample.get("preset_explanation") and sample.get("instruction")
    ]

    if tokenizer is not None and model is not None and pending_indices:
        first_pass_outputs = _run_generation_pass(
            tokenizer=tokenizer,
            model=model,
            instructions=[
                prepared_samples[idx]["instruction"] for idx in pending_indices
            ],
            args=args,
            desc="LLM generation",
        )
        for idx, text in zip(pending_indices, first_pass_outputs):
            generated_texts[idx] = text

        if args.grounded_rewrite:
            rewrite_indices = [
                idx for idx in pending_indices if _normalize_text(generated_texts[idx])
            ]
            rewrite_instructions = [
                build_grounded_rewrite_instruction(
                    image_name=prepared_samples[idx]["image_name"],
                    label=prepared_samples[idx]["label"],
                    evidence=prepared_samples[idx].get("evidence", {}),
                    draft=generated_texts[idx],
                    vl_observation=prepared_samples[idx].get("vl_observation", {}),
                )
                for idx in rewrite_indices
            ]
            rewritten_outputs = _run_generation_pass(
                tokenizer=tokenizer,
                model=model,
                instructions=rewrite_instructions,
                args=args,
                desc="Grounded rewrite",
            )
            for idx, text in zip(rewrite_indices, rewritten_outputs):
                if _normalize_text(text):
                    generated_texts[idx] = text

    explanations: List[str] = []
    for idx, sample in enumerate(prepared_samples):
        image_name = sample["image_name"]
        label = sample["label"]
        instruction = sample.get("instruction", "")
        evidence = sample.get("evidence", {})

        text = sample.get("preset_explanation", "") or generated_texts[idx]
        if not text or len(text.strip()) < 30:
            text = fallback_explanation(
                label=label, image_name=image_name, evidence=evidence
            )

        text = _finalize_text(text, sample)

        if evidence_writer is not None:
            evidence_writer.write(
                json.dumps(
                    {
                        "image_name": image_name,
                        "label": label,
                        "instruction": instruction,
                        "evidence": evidence,
                        "vl_observation": sample.get("vl_observation", {}),
                        "explanation": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        explanations.append(text)

    if evidence_writer is not None:
        evidence_writer.close()

    output_df = df.copy()
    output_df["explanation"] = explanations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Explanation generation complete: {output_path}")
    print(f"Total samples processed: {len(output_df)}")


if __name__ == "__main__":
    main()
