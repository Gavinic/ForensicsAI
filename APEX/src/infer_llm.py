import glob
import json
import os
from collections import defaultdict
from pathlib import Path

from peft import PeftModel
from PIL import Image, ImageOps
from swift import get_model_processor, get_template
from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine


def is_cuda_device_assert(error_text: str) -> bool:
    lower = error_text.lower()
    return "device-side assert triggered" in lower or "cudaerrorassert" in lower


def is_failed_output(md_path: str) -> bool:
    if not os.path.exists(md_path) or os.path.getsize(md_path) == 0:
        return False
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            head = f.read(1024)
        lower = head.lower()
        return ("INFERENCE_ERROR" in head) or ("device-side assert triggered" in lower)
    except Exception:
        return False


def sanitize_bbox_xywh_to_xyxy(bbox, img_w, img_h):
    """
    Input:
        bbox: [x, y, w, h]
    Output:
        (pixel_xyxy, text_xyxy)
        - pixel_xyxy: [x1, y1, x2, y2], float, passed to the framework
        - text_xyxy:  [x1, y1, x2, y2], int, used to write into the prompt
    Returns None if invalid.
    """
    if bbox is None or len(bbox) != 4:
        return None

    x, y, w, h = bbox

    try:
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
    except Exception:
        return None

    # width and height must be positive
    if w <= 0 or h <= 0:
        return None

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # clip to image bounds
    x1 = max(0.0, min(x1, float(img_w - 1)))
    y1 = max(0.0, min(y1, float(img_h - 1)))
    x2 = max(0.0, min(x2, float(img_w)))
    y2 = max(0.0, min(y2, float(img_h)))

    # re-check after clipping
    if not (x2 > x1 and y2 > y1):
        return None

    pixel_xyxy = [x1, y1, x2, y2]
    text_xyxy = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    return pixel_xyxy, text_xyxy


def process_single_fallback(
    engine, request_config, img_path, filename, md_path, i, total
):
    """Extracted single-image no-box fallback handler"""
    print(f"[{i}/{total}] Note: {filename} falling back to no-box inference ...")
    fallback_request = InferRequest(
        messages=[
            {
                "role": "user",
                "content": (
                    "<image>Analyze whether this image shows signs of digital forgery, "
                    "local editing, or post-processing tampering. "
                    "If no anomaly is found, explain why it conforms to a real-capture "
                    "or natural-image distribution."
                ),
            }
        ],
        images=[img_path],
    )
    try:
        resp_list = engine.infer([fallback_request], request_config=request_config)
        response_text = resp_list[0].choices[0].message.content
        print(
            f"[{i}/{total}] Note: {filename} automatically fell back to no-box inference and succeeded."
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(response_text)
        return True, response_text
    except Exception as e2:
        err_text = str(e2)
        response_text = f"No-box retry also failed: {err_text}"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(response_text)
        return False, response_text


def main():
    # ================= Configuration =================
    root_dir = Path(__file__).parent.parent.absolute()
    print(root_dir)
    test_img_dir = os.path.join(root_dir, "tmp_dir/Image")

    test_json_path = os.path.join(root_dir, "tmp_dir/cleaned_annotations.json")
    output_md_dir = os.path.join(root_dir, "tmp_dir/Output_Caption")

    base_model_path = os.path.join(root_dir, "models/Qwen3.5-9B")

    adapter_dir = os.path.join(root_dir, "models/adapters/checkpoint-2500")

    bbox_type = os.getenv("BBOX_TYPE", "real").strip().lower()
    max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    overwrite_failed_only = os.getenv("OVERWRITE_FAILED_ONLY", "1").strip() == "1"

    # Batch size configuration
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    # =================================================

    if bbox_type not in {"real", "norm1000"}:
        raise ValueError(
            f"Unsupported BBOX_TYPE: {bbox_type}, only real or norm1000 are supported"
        )
    if max_tokens <= 0:
        raise ValueError(f"MAX_TOKENS must be a positive integer, got: {max_tokens}")
    if temperature < 0:
        raise ValueError(f"TEMPERATURE cannot be negative, got: {temperature}")
    if batch_size <= 0:
        raise ValueError(f"BATCH_SIZE must be a positive integer, got: {batch_size}")

    print("=" * 80)
    print("0. Current path configuration:")
    print(f"   TEST_IMG_DIR          : {test_img_dir}")
    print(f"   TEST_JSON_PATH        : {test_json_path}")
    print(f"   OUTPUT_MD_DIR         : {output_md_dir}")
    print(f"   BASE_MODEL_PATH       : {base_model_path}")
    print(f"   ADAPTER_DIR           : {adapter_dir}")
    print(f"   BBOX_TYPE             : {bbox_type}")
    print(f"   MAX_TOKENS            : {max_tokens}")
    print(f"   TEMPERATURE           : {temperature}")
    print(f"   OVERWRITE_FAILED_ONLY : {overwrite_failed_only}")
    print(f"   BATCH_SIZE            : {batch_size}")
    print("=" * 80)

    if not os.path.isdir(test_img_dir):
        raise FileNotFoundError(f"Test image directory does not exist: {test_img_dir}")
    if not os.path.isdir(base_model_path):
        raise FileNotFoundError(f"Model directory does not exist: {base_model_path}")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    if not os.path.isfile(test_json_path):
        print(
            f"Warning: bounding-box JSON not found, will infer in no-box mode: {test_json_path}"
        )

    os.makedirs(output_md_dir, exist_ok=True)

    print("1. Parsing test-set JSON ...")
    img_to_raw_bboxes = defaultdict(list)
    if os.path.exists(test_json_path):
        with open(test_json_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        for item in predictions:
            image_id = item.get("image_id", None)
            bbox = item.get("bbox", None)
            if image_id is not None and bbox is not None:
                img_to_raw_bboxes[image_id].append(bbox)

    print("2. Loading model and weights ...")
    model, processor = get_model_processor(base_model_path)
    model = PeftModel.from_pretrained(model, adapter_dir)
    template = get_template(processor, enable_thinking=False)
    engine = TransformersEngine(model, template=template)

    print(f"3. Starting batch inference (Batch Size = {batch_size})...")
    request_config = RequestConfig(max_tokens=max_tokens, temperature=temperature)

    all_test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))
    if not all_test_imgs:
        raise RuntimeError(f"No images found in directory: {test_img_dir}")

    total = len(all_test_imgs)
    success_count = 0
    skip_count = 0
    fail_count = 0

    # collect tasks for the current batch
    current_batch_requests = []
    current_batch_meta = (
        []
    )  # stores the corresponding img_path, filename, md_path, idx, valid_pixel_bboxes

    def execute_batch(requests, meta_list):
        nonlocal success_count, fail_count
        if not requests:
            return

        try:
            # run batch inference
            resp_list = engine.infer(requests, request_config=request_config)

            # whole batch succeeded
            for idx_in_batch, resp in enumerate(resp_list):
                response_text = resp.choices[0].message.content
                meta = meta_list[idx_in_batch]
                with open(meta["md_path"], "w", encoding="utf-8") as f:
                    f.write(response_text)
                success_count += 1

        except Exception as e:
            err_text = str(e)

            # if it is a CUDA assert, the whole process must die immediately
            if is_cuda_device_assert(err_text):
                err_msg = (
                    f"\nDetected CUDA device-side assert. "
                    f"This error typically corrupts the current CUDA context, so "
                    f"subsequent in-process inference results may be unreliable. "
                    f"Please restart the Python process to continue, and consider "
                    f"setting CUDA_LAUNCH_BLOCKING=1 to locate the first triggering sample."
                )
                print(err_msg)

                # write the error into every file of the current batch for diagnosis
                for meta in meta_list:
                    with open(meta["md_path"], "w", encoding="utf-8") as f:
                        f.write(f"INFERENCE_ERROR: {err_text}\n{err_msg}")

                raise RuntimeError(
                    "Detected CUDA device-side assert, program aborted."
                ) from e

            # if it is not a CUDA assert, a single image may have caused the whole batch to fail
            print(
                f"Warning: batch execution failed, splitting into single-image fallback retries... error: {err_text}"
            )

            # split into single-image fallback retries
            for idx_in_batch, meta in enumerate(meta_list):
                single_request = requests[idx_in_batch]
                try:
                    # first retry the single image as-is
                    single_resp = engine.infer(
                        [single_request], request_config=request_config
                    )
                    response_text = single_resp[0].choices[0].message.content
                    with open(meta["md_path"], "w", encoding="utf-8") as f:
                        f.write(response_text)
                    success_count += 1
                except Exception as single_e:
                    # single-image retry also failed; if in boxed mode, fall back to no-box retry
                    if len(meta["valid_pixel_bboxes"]) > 0:
                        success, _ = process_single_fallback(
                            engine,
                            request_config,
                            meta["img_path"],
                            meta["filename"],
                            meta["md_path"],
                            meta["idx"],
                            total,
                        )
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                    else:
                        with open(meta["md_path"], "w", encoding="utf-8") as f:
                            f.write(f"INFERENCE_ERROR: {str(single_e)}")
                        fail_count += 1

    # main loop
    for i, img_path in enumerate(all_test_imgs, start=1):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        md_path = os.path.join(output_md_dir, f"{name_no_ext}.md")

        # skip already-successful outputs
        if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
            if overwrite_failed_only:
                if not is_failed_output(md_path):
                    print(
                        f"[{i}/{total}] {filename} already has a valid result, skipping."
                    )
                    skip_count += 1
                    continue

        # read image dimensions
        try:
            with Image.open(img_path) as img:
                img = ImageOps.exif_transpose(img)
                img_w, img_h = img.size
        except Exception as e:
            print(f"[{i}/{total}] Warning: image read failed, skipping {filename}: {e}")
            fail_count += 1
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"INFERENCE_ERROR: image read failed: {str(e)}")
            continue

        raw_bboxes = img_to_raw_bboxes.get(filename, [])
        valid_pixel_bboxes = []
        valid_text_bboxes = []

        for bbox in raw_bboxes:
            out = sanitize_bbox_xywh_to_xyxy(bbox, img_w, img_h)
            if out is None:
                continue
            pixel_xyxy, text_xyxy = out

            if bbox_type == "norm1000":
                x1, y1, x2, y2 = pixel_xyxy
                pixel_xyxy = [
                    x1 / float(img_w) * 1000.0,
                    y1 / float(img_h) * 1000.0,
                    x2 / float(img_w) * 1000.0,
                    y2 / float(img_h) * 1000.0,
                ]

            valid_pixel_bboxes.append(pixel_xyxy)
            valid_text_bboxes.append(text_xyxy)

        # build request
        if len(valid_pixel_bboxes) > 0:
            abs_bboxes_text = "、".join(
                [f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in valid_text_bboxes]
            )
            bbox_placeholders = "、".join(["<bbox>"] * len(valid_pixel_bboxes))

            prompt_text = (
                f"<image>There are {len(valid_pixel_bboxes)} suspicious regions in the image, "
                f"with absolute pixel coordinates: {abs_bboxes_text}. "
                f"The corresponding visual positions are: {bbox_placeholders}. "
                f"Please focus on these regions and analyze whether there are signs of "
                f"digital forgery, local editing, or post-processing tampering, "
                f"and provide an explainable basis for the judgment. "
                f"If you reference coordinates, use the absolute pixel coordinates provided here."
            )

            infer_request = InferRequest(
                messages=[{"role": "user", "content": prompt_text}],
                images=[img_path],
                objects={"bbox": valid_pixel_bboxes, "bbox_type": bbox_type},
            )
            run_mode = (
                f"boxed mode, bbox_num={len(valid_pixel_bboxes)}, bbox_type={bbox_type}"
            )
        else:
            prompt_text = (
                "<image>Analyze whether this image shows signs of digital forgery, "
                "local editing, or post-processing tampering. "
                "If no anomaly is found, explain why it conforms to a real-capture "
                "or natural-image distribution."
            )
            infer_request = InferRequest(
                messages=[{"role": "user", "content": prompt_text}], images=[img_path]
            )
            run_mode = "no-box mode"

        print(f"[{i}/{total}] Queued {filename} for batch processing | {run_mode}")

        current_batch_requests.append(infer_request)
        current_batch_meta.append(
            {
                "img_path": img_path,
                "filename": filename,
                "md_path": md_path,
                "idx": i,
                "valid_pixel_bboxes": valid_pixel_bboxes,
            }
        )

        # when accumulated tasks reach batch_size, run inference
        if len(current_batch_requests) == batch_size:
            execute_batch(current_batch_requests, current_batch_meta)
            # clear the current batch
            current_batch_requests = []
            current_batch_meta = []

    # after the loop, handle any remaining requests that do not fill a full batch
    if current_batch_requests:
        execute_batch(current_batch_requests, current_batch_meta)

    print("\n" + "=" * 80)
    print("All done!")
    print(f"Output directory: {output_md_dir}")
    print(f"Total images   : {total}")
    print(f"Succeeded      : {success_count}")
    print(f"Skipped        : {skip_count}")
    print(f"Failed         : {fail_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
