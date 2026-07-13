import gc
import json
import os
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm

# Import vllm-related libraries
from vllm import LLM, SamplingParams

# --- Config ---
# NOTE: make sure you have run the merge script, and this path is the storage path of the merged model
BASE_MODEL_ID = "./models/Qwen3-VL-8B-Merged"
# BASE_MODEL_ID = "<BASE_PATH>/Qwen3-VL-8B-Merged"
TEST_IMAGE_DIR = "./data/ForgeryAnalysis_Stage_2_Test/Image"
OUTPUT_FILE = "./data/result/result_fusai.json"

# PROMPT_TEXT = """You are a top-tier image forensics and forgery analysis expert, proficient in digital tampering detection (especially scene text, receipts/documents) and in-depth identification of AIGC-generated images. Perform an end-to-end authenticity assessment of the input image and output the result strictly in JSON format.
# [Output requirements]
# You must output exactly one valid JSON object containing the following three fields:
# 1. "boxes": a 2D list. If judged as forged, provide the list of bounding boxes for all tampered/generated anomaly regions, in the format [[x_min, y_min, x_max, y_max], ...]; coordinate values must be integers scaled to the 0-1000 range. If judged as a real image, output an empty list [].
# 2. "label": an integer. 0 means "real image", 1 means "forged image" (including AIGC generation, local tampering, erasure-rewrite, splicing, etc.).
# 3. "explanation": a string. Provide a professional, rigorous, structured, and logically clear natural-language forensic report.
# [Explanation writing spec and reasoning chain]
# Be sure to organize the explanation text strictly according to the following layered structure, and adopt the corresponding writing logic based on authenticity (label):
# Layer 1: Overall verdict
# - Forged (Label=1): state the conclusion directly (e.g. "This is a forged gas-station receipt produced by digital tampering" or "This is a forged digital image produced by artificial intelligence").
# - Real (Label=0): explicitly confirm authenticity (e.g. "This is a real photo of a shop storefront, with no signs of digital forgery or post-processing tampering").
# Layer 2: Tampering localization and content mapping (only required for forged images)
# - List each key anomaly/tampered region one by one, and bind the normalized coordinates to the content precisely within the text.
# - Format reference: "There are [N] key tampered/forged regions in the image, whose coordinates and content are respectively: [semantic description] [x1, y1, x2, y2] content is '[specific text/structure]'; ...".
# Layer 3: Visual feature and low-level signal analysis
# - For forged features:
#   - Text tampering: unnatural mixing of dot-matrix/thermal fonts with sans-serif smooth fonts; vertical baseline misalignment; abnormal stroke thickness; lack of the "ink-bleed" or "burrs" that real ink should leave on paper; smooth background smearing, pixelated noise breaks, and clone-stamp artifacts left by erasure-repair.
#   - AIGC hallucinations: meaningless repeated consonants (e.g. Swwwwn), unreadable garbled text, characters "melting" or sticking together; "waxy" skin and over-smoothing; "perfect mirror symmetry" that violates real-world tolerances; resolution inversion (e.g. blurry face but sharp hands); floating feel from shadow/light-source mismatch.
# - For real features:
#   - Describe the naturalness of physical interaction: natural perspective deformation of the text baseline following paper folds; consistency of global illumination and environmental reflection; uniformity of digital noise distribution; the real overlay of physical flaws (stains, tears) and ink.
# Layer 4: Logic and common-sense cross-checking
# - Receipt/invoice-specific checklist: perform rigorous mathematical self-consistency and business common-sense checks, focusing on the following common forgery loopholes:
#   1. Abnormal decimal places: check whether the amount violates the common sense of a specific country's currency (e.g. three decimal places in regular retail transactions for Malaysian ringgit, RMB, etc.).
#   2. Tax-rate/amount contradiction: check whether the tax calculation is correct (especially a 0% tax-rate item being wrongly computed with a non-zero tax, or a fixed rate such as 6% not matching the figure on the receipt).
#   3. Arithmetic addition errors: rigorously check whether unit price x quantity equals the item total, whether the sum of all item subtotals equals the subtotal/total, and whether the payment amount and change match.
#   4. Spatio-temporal and date fallacies: check whether the date format is valid (e.g. "month 24", "29/24" and other non-existent dates), and whether the year is anachronistic relative to the policy recorded on the receipt (e.g. GST tax system) or a specific era.
# - AIGC/scene common-sense checks: verify the clarity principle of commercial logos (a blurry or garbled sponsor logo violates business logic); the semantic reasonableness of brand naming; whether the physical structure conforms to mechanical principles (e.g. an unsupported cantilevered building).
# Layer 5: Concluding statement
# - Forged: point out the tampering intent (fabricating transactions, inflating reimbursements, building a false identity, etc.) and state that the document "lacks authenticity and credibility / is inadmissible".
# - Real: summarize the self-consistency of the logic and state that "based on comprehensive analysis, this image truthfully records the original state / the actually existing scene".
# """

PROMPT_TEXT = """You are a top-tier image forensics and forgery analysis expert, especially proficient in tampering detection and AIGC-generation identification for "scene text images (all kinds of receipts, certificates, signs, and natural-scene text)". Perform an end-to-end authenticity assessment of the input image and output the result strictly in JSON format.

[Output requirements]
You must output exactly one valid JSON object containing the following three fields:
1. "boxes": a 2D list. If judged as forged, provide the list of bounding boxes for all tampered/generated anomaly regions, in the format [[x_min, y_min, x_max, y_max], ...]; coordinate values must be integers scaled to the 0-1000 range. If judged as a real image, output an empty list [].
2. "label": an integer. 0 means "real image", 1 means "forged image" (including AIGC generation, local text tampering, erasure-rewrite, splicing, etc.).
3. "explanation": a string. Provide a professional, rigorous, structured, and logically clear natural-language forensic report.

[Explanation writing spec and reasoning chain]
Be sure to organize the explanation text strictly according to the following four layers:

Layer 1: Overall verdict
- State the nature of the image directly.

Layer 2: Tampering localization and content mapping (Grounding alignment)
- List each key tampered region one by one, and bind the normalized coordinates to the tampered content precisely within the text.
- Format reference: "There are [N] key tampered regions in the image, whose coordinates and content are respectively: [semantic description] [x1, y1, x2, y2] content is '[specific text]'; ...".

Layer 3: Visual and low-level feature analysis (Visual Forensics)
- For receipts/documents focus on: differences between tampered digits and the original text in font weight (bolder/blacker), font style (e.g. whether the decimal point is a square pixel block or round), edge transition (whether it lacks the faint graininess that dot-matrix/thermal printing should have, or is too smooth), and layout alignment (vertical baseline misalignment, unnatural spacing).
- For scene/AIGC focus on: whether text rendering is affected by scene lighting (whether it appears as an independent planar light emitter or floating), whether perspective transformation conforms to the physical bumps of the 3D surface, and whether there is unidentifiable distorted garbled text (AIGC hallucination characters).
- Background and artifact analysis: whether there are unnatural smooth, blurry, or smearing marks around tampered regions (erasure-repair artifacts), and whether the original paper-fiber texture, environmental noise, or JPEG compression ghosting is destroyed. If a human body is involved, point out anatomical distortions (e.g. fused fingers, disproportionate ratios, waxy feel).

Layer 4: Logic and common-sense cross-checking (Logical Consistency)
- Logic computation: verify the mathematical self-consistency of the data inside the receipt (e.g. whether unit price x quantity equals the total, whether the sum of all subtotals equals the grand total, and whether the tax calculation is correct).
- Industry common sense: check whether the text content conforms to the norms of the specific scenario (e.g. Malaysian ringgit should keep two rather than three decimal places, whether specific leading zeros are reasonable, and business tax-rate common sense).
- Physical laws: analyze whether the shadow projection direction matches the light source, whether the support structure conforms to mechanical common sense, and whether reflective surfaces (e.g. water reflections) show continuity contradictions.

Layer 5: Concluding statement
- Briefly describe the possible intent of the forgery (e.g. fabricating transaction records for false reimbursement, creating misleading information, etc.) and give the final untrustworthy conclusion.
"""

# PROMPT_TEXT = """You are a top-tier image forensics and forgery analysis expert, especially proficient in tampering detection and AIGC-generation identification for "scene text images (all kinds of receipts, certificates, signs, and natural-scene text)". Perform an end-to-end authenticity assessment of the input image and output the result strictly in JSON format.

# [Output requirements]
# You must output exactly one valid JSON object containing the following three fields:
# 1. "boxes": a 2D list. If judged as forged, provide the list of bounding boxes for all tampered/generated anomaly regions, in the format [[x_min, y_min, x_max, y_max], ...]; coordinate values must be integers scaled to the 0-1000 range. If judged as a real image, output an empty list [].
# 2. "label": an integer. 0 means "real image", 1 means "forged image" (including AIGC generation, local text tampering, erasure-rewrite, splicing, etc.).
# 3. "explanation": a string. Provide a professional, rigorous, structured, and logically clear natural-language forensic report.

# [Explanation writing spec and reasoning chain]
# Be sure to organize the explanation text strictly according to the following four layers:

# Layer 1: Overall verdict
# - State the nature of the image directly.

# Layer 2: Tampering localization and content mapping (Grounding alignment)
# - List each key tampered region one by one, and bind the normalized coordinates to the tampered content precisely within the text.
# - Format reference: "There are [N] key tampered regions in the image, whose coordinates and content are respectively: [semantic description] [x1, y1, x2, y2] content is '[specific text]'; ...".

# Layer 3: Visual and low-level feature analysis (Visual Forensics)
# - For mobile/computer UI screenshots, memes, financial transaction records, digital posters, or physical signs, keep a "tends to be real" prior assumption for such Chinese digital-native images. Unless you find conclusive evidence that violates low-level pixel laws (e.g. abrupt noise breaks), severe logical self-contradictions, or obvious erasure/splicing traces.
# - For receipts/documents focus on: differences between tampered digits and the original text in font weight (bolder/blacker), font style (e.g. whether the decimal point is a square pixel block or round), edge transition (whether it lacks the faint graininess that dot-matrix/thermal printing should have, or is too smooth), and layout alignment (vertical baseline misalignment, unnatural spacing).
# - For scene/AIGC focus on: whether text rendering is affected by scene lighting (whether it appears as an independent planar light emitter or floating), whether perspective transformation conforms to the physical bumps of the 3D surface, and whether there is unidentifiable distorted garbled text (AIGC hallucination characters).
# - Background and artifact analysis: whether there are unnatural smooth, blurry, or smearing marks around tampered regions (erasure-repair artifacts), and whether the original paper-fiber texture, environmental noise, or JPEG compression ghosting is destroyed. If a human body is involved, point out anatomical distortions (e.g. fused fingers, disproportionate ratios, waxy feel).

# Layer 4: Logic and common-sense cross-checking (Logical Consistency)
# - Logic computation: verify the mathematical self-consistency of the data inside the receipt (e.g. whether unit price x quantity equals the total, whether the sum of all subtotals equals the grand total, and whether the tax calculation is correct).
# - Industry common sense: check whether the text content conforms to the norms of the specific scenario (e.g. Malaysian ringgit should keep two rather than three decimal places, whether specific leading zeros are reasonable, and business tax-rate common sense).
# - Physical laws: analyze whether the shadow projection direction matches the light source, whether the support structure conforms to mechanical common sense, and whether reflective surfaces (e.g. water reflections) show continuity contradictions.

# Layer 5: Concluding statement
# - Briefly describe the possible intent of the forgery (e.g. fabricating transaction records for false reimbursement, creating misleading information, etc.) and give the final untrustworthy conclusion.
# """


def main():
    # ==================== Initialize the vLLM engine (merged-model mode) ====================
    print(f"Initializing the vLLM engine from {BASE_MODEL_ID}...")

    # Init parameters: no LoRA configuration needed anymore
    llm = LLM(
        model=BASE_MODEL_ID,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 1},
        max_model_len=7000,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        allowed_local_media_path=TEST_IMAGE_DIR,
        max_num_seqs=64,
        swap_space=12,
        mm_processor_kwargs={
            "max_pixels": 5000 * 32 * 32,
            "min_pixels": 3136,
        },
        # Removed enable_lora, max_loras, max_lora_rank and other LoRA-specific parameters
    )

    # Sampling parameter config
    sampling_params = SamplingParams(
        temperature=0.4,  # ensure deterministic output
        max_tokens=5000,
    )

    # ==================== Get the test image list ====================
    test_images = []
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"Error: test image directory does not exist {TEST_IMAGE_DIR}")
        return

    for filename in os.listdir(TEST_IMAGE_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            test_images.append(os.path.join(TEST_IMAGE_DIR, filename))
    test_images.sort()
    print(f"Number of test images: {len(test_images)}")

    # ==================== Batch processing ====================
    BATCH_SIZE = 700
    all_results = []

    print("Starting batch processing...")
    for i in range(0, len(test_images), BATCH_SIZE):
        batch_images = test_images[i : i + BATCH_SIZE]
        print(
            f"Processing batch {i // BATCH_SIZE + 1}/{(len(test_images) - 1) // BATCH_SIZE + 1}"
        )

        messages_list = []
        for img_path in tqdm(batch_images, desc=f"Building batch data"):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"file://{img_path}"},
                            },
                            {"type": "text", "text": PROMPT_TEXT},
                        ],
                    }
                ]
                messages_list.append(messages)
            except Exception as e:
                print(f"Failed to process path {img_path}: {e}")
                continue

        if not messages_list:
            continue

        # ==================== vLLM inference (normal inference mode) ====================
        try:
            # The chat call no longer needs the lora_request parameter
            outputs = llm.chat(messages=messages_list, sampling_params=sampling_params)

            # Collect results
            for img_path, output in zip(batch_images, outputs):
                generated_text = output.outputs[0].text.strip()
                submission_item = {
                    "img_path": os.path.basename(img_path),
                    "explanation": generated_text,
                }
                all_results.append(submission_item)
        except Exception as e:
            print(f"Severe error during batch inference: {e}")

        # After each batch, you may manually clean up memory (optional)
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== Save results ====================
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nInference complete!")
    print(f"Total successfully processed: {len(all_results)} / {len(test_images)}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
