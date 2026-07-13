import os

# Force V0 engine and fix socket issues
os.environ["VLLM_USE_V1"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Force distributed to use localhost
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

import argparse
import re

import numpy as np
import polars as pl
from vllm import LLM, SamplingParams

print(f"VLLM Version: {__import__('vllm').__version__}")

# Define the base few-shot messages (from original notebook headers)
MESSAGES_BASE = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """
            # Role
            You are a world-class digital image forensics expert. You possess deep knowledge of physical optics, computer vision, typography and printing, and semantic logic. Analyze images for forgery detection (forged? authentic?) and localization, and produce an explainable attribution description. Input images can be categorized as high-finance, social, education, receipt, notice, and natural-scene images; focus on character forgery, tampering and generation within the image, or distortions of digitally generated human hands, ears, and eyes.
            Your writing must be "evidence-oriented, causally closed, and stylistically consistent", and strictly follow the output protocol:
            - Output only <描述> ... </描述>; do not output chain-of-thought, step numbers, or list markers.
            - Tone: objective, clinical, evidence-oriented. Frequently use: consistent / matches / violates / abnormal / abrupt / broken / discontinuous / suggests / indicates.
            - The structural order is fixed: first a qualitative conclusion -> then lock the coordinate region -> then micro evidence -> then macro logic -> then infer means and motive -> close with "in summary".
            - Do not fabricate information not visible in the image; if text is unreadable, describe it as "illegible / unreadable garbled text"; when people / institutions / locations are uncertain, use "suspected / possibly".
            Please perform a judicial-grade digital forensic analysis of the current image. Use your multimodal perception capabilities to examine the integrity of the image from surface to core, and output the final forensic description following the established six-step structural protocol.
             # Goal
            Refer to the [detection conclusion] provided by the user (authentic or forged, and the coordinates of forged regions). Combined with the visual content of the image, generate a detailed, professional, and logically rigorous attribution description.
            # Analysis Dimensions (core analysis dimensions)
            When generating the description, you may analyze from the following dimensions:
            1. **Digital signal features**: check the consistency of sensor noise, compression artifacts, and the smoothness/saw-tooth feel of edge pixels.
            2. **Physical optics features**: check lighting direction, shadow logic, reflection/refraction (especially for transparent media), focal plane and depth of field.
            3. **Visual consistency**: check whether fonts/font weights are unified, whether ink/texture is continuous, and the fusion of background and foreground (whether there is feathering or cut traces).
            4. **Semantic and logical consistency**:
                - **Text logic**: spelling, grammar, industry terminology, typographic norms (alignment/baseline).
                - **Common-sense logic**: mathematical calculations (tax/total), geographic information (address/landmark), time/season features, currency format, etc.
            # Style Guidelines
            - **Tone**: objective, clinical, evidence-oriented. Use precise words such as "consistent", "matches", "violates", "abnormal", "abrupt".
            - **Structure**: first state the conclusion, then list micro evidence (visual + physical), then macro evidence (logic + common sense), and finally synthesize the conclusion.
            - **High-frequency evidence words** (your corpus prefers these): abnormally smooth / background texture broken / missing noise / discontinuous compression artifact pattern / edge sharp as if knife-cut / halo / pasted-on feel / baseline misalignment / inconsistent font weight / perspective mismatch / inconsistent lighting
            - **Receipt-specific high-frequency**: calculation errors and common-sense contradictions, decimal places, tax rate/GST, invalid date format (e.g. contradiction between tax amount and tax rate)
            Below are 3 image-text examples of attribution descriptions. Analyze only the style and reasoning logic of the examples, but do not copy their content!:
            """,
            },
        ],
    },
]


def get_init_sentens(text):
    match = re.match(r"^([^。]+。)([^。]+。)", text)
    sentence1 = match.group(1).strip()
    # Second sentence (to extract coordinates)
    sentence2 = match.group(2).strip()
    pattern = r"\[\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*\]"
    # 2. Extract coordinates from the second sentence
    # \[\d+,\s*\d+,\s*\d+,\s*\d+\] specifically matches brackets containing four digits and commas
    coordinates = re.findall(pattern, sentence1)
    if not coordinates:
        coordinates = re.findall(pattern, sentence2)
        if coordinates:
            res = (
                sentence1
                + f"There are {len(coordinates)} key tampered regions in the image, with coordinates and contents:"
                + "，".join(coordinates)
            )
        else:
            res = sentence1
    else:
        res = (
            sentence1.split("，")[0]
            + f". There are {len(coordinates)} key tampered regions in the image, with coordinates and contents:"
            + "，".join(coordinates)
        )
    return res


def main():
    argss = args()
    # Load Data
    input_csv = argss.csv_file
    print(f"Loading data from {input_csv}...")
    df = pl.read_csv(input_csv)

    # Initialize VLLM
    # Note: Qwen3-VL usually works with the qwen2_vl architecture in current VLLM versions if not explicitly supported,
    # but let's try auto detection.
    print("Initializing VLLM model...")
    llm = LLM(
        model=argss.vlm_model,  # "Qwen/Qwen3-VL-8B-Instruct",#"Qwen/Qwen3-VL-8B-Instruct", #
        trust_remote_code=True,
        enable_prefix_caching=True,
        tensor_parallel_size=1,  # Adjust based on available GPUs
        limit_mm_per_prompt={"image": 5},  # Allow multiple images in history
        max_model_len=84240,  # Limit context length to fit in GPU memory
        allowed_local_media_path="/",  # Allow loading images from local filesystem
        #     mm_processor_kwargs={
        #     "max_pixels": 2048 * 2048,  # limit the maximum pixels (approximately 1024x1024)
        #     # "min_pixels": 256 * 256     # limit the minimum pixels
        # }
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2478,  # increased token limit just in case
        stop_token_ids=None,
    )

    prompts = []
    vector_db_root = argss.vector_db_root
    ## Load the train captions
    train_captions = np.load(os.path.join(vector_db_root, "caption.npy"))
    ## Load the train paths
    train_paths = np.load(os.path.join(vector_db_root, "image_path.npy"))
    # Iterate and prepare prompts
    print("Preparing prompts...")
    for row in list(df.iter_rows(named=True)):
        # Create a deep copy of base messages for each request
        # We need to reconstruct the list to avoid modifying the original references if we were to mutate dictionaries,
        # but here we just append to the list.
        current_messages = MESSAGES_BASE.copy()
        ## Read the captions and image paths according to the indices
        indexs = row["indexs"]
        for ind, index in enumerate(indexs.split(",")[:3]):
            index = int(index)
            current_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{train_paths[index]}"},
                        },
                        {
                            "type": "text",
                            "text": f"""example{ind}:Detection conclusion:{get_init_sentens(train_captions[index])}Attribution description:<描述>{train_captions[index]}</描述>
                """,
                        },
                    ],
                }
            )
        ## Encode based on the already-generated captions
        # Format image path
        image_path = f"{argss.image_path}/{row['image_name']}"
        image_url = f"file://{image_path}"
        # Logic from notebook
        if int(row["label"]) == 0:
            user_text = row["explanation_ori"] + """ 
        Now, based on the detection conclusion, and referring to the chain-of-thought below and the previous attribution examples, write the attribution description for this image.
        ## Authentic-image analysis chain-of-thought (for reference, no need to output)
        1. **Overall perception**: do not just say "clear"; describe the evenly distributed resolution with no local degradation.
        2. **Micro details**: look for "imperfect authenticity", such as: random noise, paper microfibers, natural stains, wear scratches (forged images are often too clean).
        3. **Light and shadow logic**: confirm that highlights and shadows match the ambient light (e.g. overcast diffuse light, indoor point light source).
        4. **Content logic**: confirm that the text content conforms to local laws and industry habits (e.g. tips left blank, credit card masking rules).
        ## Evidence Checklist (selectively reference the following evidence dimensions according to the image type)
        A. Global imaging consistency (select at least 2)
        - Noise/grain distribution is uniform across the whole image, and the transition from highlight to shadow is coherent; no abnormally smooth local regions, patching, repainting, or broken noise
        - Resolution and blur are consistent across the whole image; JPEG compression artifacts are evenly distributed, with no local over-sharpening or abnormal degradation
        - The foreground/background transition is natural, with no halos, aliasing, color fringing, cutout edges, or unreasonable feathering
        B. Physical optics and geometric consistency (select 2-3)
        - The light source direction is unified; the highlight and shadow projection directions on the lit side are consistent; contact shadows are soft and conform to the 3D attachment relationship
        - Perspective convergence and scale proportions are natural; occlusion relationships are reasonable; depth-of-field changes follow lens imaging laws
        - Material details are authentic: metal reflections/engraved chamfers/paper microfibers/wear scratches/stains and dust are random and consistent with long-term exposure or use
        C. Text/receipt-specific (use when text is visible, select 2-4)
        - Font/font weight/glyph structure are consistent; repeated digit glyphs are stable; baseline alignment is natural, with no misalignment/scaling distortion/mixed fonts
        - Column alignment follows industry typography (e.g. amount columns right-aligned, decimal places unified, currency symbols and formats standard)
        - Physical creases/curls cause natural continuous breaking, darkening, or perspective deformation of text (e.g. thermal-paper ink characteristics)
        - Amount/tax/total are verifiable and self-consistent; date, address, institution hierarchy, brand naming, etc. conform to common sense and industry habits within the visible information
        ## Reference high-frequency attribution phrases:
        - Noise/texture:
            "Noise is evenly distributed across the whole image, and the grain is coherent from highlight to shadow" "No abnormally smooth local regions or patching traces"
        - Edge/splicing:
            "The foreground-background transition is natural, with no splicing artifacts such as halos, aliasing, or color fringing"
        - Lighting/shadow:
            "The light source direction is unified, and the projection direction is consistent with the lit side of the object" "Contact shadows are soft and conform to the 3D attachment relationship"
        - Receipt/ticket text:
            "Dot-matrix/thermal fonts are consistent, amount columns are right-aligned, and repeated digit glyphs have consistent structure" "Text at creases shows natural breaking/darkening with paper deformation"
        - Summary: "Based on the analysis, this image authentically records the actual scene/transaction/physical object.
        ## Attribution description template (you may output in this format, replacing the content in {{...}} at the corresponding positions, written as a coherent paragraph; semicolons may be used to concatenate evidence points). The whole description should be logically coherent, with strong logical-causal analysis.
        <描述>
        This is a {{...}} {{brief description of the scene/subject}} photo, with no signs of digital forgery or post-processing tampering. {{From the visual-signal level, describe digital or object features}};
    {{You may describe physical and detail levels in depth, lighting and material}};
    {{You may describe, at the logical-content level, the scene or characters and common sense}};
    {{If it contains text/amount/address, write font consistency + alignment norms + verifiable/self-consistent values (optionally give one or two visible specific field examples}}; Based on the analysis, this image authentically records {{summary: this picture is an actual scene that conforms to what physical laws or what scene and what logic holds}}.
        </描述>
         """
        else:
            user_text = (
                row["explanation_ori"]
                + "The content corresponding to the specific coordinates (the region enclosed by the green line in the image) needs to be judged and corrected by you"
                + """ 
       Now, based on the detection conclusion and the provided coordinate regions, and referring to the chain-of-thought below and the previous attribution examples, write the attribution description for this image.
        ## Forged-image analysis chain-of-thought (for reference, no need to output)
        1. **Lock the differences**: compare the blue-box region with the surrounding region. Find the differences: color too pure? too bright? too blurry? edges too sharp?
        2. **Digital traces**: look for "editing artifacts", such as: smearing left by erasure, even dark spots left behind, heavily pixelated edges.
        3. **Physical inconsistency**: is there a lack of material texture? is the lighting direction consistent?
        4. **Logical collapse**: focus on checking mathematical errors (incorrect amount totals), semantic nonsense (invented words), format errors (wrong decimal places), and whether it conforms to the physical real world?
        5. **Infer motive (optional)**: why was it changed? (e.g. desensitization, inflated amounts, identity forgery).
        Based on the detection conclusion and the coordinate regions, generate the "attribution description". It must satisfy:
        1) The first paragraph is a one-sentence characterization: this is a {image type} that has undergone {{e.g. digital tampering / manual synthesis / AIGC generation}}.
        2) The second paragraph must state: there are {{N}} key tampered regions in the image, with their coordinates and corresponding contents listed one by one: {list each: coordinate + content/object}.
        3) The third paragraph (micro evidence): for the tampered regions, cover at least 3 categories of evidence (select from the following and write as causal sentences):
        - Digital traces: whether noise/texture/compression artifacts/JPEG blocking are "broken, discontinuous, abnormally clean, patchy"
        - Edge and fusion: overly sharp / pixelated / aliased / halo / unnatural feathering / pasted-on feel / overly uniform brightness
        - Physical optics: whether lighting direction, shadows, reflection/refraction, perspective, depth of field, and material texture violate the unified imaging law
        - Typography and printing: whether font family / font weight / letter spacing / baseline alignment are inconsistent or misaligned
        4) The fourth paragraph (macro logic): give 1-2 contradictions at the "functional or common-sense" level:
        - Receipt class: must perform verifiable checks (unit price x quantity, tax rate, total, change, decimal-place norms, date-format validity)
        - Signage/poster/book cover: must check whether the semantics are usable, spelling/grammar errors, whether the scene matches the content, and whether there is a real publication
        - AI-generated (AIGC): observe whether there is "unreadable garbled text, structure melting, organ fusion, facial symmetry, structural continuity, global over-smoothing, lack of real noise/material detail", etc.
        5) The fifth paragraph (infer means and motive, optional): use "more consistent with / suggests / indicates" to infer the possible tampering method (erase-repair-rewrite / 2D overlay / local repainting / AI generation), and use "possibly used for ..." to give the motive (desensitization / fictitious identity / inflated amount / forged record, etc.).
        6) The last paragraph must begin with "In summary" and end with "This (image/invoice/voucher/photo) is (a sampled means, such as digital tampering / manual editing / AIGC generation / digital replacement / splicing synthesis / fictitious transaction content and amount to create falsehood), and lacks authenticity and credibility".
    The whole description should be logically coherent, with strong logical-causal analysis. Output format (strictly follow):
    <描述>
    {Write a coherent paragraph in the above order; semicolons may be used to concatenate evidence points, but do not use bullet symbols. The style, wording, and descriptive means should be close to the example provided above, but do not copy the content!}
    </描述>
        """
            )
        # Append new message
        current_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_text},
                ],
            }
        )
        prompts.append(current_messages)

    # Run Inference
    print(f"Running inference on {len(prompts)} items...")
    outputs = llm.chat(prompts, sampling_params)

    # Parse results
    print("Parsing results...")
    new_results = []
    pattern = r"<描述>(.*?)</描述>"

    for i, (output, row) in enumerate(zip(outputs, list(df.iter_rows(named=True)))):

        generated_text = output.outputs[0].text
        # Extract content using regex
        match = re.search(pattern, generated_text, re.DOTALL)
        if match:
            extracted_text = match.group(1).replace("\n", "").replace("*", "").strip()
            final_text = extracted_text
        else:
            final_text = generated_text.strip()  # Fallback
        final_text = (
            final_text.replace("Logic and common-sense analysis:", "")
            .replace("Visual and digital feature analysis:", "")
            .replace("At the macro-logic level, ", "")
            .replace("At the macro-logic level: ", "")
            .replace("At the micro level, ", "")
            .replace("On inferring means and motive, ", "")
            .replace("At the micro-evidence level: ", "")
        )
        new_results.append(
            {
                "image_name": row["image_name"],
                "label": row["label"],
                "location": row["location"],
                "explanation": final_text,
            }
        )

    # Save to CSV
    print("Saving results...")

    df_out = pl.from_dicts(new_results)
    df_out.write_csv(argss.output_path)
    print("Done. Saved to ", argss.output_path)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file", type=str, default=""
    )  # read the image paths inside, as well as the pre-judged
    parser.add_argument(
        "--vector_db_root", type=str, default=""
    )  # vector database location
    parser.add_argument(
        "--image_path", type=str, default=""
    )  # root path for reading images
    parser.add_argument(
        "--output_path", type=str, default=""
    )  # path for saving the results
    parser.add_argument("--vlm_model", type=str, default="")  # VLM model path
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
