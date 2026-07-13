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
            You are a world-class digital image forensics expert. You possess deep knowledge of physical optics, computer vision, typography and printing, and semantic logic. Analyze images for forgery detection (forged? authentic?) and localization, and produce an explainable attribution description. The input images cover high-finance, social, education, receipt, notice, and natural-scene image types.
            The input images are categorized as high-finance, social, education, receipt, notice, and natural-scene images; focus on character forgery, tampering and generation within the image, or distortions of digitally generated human hands, ears, and eyes.
            Your writing must be "evidence-oriented, causally closed, and stylistically consistent", and strictly follow the output protocol:
            - Output only <描述> ... </描述>; do not output chain-of-thought, step numbers, or list markers.
            - Tone: objective, clinical, evidence-oriented. Frequently use: consistent / matches / violates / abnormal / abrupt / broken / discontinuous / suggests / indicates.
            - The structural order is fixed: first a qualitative conclusion -> then lock the coordinate region -> then micro evidence -> then macro logic -> then infer means and motive -> close with "in summary".
            - Do not fabricate information not visible in the image; if text is unreadable, describe it as "illegible / unreadable garbled text"; when people / institutions / locations are uncertain, use "suspected / possibly".
            # Goal
            Refer to the [detection conclusion] provided by the user (authentic or forged, and the coordinates of forged regions), and combine it with the visual content of the image to identify forgery, generating a detailed, professional, and logically rigorous attribution description.
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
            - **Receipt-specific high-frequency**: calculation errors and common-sense contradictions, decimal places, tax rate/GST, invalid date format (e.g. three decimal places, contradiction between tax amount and tax rate)
            Below are 3 attribution-description examples; carefully analyze the style and attribution-reasoning logic of the examples for subsequent generation:
            """,
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "file://../data/demo/0d620e6cf2b0417fbf4edd0dee0ec549.jpg"
                },
            },
            {
                "type": "text",
                "text": """example1: 检测结论：这是一份伪造的{...}照片。存在多处伪造区域,其坐标分别为：[348, 423, 372, 462],[318, 421, 334, 454],[178, 282, 192, 304],[129, 282, 143, 302],[400, 275, 421, 302],[326, 274, 377, 363]
         归因描述:<描述>这是一份人工智能生成的数字伪造户外场景图像，呈现一群人站在写有“Breaking Bald”字样的标牌前。图像中存在多处典型AIGC生成缺陷：坐标[200, 216, 282, 254]区域本应为可读文字的位置呈现无法辨识的扭曲符号，不构成有效字符，属于文本生成失败；覆盖全部八个人物面部的区域[92, 269, 426, 309]普遍存在五官模糊、形态扭曲及部分人脸融合现象，系统性失真远超摄影误差范畴；标牌顶部及右侧附着形态怪异的粉色球体和蓝色异形物体，与标牌主题无关且结构不符合物理逻辑，属于AIGC幻觉产物。视觉层面表现为字体风格不统一，“Breaking”与“Bald”在质感与渲染方式上存在差异，左上角“EABS B”字体生硬且间距不均，违背专业设计规范；整体图像缺乏真实照片应有的传感器噪声和精细纹理，呈现过度平滑的数字感，尤其在阴影与背景区域更为明显。内容逻辑上，真实标牌不会包含无法解读的乱码文字或无功能性的怪异装饰物，此类设计在现实世界的信息传达体系中不可能被采用，尽管“Breaking Bald”本身是对影视作品的戏仿，具有虚构性质，但结合上述视觉伪造痕迹与逻辑谬误，足以确认该图像系AI生成的非真实内容。综上所述，该图像系人工智能生成的伪造内容，其创作目的可能在于娱乐或艺术表达，而非记录真实事件。</描述>
            """,
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "file://../data/demo/c33c0405c4984e4f8845d3ff4e72f9fa.jpg"
                },
            },
            {
                "type": "text",
                "text": "example2:检测结论：检测结论：这是一张真实拍摄{{描述图片内容...的图片}},未发现数字伪造或后期篡改的痕迹。归因描述:<描述>这是一张真实拍摄的夜间城市街道出租车队列照片，未发现数字伪造或后期篡改的痕迹。前景绿色涂装的日产Cedric出租车车身上的“東京無線”标识、司机姓名“坂本”以及后窗“眠眠打破”广告贴纸在字体、颜色、反光特性和透视关系上均表现自然，与玻璃介质和车身曲面的光学特性一致，文字边缘未出现对齐偏移或合成伪影；车牌“足立500 う 19-59”采用日本标准车牌字体，字形与间距符合官方规范；中景橙色“中央無線”出租车与背景模糊的城市建筑、霓虹灯光斑共同构成合理的景深层次，背景散景光斑形态多样且亮度分布随机，无重复或镜像拼接迹象；全图噪点在暗部与亮部之间平滑过渡，信噪比变化符合夜间长曝光摄影规律，车身多光源反射（如街灯、霓虹、车灯）在位置、色温和强度上逻辑自洽，尾灯红光在保险杠及地面形成的漫反射柔和自然，车队沿道路纵深方向的尺寸递减与车道线收敛点也符合正确透视关系。“足立500 う 19-59”车牌格式符合东京都足立区普通车辆号牌规范，“東京無線”与“中央無線”均为东京真实运营的出租车公司，分别使用绿色与橙色涂装，前景车型与日产Cedric Y31高度吻合，该车型长期作为日本出租车主力，“眠眠打破”为日本市售提神饮料，其在出租车窗投放广告符合针对司机群体的常识，而夜间多公司出租车在路边排队候客也是东京常见的交通场景。综合分析，该图像真实记录了东京城市环境中的一次夜间拍摄场景。</描述>",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "file://../data/demo/e3352d98645346f19f73d14a0bb579e7.jpg"
                },
            },
            {
                "type": "text",
                "text": "example3:检测结论：这是一份伪造的{...}照片。存在3处伪造区域,关键篡改区域包括:[920, 450, 1024, 513],[400, 122, 511, 176],[668, 22, 756, 78]。归因描述:<描述>这是一份伪造的夜间足球比赛场景体育场广告牌图像，其中包含三处关键篡改区域，其坐标及内容分别为：[401, 115, 512, 176] 处的 “spectacular”、[670, 18, 748, 71] 处的 “DROYLSDEN”，以及 [920, 450, 1019, 509] 处的 “Soil”。这些区域在视觉层面存在多处物理异常：“spectacular” 的字体倾斜度与字重与同一广告牌上“East Manchester Builders”及电话号码不匹配，风格突兀；“Soil” 使用异常锐利干净的现代无衬线字体，而广告牌下方的网址文本呈现自然像素化与轻微磨损，二者风格差异显著；“DROYLSDEN” 的字母基线排列过于水平，未体现广告牌因悬挂产生的自然弧度与透视变形，与下方街道地址文本的贴合度不一致；“spectacular” 周围白色背景有轻微涂抹感，边缘模糊，疑似为融合粘贴文本而进行的后期平滑处理；“Soil” 所在区域的蓝色背景噪声模式异常平滑，缺乏原始广告材质应有的纹理与噪点，符合“擦除-填充-粘贴”操作后的典型痕迹；“spectacular” 文本分辨率略低于相邻原始文本，边缘清晰度不足，暗示其来源于不同压缩或缩放历史的外部图像源；“DROYLSDEN” 文本亮度分布过于均匀，未能复现广告牌在夜间多光源照射下应有的从左至右的轻微亮度衰减；“Soil” 后方留有大量空白直至边框，不符合商业广告紧凑排版的常识。在逻辑层面，广告文本语义结构严重断裂：“spectacular Sponsors To Local Football” 在英语语法与商业宣传语境中不合逻辑，标准表述应为 “Proud Sponsors of Local Football” 等规范结构；“Soil” 作为独立品牌名称虽非不可能，但结合其异常排版与字体风格，更符合随意替换的伪造特征。综上所述，该图像系人为编辑伪造，不具备原始真实性与可信度。</描述>",
            },
        ],
    },
]


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
        model=argss.vlm_model,  # "Qwen/Qwen3.5-9B",#,"Qwen/Qwen3-VL-8B-Instruct"
        trust_remote_code=True,
        enable_prefix_caching=True,
        # gpu_memory_utilization=0.9,
        tensor_parallel_size=1,  # Adjust based on available GPUs
        limit_mm_per_prompt={"image": 5},  # Allow multiple images in history
        max_model_len=36768,  # Limit context length to fit in GPU memory
        allowed_local_media_path="/",  # Allow loading images from local filesystem
    )

    sampling_params = SamplingParams(
        temperature=0.7,  # 0.7
        max_tokens=4399,  # increased token limit just in case
        stop_token_ids=None,
    )

    prompts = []

    # Iterate and prepare prompts
    print("Preparing prompts...")
    for row in list(df.iter_rows(named=True)):
        # Create a deep copy of base messages for each request
        # We need to reconstruct the list to avoid modifying the original references if we were to mutate dictionaries,
        # but here we just append to the list.
        current_messages = [msg.copy() for msg in MESSAGES_BASE]

        # Format image path
        image_path = f"{argss.image_path}/{row['image_name']}"
        image_url = f"file://{image_path}"
        #  ## Authentic-image analysis chain-of-thought
        # 1. **Overall perception**: do not just say "clear"; describe the evenly distributed resolution with no local degradation.
        # 2. **Micro details**: look for "imperfect authenticity", such as: random noise, paper microfibers, natural stains, wear scratches (forged images are often too clean).
        # 3. **Light and shadow logic**: confirm that highlights and shadows match the ambient light (e.g. overcast diffuse light, indoor point light source).
        # 4. **Content logic**: confirm that the text content conforms to local laws and industry habits (e.g. tips left blank, credit card masking rules).
        # Logic from notebook
        if row["label"] == 0:
            user_text = row["explanation"] + """ 
        # Instruction
        Based on the detection conclusion, refer to the [authentic-image analysis chain-of-thought] below.
        ## Evidence Checklist (selectively reference the following evidence dimensions according to the image type)
        A. Global imaging consistency (select at least 2)
        - Noise/grain distribution is uniform across the whole image, and the transition from highlight to shadow is coherent; no abnormally smooth local regions, patching, repainting, or broken noise
        - Resolution and blur are consistent across the whole image; no local over-sharpening or abnormal degradation
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
        The whole description should be logically coherent, with strong logical-causal analysis.
        - Summary:
            "Based on the analysis, this image authentically records the actual scene/transaction/physical object.
        ## Attribution description template (you may output in this format, replacing the content in {{...}} at the corresponding positions, written as a coherent paragraph; semicolons may be used to concatenate evidence points)
        The whole description should be logically coherent, with strong logical-causal analysis.
        <描述>
        This is a {{...}} {{brief description of the scene/subject}} photo, with no signs of digital forgery or post-processing tampering. {{From the visual-signal level, describe digital or object features in detail: e.g. noise is evenly distributed across the whole image, with no local smoothing or noise anomalies caused by splicing; edge transitions are natural and blend clearly with the background, with no synthetic cutout feel or color fringing}};
    {{You may describe physical and detail levels in depth, lighting and material: e.g. lighting matches the xx environment, shadow/reflection logic is self-consistent; material textures (e.g. paper fibers, metallic luster) are authentically visible; natural stains/scratches on the surface prove the existence of a physical object; no halos, aliasing, or over-smoothing common in digital synthesis}};
    {{You may describe, at the logical-content level, text and common sense in detail: e.g. fonts/typography conform to industry standards (example: font consistency, baseline alignment); content logic is self-consistent (example: address matches, mathematical calculations are correct, conforms to regulations/habits of the xx region)}};
    {{If it contains text/amount/address, write font consistency + alignment norms + verifiable/self-consistent values (give one or two visible specific field examples) [optional]}};
     Based on the analysis, this image authentically records {{summary: this picture is an actual scene that conforms to what physical laws or what scene and what logic holds}}.
        </描述>
         """
        else:
            user_text = (
                row["explanation"]
                + "The content corresponding to the specific coordinates (the region enclosed by the green line in the image) needs to be judged by you"
                + """ 
       Based on the detection conclusion and the provided coordinate regions, you may consider the [forged-image analysis chain-of-thought] below.
        ## Forged-image analysis chain-of-thought (for reference, no need to output)
        1. **Lock the differences**: compare the blue-box region with the surrounding region. Find the differences: color too pure? too bright? too blurry? edges too sharp?
        2. **Digital traces**: look for "editing artifacts", such as: smearing left by erasure, even dark spots left behind, heavily pixelated edges.
        3. **Physical inconsistency**: is there a lack of material texture (too smooth)? is the lighting direction wrong?
        4. **Logical collapse**: focus on checking mathematical errors (incorrect amount totals), semantic nonsense (invented words), format errors (wrong decimal places), and whether it conforms to the physical real world?
        5. **Infer motive (optional)**: why was it changed? (e.g. desensitization, inflated amounts, identity forgery).
        Based on the detection conclusion and the red-box coordinate regions, generate the "attribution description". It must satisfy:
        1) The first paragraph is a one-sentence characterization: this is a {image type} that has undergone {digital tampering / manual synthesis / AIGC generation}.
        2) The second paragraph must state: there are {N} key tampered regions in the image, with their coordinates and corresponding contents listed one by one: {list each: coordinate + content/object}.
        3) The third paragraph (micro evidence): for the tampered regions, cover at least 3 categories of evidence (select from the following and write as causal sentences):
        - Digital traces: whether noise/texture/compression artifacts/JPEG blocking are "broken, discontinuous, abnormally clean, patchy"
        - Edge and fusion: overly sharp / pixelated / aliased / halo / unnatural feathering / pasted-on feel / overly uniform brightness
        - Physical optics: whether lighting direction, shadows, reflection/refraction, perspective, depth of field, and material texture violate the unified imaging law
        - Typography and printing: whether font family / font weight / letter spacing / baseline alignment are inconsistent or misaligned
        4) The fourth paragraph (macro logic): give 1-2 contradictions at the "functional or common-sense" level:
        - Receipt class: must perform verifiable checks (unit price x quantity, tax rate, total, change, decimal-place norms, date-format validity)
        - Signage/poster/book cover: must check whether the semantics are usable, spelling/grammar errors, whether the scene matches the content, and whether there is a real publication
        - AI-generated (AIGC): emphasize "unreadable garbled text, structure melting/organ fusion, facial symmetry, structural continuity, global over-smoothing, lack of real noise/material detail", etc.
        5) The fifth paragraph (infer means and motive, optional): use "more consistent with / suggests / indicates" to infer the possible tampering method (erase-repair-rewrite / 2D overlay / local repainting / AI generation), and use "possibly used for ..." to give the motive (desensitization / fictitious identity / inflated amount / forged record, etc.).
        6) The last paragraph must begin with "In summary" and end with "This (image/invoice/voucher/photo) is (a sampled means, such as digital tampering / manual editing / AIGC generation / digital replacement / splicing synthesis / fictitious transaction content and amount to create falsehood), and lacks authenticity and credibility".
    The whole description should be logically coherent, with strong logical-causal analysis. Output format (strictly follow):
    <描述>
    {Write a coherent paragraph in the above order; semicolons may be used to concatenate evidence points, but do not use bullet symbols. The style, wording, and descriptive means should be close to the example provided above.}
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
        row["explanation"] = final_text
        print(final_text)
        new_results.append(row)

    # Save to CSV
    print("Saving results...")

    df_out = pl.from_dicts(new_results)
    df_out.write_csv(argss.output_path)
    print("Done. Saved to ", argss.output_path)


## transformer 4.57.3
## vllm
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file", type=str, default=""
    )  # read the image paths inside, as well as the pre-judged
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
