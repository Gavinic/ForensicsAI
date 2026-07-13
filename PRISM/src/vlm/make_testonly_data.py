import argparse
import json
import os
import random
import re
from pathlib import Path


def process_folder(root_folder):
    """Process a single root folder and return the data."""
    # Define paths
    image_folder = Path(root_folder) / "ForgeryAnalysis_Stage_2_Test/Image"

    if not image_folder.exists():
        print(f"Warning: Image folder missing under {root_folder}")
        return []

    # Get all image files (support common image formats)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f"*{ext}"))
        image_files.extend(image_folder.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"Warning: no image files found in {image_folder}")
        return []

    # Match images with their captions
    data_pairs = []
    for img_path in image_files:
        # Build the data item
        data_item = {
            "messages": [
                {
                    "role": "user",
                    "content": """# Role
你是一位专业的场景文本图像取证专家，专注于小票、路牌、街景门头、人像、宠物、汽车等常见场景的真实性鉴别，擅长识别数字篡改、文字替换、图像合成及逻辑矛盾，并能精准定位异常区域坐标。

# Task
请分析输入图片，**判断其是否含有伪造、篡改或AI生成内容**。结合视觉特征、内容逻辑与场景常识进行综合论证，**定位关键异常区域坐标**，并严格按照 JSON 格式输出结论。

# Analysis Dimensions
请从以下维度进行综合分析（无需输出思考过程，将证据融入 explanation 字段）：

## 1. 视觉取证特征（按场景聚焦）
- **小票/票据类**：热敏打印应有的像素化/点阵质感是否一致？文字边缘是否存在过度平滑或分辨率突变？金额数字是否有重影、擦除或粘贴痕迹？
- **路牌/门头类**：字体是否符合当地规范？字符间距、反光特性、安装角度是否自然？是否存在"浮贴"感、背景纹理断裂或透视矛盾？
- **人像/宠物类**：五官/面部边缘是否存在拼接痕迹、肤色/光照不一致？毛发纹理是否连续自然？肢体解剖结构（手指、关节、爪部）是否符合生物学特征？
- **汽车/车牌类**：车牌字符字体、间距、反光是否与标准一致？边框比例、固定螺丝、安装位置是否合理？车身标识/文字是否存在边缘锯齿、光影矛盾或物理遮挡异常？

## 2. 逻辑与常识验证（按场景聚焦）
- **小票逻辑**：数量×单价=小计？各项之和=总额？税率计算是否符合标示？时间/商户信息是否合理？
- **路牌/门头逻辑**：地名/店名是否符合当地命名规范？联系方式、营业时间格式是否正确？
- **人像/宠物逻辑**：场景元素（服饰、背景、道具）是否与人物/宠物身份匹配？是否存在时空矛盾？
- **汽车逻辑**：车牌归属地与场景地理位置是否一致？车型与标识、品牌信息是否匹配？

# Output Format
请**仅输出一个合法的 JSON 对象**，格式如下：
{"label": 0 或 1, "explanation": "连贯的分析报告段落"}

**字段说明**：
- `label`：**整数类型**（非字符串），**伪造/AI生成/篡改 → 1**，**真实未发现异常 → 0**
- `explanation`：字符串类型，必须为**一段连贯的分析报告段落**（不要分点/标题/换行），严格遵循以下4步结构内容：
  1. **开篇结论**：直接说明图像性质（真实 / 伪造 / AI 生成）。
  2. **关键定位**：若发现伪造，必须指出关键异常区域坐标，格式为 `[x1, y1, x2, y2]`。
  3. **证据阐述**：视觉层面列举1-2项最显著的视觉异常（如边缘拼接、光影矛盾、解剖畸形等）；逻辑层面列举1项计算错误、格式矛盾或常识冲突（若存在）。
  4. **综合说明**：1句话总结判断依据及篡改意图/真实性确认。

# Style Requirements
- **语气**：专业、客观、严谨，使用取证分析术语
- **证据导向**：所有判断必须基于图像中可观察的具体特征，避免主观猜测
- **简洁明确**：不使用"可能""似乎""疑似"等模糊词汇，基于证据给出确定结论
- **长度控制**：`explanation` 字段内容严格控制在 **150-250 字**

# Coordinate Specification
- **坐标系**：使用相对图像宽高的归一化坐标，范围 0~1000，左上角为 (0,0)，右下角为 (1000,1000)
- **边界框格式**：`[x1, y1, x2, y2]`，其中 `(x1,y1)` 为左上角坐标，`(x2,y2)` 为右下角坐标
- **数值要求**：所有坐标值必须为**整数**，且严格在 `[0, 1000]` 区间内
- **输出示例**：`位于坐标 [120, 85, 340, 210]`

# Constraint
- **仅输出 JSON**：禁止输出任何额外文本、Markdown 代码块标记（```json）、注释、解释或换行。
- **坐标必填**：若 `label=1`，`explanation` 中**必须**包含关键异常区域坐标，格式为 `[x1, y1, x2, y2]`，且严格遵循 # Coordinate Specification 规范。
- **真实场景**：若 `label=0`，`explanation` 中需明确说明"未发现数字伪造或后期篡改痕迹"，并列举 1-2 项支持真实性的正面证据。
- **JSON 规范**：确保输出为合法 JSON，键名使用双引号，`label` 为整数类型，`explanation` 为单行字符串（无换行、无转义问题）。

请开始分析提供的图像。\n""",
                },
            ],
            "images": [str(img_path.absolute())],
        }
        data_pairs.append(data_item)

    print(f"Found {len(data_pairs)} images in {root_folder}")
    return data_pairs


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Generate the Qwen-VL training dataset"
    )
    parser.add_argument(
        "--input_path",
        "-i",
        required=True,
        help="Input directory path (must contain an Image subdirectory)",
    )
    parser.add_argument(
        "--output_name",
        "-o",
        default="qwen3vl_testBonly.json",
        help="Output file name (default: qwen3vl_test.json)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set the random seed
    random.seed(args.seed)

    # Check whether the input directory exists
    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: input directory does not exist: {input_path}")
        return

    # Process the input directory
    print(f"Processing directory: {input_path}")
    all_data = process_folder(input_path)

    if not all_data:
        print("Error: no image files found")
        return

    # Save directly to the current directory
    output_file = args.output_name
    with open(output_file, "w", encoding="utf-8") as f:
        for d in all_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nDataset generation complete!")
    print(f"Output file: {output_file} (current directory)")
    print(f"Total images: {len(all_data)}")


if __name__ == "__main__":
    main()
