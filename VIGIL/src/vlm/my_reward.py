import logging
import os
import re

import torch
from bert_score import BERTScorer, score
from openai import OpenAI

SYSTEM_TEXT = """
   # Role
You are a senior digital image forensics expert and evaluation judge. Your task is to score (0-100) multiple model-generated [Candidate Explanations] strictly and automatically based on the [Reference Answer (Ground Truth)]. Each candidate explanation is distinguished by an Id. This score will be used as the Reward signal for reinforcement learning, so your scoring must be objective, detailed, and highly discriminative.
# Task Description
Evaluate whether the candidate explanation accurately, comprehensively, and logically explains the authenticity of the image. A high-quality explanation must contain a precise conclusion. For [forged images], it must include precise localization coordinates, low-level tampering-trace analysis, and high-level logical-flaw reasoning; for [authentic images], it must clearly state that no tampering traces are found and argue for authenticity from the perspectives of visual consistency and logical plausibility.
# Scoring Rubrics (Total 100 points)
Please score independently across the following five dimensions and sum them up:

## Dimension 1: Qualitative conclusion and overall positioning (0 - 10 points)
- **10 points**: Echoes from beginning to end; accurately judges authentic or forged at the opening (including the image type, e.g. "a forged digital image", "an authentically photographed receipt"), and ends with a clear "in summary" conclusion and intent inference.
- **5 points**: The conclusion is correct but lacks a complete beginning-and-end structure, or the description of the image type is inaccurate.
- **0 points**: The qualitative conclusion is wrong (judging authentic as forged or forged as authentic). **Note: if this item is 0, the total score is immediately 0 and the other dimensions need not be evaluated.**

## Dimension 2: Tampering localization and object description (0 - 20 points)
This dimension uses a dual-track scoring; first determine whether the reference answer is an authentic or a forged image:
- **16-20 points**:
  - [Forged image] Accurately lists the coordinates of all tampered/abnormal regions (in the format `[x1, y1, x2, y2]`), and the content description of the region is fully consistent with the reference answer.
  - [Authentic image] Clearly states that no forged or tampered region is found, and **does not output any tampering coordinates**.
- **8-15 points**:
  - [Forged image] Contains coordinates but with omissions (e.g. there are actually 4 regions but only 2 are listed), or the coordinate format is non-standard, or the object description has minor deviations.
  - [Authentic image] Clearly states it is authentic, but the text contains ambiguous "suspicious region" descriptions (without coordinates).
- **1-7 points**:
  - [Forged image] Only text description without coordinates, or the coordinates show severe hallucination (completely inconsistent with the reference answer).
  - [Authentic image] This score range does not apply.
- **0 points**:
  - [Forged image] Does not point out any tampering location.
  - [Authentic image] **Severe hallucination: forcibly outputs tampering regions or coordinates for an authentic image.**

## Dimension 3: Visual and low-level feature analysis (0 - 30 points)
*(Evaluates the ability to analyze pixel-level, optical-level, and rendering-level traces)*
- **25-30 points**: Describes visual features deeply and accurately.
  - [Forged image] Accurately points out edge anomalies, depth-of-field contradictions, splicing traces, font distortion, etc.
  - [Authentic image] Accurately argues for visual consistency, such as consistent lighting and shadow systems, reflections that follow physical laws, uniform digital noise distribution, natural edge transitions, etc.
- **15-24 points**: Points out visual features but the description is relatively superficial, lacking specific forensic analysis vocabulary (e.g. missing professional descriptions such as "baseline misalignment", "noise distribution").
- **1-14 points**: The visual analysis contains factual errors, or mechanically applies many irrelevant visual features.
- **0 points**: No visual feature analysis at all.

## Dimension 4: Logic and common-sense reasoning ability (0 - 30 points)
*(Evaluates the ability to reason about physical laws, mathematical calculations, and common-sense semantics)*
- **25-30 points**: Reveals deep logic extremely accurately.
  - [Forged image] Accurately reasons about structural-mechanics violations, mathematical calculation errors, tax/currency common-sense contradictions, or missing physical interactions.
  - [Authentic image] Accurately reasons about the plausibility of the scene, such as tight institutional-attribution and geographic logic, conformance to real operating models of specific regions/industries (e.g. "one-stop" services), and mutually corroborating text information without contradictions.
- **15-24 points**: Has a reasoning process but is not comprehensive enough, or does not hit the most central logical point in the reference answer.
- **1-14 points**: The logical analysis has obvious calculation errors or "machine hallucinations", and the reasoning process is self-contradictory.
- **0 points**: Stays only at describing the image, with no common-sense or logical reasoning.

## Dimension 5: Forensic language style and fluency (0 - 10 points)
- **8-10 points**: The language is highly professional, adopting a forensic analysis register (e.g. "consistent with ... features", "presents ... contradictions"), with appropriate logical connectors ("furthermore", "meanwhile", "in summary"), smooth writing, and no grammatical errors.
- **4-7 points**: Clear expression, but too colloquial, or with slight jumps in the writing logic.
- **0-3 points**: Sentences are incoherent, generating garbled text, repetition, or completely irrelevant nonsense.

# Deduction Rules (strict deductions - anti-cheating and hallucination)
On top of the above base score, deduct points if any of the following occurs:
1. **Coordinate forgery (-10 points)**: Fabricating coordinates completely unrelated to the reference answer.
2. **False alarm (-10 points)**: **For an authentic image, forcibly fabricating tampering traces or outputting coordinates (False Alarm).**
3. **Arithmetic hallucination (-10 points)**: In receipt/ticket analysis, the multiplication/addition calculation process listed by the model contains basic mathematical errors (e.g. 3*2.90=9.00).
4. **Mechanical application (-10 points)**: Forcibly using boilerplate that does not fit the current scenario (e.g. analyzing text layout for a landscape photo).

# Output Format (strictly follow JSON format)
Please output only one JSON object, without any markdown markers (such as ```json) or other explanatory text. It must contain each dimension score, the deduction, and the final total score.
[{ "Id": <int>,
  "Dimension_1_Conclusion": <int>,
  "Dimension_2_Grounding": <int>,
  "Dimension_3_Visual": <int>,
  "Dimension_4_Logic": <int>,
  "Dimension_5_Style": <int>,
  "Penalty_Score": <int>,
  "Total_Score": <int>,
  "Rationale": "<a brief scoring rationale within 100 characters, explaining the deductions>"
}]
===
[Input data]
Reference answer (Ground Truth): {GT_TEXT}

"""
CANDIATE = "Id: {id}, Candidate Explanation: {PRED_TEXT} \n"
client = OpenAI(
    # If the environment variable is not configured, replace with your Alibaba Cloud Bailian API Key: api_key="sk-xxx"
    api_key="<YOUR_API_KEY>",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Pre-compile regex for efficiency
TAG_PATTERN = r"<description>(.*?)</description>"
COORD_PATTERN = r"\[\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*[,，]\s*\d+\s*\]"


# You can adjust the batch_size parameter according to your GPU memory (default is usually 64)
bert_scorer = BERTScorer(lang="zh", device="cuda")


def get_coordinates(text):
    """
    Extract coordinates from the first two sentences of the text according to your logic
    """
    try:
        # Try to match the first two sentences
        match = re.match(r"^([^。]+。)([^。]+。)", text)
        if match:
            s1 = match.group(1).strip()
            s2 = match.group(2).strip()
            coords = re.findall(COORD_PATTERN, s1)
            if not coords:
                coords = re.findall(COORD_PATTERN, s2)
            return coords
    except Exception:
        pass
    return []


def grpo_reward_fn(completions, answers, **kwargs):
    """
    GRPO reward function
    completions: list of complete model outputs (with tags) [ [{'rol':xxx,'content':xxx}],..]
    answer: list of Ground Truth answers (usually contains the reference text and the number of reference coordinates)
    """
    rewards = []
    # print(completions)
    # 1. Extract the actual content generated by the model
    extracted_texts = []
    for gen in completions:
        tag_match = re.search(TAG_PATTERN, gen[0]["content"], re.DOTALL)
        if tag_match:
            extracted_texts.append(tag_match.group(1).strip())
        else:
            extracted_texts.append(
                ""
            )  # format error; the subsequent content score will be low

    # 2. Batch-compute BERTScore (for performance, compute here in a unified way)
    # Filter out entries with no extracted content to reduce computation, but keep the list length consistent
    valid_indices = [i for i, t in enumerate(extracted_texts) if len(t) > 0]
    f1_scores = [0.0] * len(completions)

    if valid_indices:
        cands = [extracted_texts[i] for i in valid_indices]
        refs = [answers[i] for i in valid_indices]
        # P, R, F1 have shape (len(cands),)
        P, R, F1 = bert_scorer.score(cands, refs)
        for idx, score_val in zip(valid_indices, F1.tolist()):
            f1_scores[idx] = score_val
        #  =============== qwen-max scoring ================
        sys_text = SYSTEM_TEXT.replace("{GT_TEXT}", answers[0])
        # .replace("{GT_TEXT}", gt_data)
        for i, pred_data in enumerate(cands):
            sys_text += CANDIATE.replace("{id}", str(i)).replace(
                "{PRED_TEXT}", pred_data
            )
        messages = [{"role": "system", "content": sys_text}]
        completion = client.chat.completions.create(
            model="qwen3-max",  # you can replace it with other deep-thinking models as needed
            messages=messages,
            top_p=0.1,
            seed=2026,
            # temperature=
            # extra_body={"enable_thinking": False},
            # stream=True
        )
        response = eval(completion.choices[0].message.content)
        print(response)
        qwen_score = [data["Total_Score"] for data in response]
    # sys_text
    # 3. Compute the final combined Reward
    gt_data = answers[0]  #
    print(gt_data)
    for i in range(len(completions)):
        reward = 0.0
        gen_text = extracted_texts[i]
        gt_data = answers[i]
        print(gen_text, response[i]["Rationale"])
        # print(f"{i} is each gt the same?",gt_data,"pred: ",gen_text)
        # --- Rule A: tag-format reward ---
        if len(gen_text) > 0:
            reward += 0.1  # base score as long as a tag is written
        else:
            rewards.append(0)  # no tag written at all, directly 0 points
            continue

        # --- Rule B: similarity score (BERTScore) ---
        # F1 is usually between 0.6-0.9; the weight is set to 1.0
        tmp_f1 = f1_scores[i]
        reward += tmp_f1
        tmp_qwen = qwen_score[i] * 0.01
        reward += tmp_qwen
        print(tmp_f1, tmp_qwen)
        ## Judge whether the two texts' judgments are consistent
        # if ('real' in gen_text[:100]) == ('real' in gt_data[:100]):
        #     reward += 0.1
        # else:
        #     reward -= 0.1
        # if ('real' in gen_text[:100]) and ('fake' in gen_text[:100]):
        #     reward -=0.2

        # --- Rule C: coordinate-count consistency ---
        # pred_coords = get_coordinates(gen_text)
        # gt_coord_count = get_coordinates(gt_data)
        # if len(pred_coords) == len(gt_coord_count):
        #     reward += 0.05  # reward when the counts fully match
        # elif len(gt_coord_count) and len(pred_coords) > 0:
        #     reward += 0.01  # at least coordinates were written, but the count is wrong; give a small effort score
        # elif len(gt_coord_count)==0 and len(pred_coords) > 0:
        #     reward -= 0.06

        # --- Rule D: penalty (optional) ---
        if len(gen_text) > 900:  # prevent the model from repeating or being too long
            reward -= 0.1

        rewards.append(reward)
    # print(list(zip(extracted_texts,rewards,kwargs['names'])))
    return rewards
