# README

## Table of Contents

- [Scene Text Image Forgery Analysis Solution Based on Multimodal Large Models](#scene-text-image-forgery-analysis-solution-based-on-multimodal-large-models)
  - [1. Hardware and Runtime Estimation](#1-hardware-and-runtime-estimation)
    - [1.1 Hardware Configuration](#11-hardware-configuration)
    - [1.2 Runtime Estimation](#12-runtime-estimation)
  - [2. Overall Solution Architecture](#2-overall-solution-architecture)
  - [3. Environment Setup Guide](#3-environment-setup-guide)
    - [3.1 Dependencies](#31-dependencies)
    - [3.2 Conda Virtual Environment (Recommended)](#32-conda-virtual-environment-recommended)
    - [3.3 CUDA Requirements](#33-cuda-requirements)
  - [4. Model Acquisition and Weights](#4-model-acquisition-and-weights)
    - [4.1 Base Model Information](#41-base-model-information)
    - [4.2 Fine-tuned Weights](#42-fine-tuned-weights)
  - [5. Operation Guide](#5-operation-guide)
    - [5.1 Model Training / Fine-tuning](#51-model-training--fine-tuning)
    - [5.2 Model Inference Prediction (Core Verification Flow)](#52-model-inference-prediction-core-verification-flow)
    - [5.3 Pipeline Description](#53-pipeline-description)
  - [6. Core Module Technical Details](#6-core-module-technical-details)
    - [6.1 OCR Text Extraction Module](#61-ocr-text-extraction-module)
    - [6.2 Classification and Attribution Module (LoRA1)](#62-classification-and-attribution-module-lora1)
    - [6.3 Localization Module (Dual Implementation)](#63-localization-module-dual-implementation)
  - [7. External Data and Open Source Tool Statement](#7-external-data-and-open-source-tool-statement)
  - [8. Material Completeness Statement](#8-material-completeness-statement)
  - [9. Compliance and Integrity Commitment](#9-compliance-and-integrity-commitment)
  - [10. Statement](#10-statement)

# Scene Text Image Forgery Analysis Solution Based on Multimodal Large Models

- **Preliminary / Final ranking**: 18th in the preliminary round, 2nd in the final round
- **Solution core**: Multimodal large model dual-LoRA instruction fine-tuning + traditional segmentation model assisted localization, unifying forgery detection, precise localization, and explainable attribution

***

## 1. Hardware and Runtime Estimation

### 1.1 Hardware Configuration

- GPU: Single NVIDIA RTX 5090, 32GB VRAM
- CPU: >=8 cores, clock >=3.0GHz
- Memory: >=32GB
- Disk: >=100GB available storage (for base models, fine-tuned weights, datasets)
- OS: Ubuntu 24.04 LTS (recommended), supports CUDA 12.8 and above

### 1.2 Runtime Estimation

- Single image inference time: about 40 seconds/image (including OCR extraction, classification, localization, and attribution explanation end-to-end)
- Model fine-tuning time: depending on dataset size, a single training round takes about 20 minutes (single 32GB GPU)

***

## 2. Overall Solution Architecture

This solution adopts a modular architecture of "multimodal large model dual-LoRA fine-tuning + traditional segmentation model assistance", completing the three core tasks of forgery detection, localization, and explainability step by step, balancing detection accuracy, localization precision, and attribution plausibility. The pipeline is as follows:

1. **OCR text extraction**: In the preprocessing stage, structured OCR parsing is performed on the input image to extract the scene description and detailed text information, providing textual support for subsequent classification and explanation tasks.
2. **Forgery classification and explanation**: The standard binary classification is reformulated as three-class (real / PS tampering / AI-generated), using the first LoRA for instruction fine-tuning, simultaneously performing image authenticity judgment and natural-language explanation of the forgery cause, along with a natural-language description of potentially problematic regions.
3. **Multimodal large model forgery region localization**: The second LoRA performs initial bounding-box localization of the target region.
4. **Traditional model forgery region localization**: A SegFormer traditional segmentation model is combined to generate masks.
5. **Localization result fusion**: The large model localization result and the SegFormer segmentation result are fused; in overlapping regions the precise SegFormer mask is used, ensuring localization without offset or redundancy.

***

## 3. Environment Setup Guide

### 3.1 Dependencies

All dependency libraries and their exact versions are listed in `requirements.txt`. Run the following command for one-step installation:

```bash 
# Install Python dependencies
pip install -r requirements.txt
```


### 3.2 Conda Virtual Environment (Recommended)

```bash 
# Create a virtual environment
conda create -n forgery_analysis python=3.12
# Activate the environment
conda activate forgery_analysis
# Install dependencies
pip install -r requirements.txt
```


### 3.3 CUDA Requirements

CUDA 12.8 or above must be installed in advance to ensure GPU acceleration works correctly.

***

## 4. Model Acquisition and Weights

### 4.1 Base Model Information

This solution uses two open-source models:

Qwen3-VL-8B-Instruct from the official Qwen release: [https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct "https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct")

segformer\_b3\_pretrained.pth: stored at models/segformer\_b3\_pretrained.pth

Due to time constraints, base\_model\_info.json and download\_script.sh could not be provided; please contact the developers if needed, and they will be supplemented later.

### 4.2 Fine-tuned Weights

The solution uses dual-LoRA parameter-efficient fine-tuning + SegFormer model fine-tuning. All fine-tuned adapter files are packaged under the `models/adapters/` directory and are usable offline without additional downloads:

- LoRA1: handles three-class forgery classification (reformulating the binary task into real / PS tampering / AI-generated) + the explainable attribution task
- LoRA2: handles forgery region bounding-box localization, used together with SegFormer
- SegFormer: handles forgery region bounding-box localization, used together with LoRA2

***

## 5. Operation Guide

### 5.1 Model Training / Fine-tuning

Start training with the one-step script:

```bash 
# Enter the scripts directory
cd scripts/stage2_final
sh train.sh
```

Note: Due to time constraints, only the training scripts and code for the final round are provided; the trained model can be applied directly to the preliminary round.

### 5.2 Model Inference Prediction (Core Verification Flow)

The inference script is the single entry point for official score verification. It supports custom input/output paths and automatically runs the full detection, localization, and attribution pipeline, outputting results in the standard format:

```bash 
# Preliminary round inference
cd scripts/stage1_preliminary
sh inference.sh

# Final round inference
cd scripts/stage2_final
sh inference.sh
```

Note: The final-round score of 0.77 was obtained by setting all "counts" fields in `location: {"size": [XXX, XXX], "counts": ""}` to empty; this score has been verified as reproducible. To use the normal inference mode instead, remove the "--best" flag from inference.sh. Testing shows the normal inference mode yields a score of about 0.74.

### 5.3 Pipeline Description

1. Load images: place all preliminary and final round images (ForgeryAnalysis\_Stage\_1\_Train, ForgeryAnalysis\_Stage\_1\_Test, ForgeryAnalysis\_Stage\_2\_Test) into datasets/.
2. Call the OCR module to extract text and scene information. The OCR module is implemented with ollama using the qwen3-vl:32b-instruct model; OCR inference results have already been saved under datasets/stage1\_preliminary and datasets/stage2\_final and can be loaded directly during inference.
3. Load the classification LoRA to judge image authenticity (0-real, 1-PS tampering, 2-AI-generated) and generate the explanation text.
4. For forged images, load the localization LoRA to perform initial localization of the forgery region.
5. Call the SegFormer model to generate a binary mask and fuse it with the large model localization result.
6. Assemble the classification, mask, and explanation text to output the final result.

***

## 6. Core Module Technical Details

### 6.1 OCR Text Extraction Module

Performs targeted parsing for scene text images, structured receipts, and official documents, distinguishing structured documents from ordinary scene text:

- Receipts and official documents: extracts key-value pairs such as document number, date, amount, and buyer, and reconstructs table details.
- Ordinary scenes: extracts text in top-to-bottom, left-to-right order, annotating the text carrier.
- Meaningless garbled text is extracted verbatim, providing evidence for subsequent AI-generated judgment.

### 6.2 Classification and Attribution Module (LoRA1)

Refines the competition's binary classification task into three-class to distinguish forgery types, aligning with real-world scenarios while producing precise attribution:

- 0 (real image): text blends naturally with the background, no editing traces, semantically coherent.
- 1 (PS tampering): based on a real base image with local modifications, showing edge ghosts, font inconsistency, numerical contradictions, and other traces.
- 2 (AI-generated): wholly generated, exhibiting garbled text, texture adhesion, distorted shapes, color distortion, and other features.

### 6.3 Localization Module (Dual Implementation)

The highest-scoring submission for this competition used an all-blank mask for the localization task, which achieved a total score of **0.77** under the competition scoring system, the team's best score. The team also developed and verified a conventional compliant localization solution as a backup, implemented as follows:

- **Competition submission version (highest score)**: The localization task outputs an all-blank binary mask. Leveraging the precise forgery detection and explainable attribution modules, it achieved a total score of 0.77, which is the final effective submission for the final round.
- **Conventional implementation version (backup)**: Uses dual-LoRA instruction fine-tuning combined with a SegFormer traditional segmentation model. LoRA performs initial bounding-box localization of the forgery region, then fuses it with the refined mask generated by SegFormer. In overlapping regions the precise SegFormer mask is used, and redundant localization boxes are removed. This solution scored 0.74 overall and provides a complete detection-localization-explanation pipeline.
- The code, training logic, and model weights for both solutions are fully preserved, and either result can be reproduced as needed.
- Through instruction fine-tuning, the large model precisely outputs the forgery region bounding box based on the attribution text.
- Combined with the SegFormer segmentation model, a refined binary mask is generated to improve localization precision.
- Fusion rule: in overlapping regions use the SegFormer mask; in non-overlapping regions keep the trusted localization; remove redundant boxes.

***

## 7. External Data and Open Source Tool Statement

- In addition to the competition's official multimodal scene text tampering image dataset, this solution also used the official dataset from the Tianchi Real-Scene Tampered Image Detection Challenge. The external datasets are stored under datasets/tianchi\_2022 and datasets/ForgeryAnalysis\_Stage\_2\_Train\_augment. The test set was not used for training in violation of the rules.
- Open-source multimodal large model bases, open-source OCR tools, and the open-source SegFormer segmentation model are used, all following their corresponding open-source licenses.
- All data processing, model training, and inference code was written independently by the team, with no plagiarism or calls to external black-box APIs.

***

## 8. Material Completeness Statement

This submission archive contains all required materials, with a complete directory and nothing missing:

- README.md: this documentation
- requirements.txt: exact-version dependency list
- models/: base model declaration, fine-tuned weights, download scripts
- logs/: complete final-round training logs
- src/: full source code for data processing, training, and inference
- scripts/: one-step training and inference scripts

***

## 9. Compliance and Integrity Commitment

The team solemnly commits:

1. All submitted training logs, code, and weights are authentic and valid, with no tampering, fabrication, or splicing.
2. The final score was generated by independent inference of this solution's code, with no manual annotation, multi-account leaderboard farming, or rule-breaking API calls or other cheating.
3. The team will fully cooperate with the organizing committee's review; if any materials become invalid or cannot be reproduced, it will respond promptly and cooperate to resolve the issue.
4. If, due to special circumstances such as intranet restrictions, materials must be transferred through private channels within the stipulated time.

## 10. Statement

If you have any questions about reproducing the results, please contact the developers for an explanation or to supplement any needed content.
