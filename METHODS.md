# Document Security — Forgery Analysis Methods

> A collection of multi-modal large-model methods for scene text image forgery analysis.
> This directory contains the reference implementations of 7 method variants. All content has been desensitized.

## Method Overview

| Method | Full Name | Technical Route | Status | Main Models |
|--------|-----------|-----------------|--------|-------------|
| **FORGE** | Forensics-Oriented Recursive Guidance Engine | ForensicsSAM classification + HRNet/ViT segmentation + QLoRA explanation + SAM+LoRA | ✅ Accepted | ForensicsSAM, HRNet, ViT, Qwen2.5-QLoRA, SAM |
| **APEX** | Adaptive Patch-level EXplanation | CO-DETR object detection + Qwen2.5-VL LLM reasoning | ⚠️ Conditionally accepted | CO-DETR, Qwen2.5-VL |
| **SWIFT** | Structured Workflow for Image Forensics Training | ms-swift fine-tuning of Qwen-VL (two-stage) | ⚠️ Conditionally accepted | Qwen-VL (ms-swift) |
| **TWIN** | Two-Weight Integrated Network | SegFormer segmentation + QwenVL dual-LoRA reasoning | ⚠️ Conditionally accepted | SegFormer, QwenVL LoRA×2 |
| **FLUX** | Fusion Learning with Unified eXpertise | VLM (vllm+merge) inference + data preprocessing | ⚠️ Conditionally accepted | Qwen-VL (vllm) |
| **PRISM** | Patch-level Reasoning & Integrated Segmentation Model | MaxViT classification + Co-DETR detection + Qwen3.5 VLM | ⚠️ Conditionally accepted | MaxViT, Co-DETR, Qwen3.5 |
| **VIGIL** | Visual Intelligence with GRPO-guided Inference Learning | IDBM-ViT detection + Qwen3-VL LoRA | ⚠️ Conditionally accepted | IDBM-ViT, Qwen3-VL |

## Technical Route Comparison

### Classification / Detection (Image Tampering Recognition)
| Method | Classification Model | Detection Model | Characteristics |
|-----|---------|---------|------|
| FORGE | ForensicsSAM + K-fold | - | 4-fold cross-validation ensemble |
| APEX | - | CO-DETR | Object detection to localize tampered regions |
| PRISM | MaxViT | Co-DETR | Classification + detection dual model |
| VIGIL | - | IDBM-ViT | Image tampering boundary detection |

### Semantic Segmentation (Pixel-level Tampering Localization)
| Method | Segmentation Model | Characteristics |
|-----|---------|------|
| FORGE | HRNet/ViT | Dual-model fusion |
| TWIN | SegFormer | Lightweight segmentation |

### VLM Explanation (Natural-Language Tampering Analysis)
| Method | VLM Model | Fine-tuning | Characteristics |
|-----|---------|---------|------|
| FORGE | Qwen2.5 | QLoRA | 14B model quantized fine-tuning |
| APEX | Qwen2.5-VL | LoRA | Detection-guided reasoning |
| SWIFT | Qwen-VL | ms-swift full-parameter | Two-stage reasoning |
| TWIN | QwenVL | Dual-LoRA | LoRA-A + LoRA-B ensemble |
| FLUX | Qwen-VL | vllm inference | Data augmentation + model fusion |
| PRISM | Qwen3.5 | LoRA | Multi-turn dialogue fine-tuning |
| VIGIL | Qwen3-VL | LoRA (GRPO) | Reinforcement-learning fine-tuning |

## Directory Structure

```
FORGE/    Forensics-Oriented Recursive Guidance Engine
├── stage1/classification/   Image tampering classification (ForensicsSAM)
├── stage1/segmentation/     Pixel-level segmentation (HRNet/ViT)
├── stage1/explanation/      Tampering explanation generation (QLoRA Qwen2.5)
└── stage2/                   Second stage (SAM + LoRA)

APEX/     Adaptive Patch-level EXplanation
├── src/infer_det.py         Object detection inference
├── src/infer_llm.py         LLM inference
└── scripts/                 Run scripts

SWIFT/    Structured Workflow for Image Forensics Training
├── src/                     Inference code (two-stage)
└── scripts/                 Run scripts

TWIN/     Two-Weight Integrated Network
├── src/qwenvl/              QwenVL training and inference
├── src/train_segformer.py   SegFormer training
└── scripts/                 Run scripts

FLUX/     Fusion Learning with Unified eXpertise
├── src/                     Data processing + inference
└── scripts/                 Run scripts

PRISM/    Patch-level Reasoning & Integrated Segmentation Model
├── src/classification/      MaxViT classification
├── src/vlm/                 VLM fine-tuning and inference
└── scripts/                 Run scripts

VIGIL/    Visual Intelligence with GRPO-guided Inference Learning
├── src/detection/           IDBM-ViT detection
├── src/vlm/                 VLM inference and fine-tuning
└── scripts/                 Run scripts
```

## Desensitization Notes

- Team identifiers and member names have been removed.
- Hardcoded server paths have been replaced with `<BASE_PATH>`.
- API keys have been replaced with `<YOUR_API_KEY>` / `<YOUR_ACCESS_KEY>`.
- Third-party framework repositories (Co-DETR/mmdet, ms-swift, LlamaFactory, etc.) are not included.
- Model weight files are not included; base models must be downloaded separately per each method's instructions.

## Notes

- Replace `<BASE_PATH>` in each method's code with the actual model/data path before running.
- Base models (Qwen2.5-VL, SAM, etc.) must be downloaded independently.
- `adapter_config.json` may still contain hardcoded paths; override them at runtime as needed.
