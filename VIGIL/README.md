# Multimodal Large Model Scene Text Image Forgery Attribution Analysis System

This project targets the application of multimodal large models in the AI security domain, building an end-to-end integrated "detection-localization-explanation" forgery analysis system. The system adopts a two-stage decoupled architecture of "visual perception + attribution explanation": Stage1 focuses on high-precision visual anomaly localization (based on DINOv3 SVD-enhanced fine-tuning), and Stage2 leverages MLLM reasoning to achieve hallucination-free logical attribution (based on Qwen3-VL, dynamic RAG enhancement, and post-training alignment), achieving a final weighted score of 0.84778 (preliminary round).

## Hardware and Runtime Estimation

To ensure smooth large-model inference and high-resolution image processing, the following hardware configuration is recommended:
- Minimum hardware requirements: * GPU: a single NVIDIA A100 / RTX 4090 or equivalent card. VRAM: $\ge$ 48GB (the system uses the vLLM framework for inference acceleration; 48GB can meet the inference needs of an 8B-scale VLM combined with the RAG pipeline).
- RAM: $\ge$ 64GB (for loading the retrieval vector database and data preprocessing).
- Estimated inference time:
    * Stage1 (visual detection): ~0.05 s/sample
    * Stage2 (VLM initial detection + RAG retrieval + enhanced attribution): ~3.5 - 4.5 s/sample; total: ~5 - 8 s/sample (varies slightly with input image resolution and text length).


## Environment Setup Guide

It is recommended to use Conda to build the base environment from scratch. Below is a complete example command sequence:

```bash
# 1. Create and activate the base environment
conda create -n forgery_analysis python=3.10 -y
conda activate forgery_analysis

# 2. Install the base PyTorch environment (adjust for your actual CUDA version; here CUDA 12.1 is used as an example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install core dependencies (vLLM for inference acceleration, timm for the visual backbone, LlamaFactory for LoRA fine-tuning)
pip install vllm transformers peft accelerate datasets huggingface_hub
pip install timm pandas numpy opencv-python Pillow

# 4. Install LlamaFactory (required if re-training)
cd src/llmexp/LlamaFactory
pip install -e ".[torch,metrics]"
cd ../../../

# 5. Install system dependencies (for automated script validation)
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install -y jq
# CentOS/RHEL:
# sudo yum install jq
```
Please ensure version compatibility among vllm, transformers, huggingface_hub, etc.

`environment.yml` and `requirements.txt` provide an environment reference.



## Model Acquisition Notes

To be compatible with enterprise intranet and offline evaluation environments, this project explicitly distinguishes bundled weights from base models that require online download.

1. File inclusion status
  - Files already packaged in the project:
    * Stage1 detection model weights and config: all files under `models/idbmvit_idmbvit_HT-16_2026-03-26-16-51-24/`.
    * Fine-tuned LoRA adapter weights (Stage2).
    * Pre-built index of the local RAG vector knowledge base: `tmp/H-16_2026-03-11-12-43-06_indexdb/`.

- Files requiring online download:
    - VLM base model: Qwen/Qwen3-VL-8B-Instruct
    - Embedding base model: Qwen/Qwen3-Embedding-8B
    - Stage1 visual backbone (DINOv3) pretrained weights (downloaded automatically by timm only during training; inference does not depend on external network).


2. Offline deployment plan (for environments without public internet)

If the review machine cannot access the internet, perform the following steps on a networked machine, then copy to the review machine via removable storage or LAN:

**Step 1: Download base models on a networked machine**

```bash
# Enter the models directory
cd Team_Submission/models
# Run the automated download script (with multi-source failover and hash verification)
bash download_script.sh
```
Note: This script automatically downloads `Qwen3-Embedding-8B` and `Qwen3-VL-8B-Instruct` from Hugging Face or a domestic mirror, and verifies integrity.

**Step 2: Copy to the review machine**

Copy the downloaded models to the `models/Qwen3-Embedding-8B` and `models/Qwen3-VL-8B-Instruct` folders, and copy the entire `Team_Submission` directory to the corresponding location on the offline review machine. The `inference.sh` in the project is already configured to read local model paths first.

**Step 3: Place the data**

Place the official dataset under the `data/train` directory.

## Run Guide

Before running the following scripts, make sure the terminal's current working directory is under the `scripts` folder.

###  **Inference Review (Inference)**

The inference script contains the complete integrated pipeline: visual anomaly localization -> LoRA weight merging -> preliminary VLM inference -> vector index retrieval -> enhanced generation and fine-tuned model output.

Before running, make sure the models from the previous step have been downloaded and copied, and set `TEST2_PATH` in `inference.sh` to the actual inference image path.

**Make sure** `data/train` contains the competition **training data** `Black` and `White`.
```bash
cd Team_Submission/scripts
# Grant execute permission to the script
chmod +x inference.sh
# Run the inference script
bash inference.sh
```
After completion, the final submission file will be generated at: `script/qwen3_8b_lorasftv3_instruction_HT-16_2026-03-26-16-51-24_submit.csv`

### Training Review (Training)

If you need to reproduce the training process from scratch, follow these steps:

1. Data preparation:
    - Obtain augmented data according to `data/额外数据集地址.txt` and extract it to the corresponding location under `data/`.
    - The project provides preprocessed metadata: `data/cutted_datasets_alls_array.npy` (to avoid repeated traversal).
    - `data/train.csv` (Tianchi official training set).
    - `data/vertices_lesseq4_pos.csv` (filtered sample metadata).
    - Modify the data loading path in `train.py`.

Start training:

(Note: running the training script requires an internet connection so that timm can download the pretrained detection backbone.)

1. You need to modify the corresponding data loading path in `src/detection/training/train.py`.
2. The actual experiments performed multi-step data fine-tuning; see the training logs in `logs/` for details.

```bash
cd Team_Submission/scripts
chmod +x train.sh
bash train.sh
```
`train.sh` is for demonstration only; it covers detection model fine-tuning, multimodal SFT instruction fine-tuning based on LlamaFactory, and RL post-training code.

## External Data and Tools Statement

Per the competition rules, this solution complies with the use of the following external open-source datasets and tools, hereby declared:

- External data sources: see the addresses provided in `data/额外数据集地址.txt`.
- The lightweight local vector knowledge base dynamically recalls samples to construct Few-shot examples, based on the competition-provided training set `Cations`.

Third-party open-source frameworks:

- LlamaFactory (Apache 2.0 license): used to efficiently manage and execute multimodal SFT and LoRA fine-tuning.

- vLLM (Apache 2.0 license): used for high-throughput, low-latency large-model inference acceleration.

- timm (Apache 2.0 license): used to load DINOv3 and other visual backbone weights.

Open-source model dependencies:

> Uses `Qwen3-VL-8B-Instruct` and `Qwen3-Embedding-8B` open-sourced by Alibaba Tongyi Qianwen. Usage complies with their official open-source license and requirements.\
All training customization code (RAG construction, data cleaning, and SFT/RL fine-tuning code) is included under the `src/` directory tree, ensuring full transparency and reproducibility.
