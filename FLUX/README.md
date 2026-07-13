# Scene Text Image Forgery Analysis Based on Multimodal Large Models

---

## 1. Hardware and Time Estimation

| Hardware | Requirement | Estimated Training Time | Estimated Inference Time |
|--------|------|------|------|
| Single L40 | 48G VRAM | Training takes 4h on a single A800 80g; tested to run on a 48g VRAM card, estimated training time 4-6h | 700 samples take 5 minutes total on a single A800 80g, i.e. 0.42s per sample

---

## 2. Environment Setup Guide

### 2.1 Environment Requirements
Because the training and inference environments conflict, they are separated into `requirements_train.txt` and `requirements_infer.txt` in this directory. Two virtual environments must be configured before running: one for training and one for inference.

### 2.2 Environment Creation Commands
First ensure that anaconda or miniconda is installed and that conda is available in the current bash environment, then run the following commands to create the training and inference environments.
#### 2.2.1 Create the Training Environment
conda create -n train python=3.10

conda activate train

pip install -r requirements_train.txt -i https://mirrors.aliyun.com/pypi/simple

Install flash-attn:

pip install flash-attn==2.6.0.post1 --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple

#### 2.2.2 Create the Inference Environment
conda create -n infer python=3.10

conda activate infer

pip install -r requirements_infer.txt -i https://mirrors.aliyun.com/pypi/simple

---

## 3. Model Acquisition Notes

The LoRA model weights are included. The base model (qwen3-vl-8b-instruct) needs to be downloaded from the internet.

The download script is `download_scipt.sh` in the `models` folder. The downloaded model should be placed in the `models` folder, forming the directory structure `models/Qwen3-VL-8B-Instruct`.

---

## 4. Run Guide
Note: The same model is used for the preliminary and final rounds; train once and then run inference on the preliminary and final test sets respectively.
### 4.1 Data Preparation
1. Set up the `data` directory. Place the officially provided training set, test set, and any additional personal datasets under the `data` directory. The `data` directory should be organized as follows:

data/

├── anytext/

├── diffste/

├── DiffUTE_LG_500/

├── DiffUTE_LG_500_2/

├── ForgeryAnalysis_Stage_1_Test/

├── ForgeryAnalysis_Stage_1_Train/

├── ForgeryAnalysis_Stage_2_Test/

├── mostel/

├── OSTF_LG_336/

├── OSTF_LG_384/

├── srnet/

├── stefann/

└── udifftext/

2. Run the data processing script to generate the trainable jsonl file:

bash ./scripts/process_data.sh

### 4.2 Training
conda activate train

python ./src/train_json.py

Training takes about 4-6 hours. After training, the model weights are saved in the `models/lora_checkpoints` directory, and training logs are saved under `models/lora_checkpoints/logs`.
### 4.3 Inference
1. To run inference on the preliminary-round dataset, run:

bash ./scripts/inference_chusai.sh

2. To run inference on the final-round dataset, run:

bash ./scripts/inference_fusai.sh

Inference requires merging the LoRA model with the original model. The merged model is saved in the `models/Qwen3-VL-8B-Merged` directory and is used for inference. After this script finishes, the `result_chusai.csv` / `result_fusai.csv` files in the `data/result` directory are the final preliminary/final submission results.

---

## 5. External Data Statement

### 5.1 External Open-Source Datasets Used

Dataset usage notes:

Data sources: (1) OSTF (2) TFR

Data usage:
(1) TFR: copied from the train_data and test_data of this dataset to obtain DIffute_LG_500, DIffute_LG_500_2, OSTF_LG_336, and OSTF_LG_384

(2) OSTF: sampled anytext, diffste, mostel, srnet, stefann, and udifftext from this dataset
