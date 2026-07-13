### Overview
Our overall solution trains three tasks: classification + instance segmentation + multimodal description. The classification code is under the `classification` directory, instance segmentation under the `Co-DETR` directory, and multimodal description code under the `Forgery` directory. Training and inference are each run in separate conda environments. Hardware: 4x L20 GPUs; a single L40 should be sufficient for inference.

### Training Reproduction
#### Download Pretrained Weights
Enter the `models` folder and download the pretrained weights.

- Classification weights

wget -O maxvit_xlarge_tf_512.in21k_ft_in1k.bin https://huggingface.co/timm/maxvit_xlarge_tf_512.in21k_ft_in1k/resolve/main/pytorch_model.bin

wget -O maxvit_large_tf_512.in21k_ft_in1k.bin https://huggingface.co/timm/maxvit_large_tf_512.in21k_ft_in1k/resolve/main/pytorch_model.bin

wget -O maxvit_base_tf_512.in21k_ft_in1k.bin https://huggingface.co/timm/maxvit_base_tf_512.in21k_ft_in1k/resolve/main/pytorch_model.bin

- Segmentation weights

wget -O co-detr-vit-large-coco-instance.pth https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance/resolve/main/pytorch_model.pth

- Multimodal description weights

Note: If the qwen3.5-9B model fails to download, try an alternative mirror and place it in this directory.

brew install git-xet

git xet install

git clone https://huggingface.co/Qwen/Qwen3.5-9B

#### Classification Training Environment Setup
Create a Python 3.10 conda environment, then run the `requirements.txt` under `src/hw`:

conda create -n python310 python=3.10 -y

cd src/hw

pip install -r requirements.txt

#### Instance Segmentation Training Environment Setup
Create a Python 3.8 conda environment, then run the `requirements.txt` under `src/Co-DETR`:

conda create -n codetr python=3.8 -y

cd src/Co-DETR

pip install -r requirements.txt

pip install -v -e .

#### Multimodal Description Training Environment Setup
Create a Python 3.12 conda environment, then run the `requirements.txt` under `src/Forgery`:

conda create -n qwen3.5 python=3.12 -y

cd src/Forgery

pip install -r requirements.txt

cd causal-conv1d

pip install -v -e .

cd ../flash-linear-attention

pip install -v -e .

### Start Training
Before training, ensure all pretrained weights are downloaded and placed under the `models` folder. Copy the competition dataset to the `data` directory of the corresponding module, e.g. `<BASE_PATH>/data`.

The data required for training and validation is included in the corresponding code directories.

Training command: `bash scripts/train.sh`

### Inference Reproduction

Assuming all environments are installed without errors and the dataset directory has been placed under the current `data` directory, inference can be run normally.

Inference command: `bash scripts/inference.sh --input_path <BASE_PATH>/data --output_path <BASE_PATH>/output.csv`
