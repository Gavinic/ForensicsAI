#!/bin/bash

cd ../../src/stage2_final/

nohup python inference.py \
  --base_model_path <BASE_PATH>/models/qwen3-vl-8b-instruct \
  --input_path ../../datasets/ForgeryAnalysis_Stage_1_Test/Image \
  --lora_a_model_path ../../models/adapters/stage2_final/lora_a/ \
  --lora_b_model_path ../../models/adapters/stage2_final/lora_b/ \
  --segformer_model_path ../../models/adapters/stage2_final/segformer.pth \
  --test_ocr_path ../../datasets/stage1_preliminary/ocr_test.xlsx \
  --mask_dir output_masks \
  --output_path result.csv \
  --best \
  > ../../scripts/stage1_preliminary/inference.out &
