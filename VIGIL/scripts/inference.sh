
export CUDA_VISIBLE_DEVICES=0

# Detection model version (date)
DETECT_DATE='HT-16_2026-03-26-16-51-24'

# Inference path, using wildcards
TEST2_PATH='<BASE_PATH>/data/2026forgery/data/ForgeryAnalysis_Stage_2_Test/Image/*'

## ================================= Stage1 Detection ==================================================
python3 ../src/detection/training/local_infer.py  --config  ../models/idbmvit_idmbvit_${DETECT_DATE}/idbm_vit.yaml \
    --model_path ../models/idbmvit_idmbvit_${DETECT_DATE}/test/foldonetrain_0_ckpt_best.pth \
    --test_path  "${TEST2_PATH}" \
    --savedir ../tmp/vit_dinov3_idmbvit_${DETECT_DATE}_test2 \
    --submit  ../tmp/${DETECT_DATE}_test2.csv
# #


# ### Download base models: Qwen3-Embedding-8B  Qwen3-VL-8B-Instruct

bash ../models/download_script.sh

# ## Merge LoRA weights

python3 ../src/llmexp/merge_lora.py

##  ============++++++++++++++++++==== Stage2 VLM ==========================================================================

## ============== 1. Preliminary inference results, used for subsequent retrieval enhancement =================
VLM_MODEL=../models/Qwen3-VL-8B-Instruct
python3 ../src/llmexp/vlm_inference_vllm_v2.py  --csv_file ../tmp/${DETECT_DATE}_test2.csv \
    --vlm_model "${VLM_MODEL}" \
    --image_path ../tmp/vit_dinov3_idmbvit_${DETECT_DATE}_test2 \
    --output_path  ../tmp/qwen3_8b_instruction_${DETECT_DATE}_test2.csv

# # ================== 2. Indexing ==========================
# ## Index database path
INDEX_DB=../tmp/${DETECT_DATE}_indexdb
EMBEDING_MODEL=../models/Qwen3-Embedding-8B ## embedding model: replace with local path if offline

python3 ../src/llmexp/make_indexs.py --vector_db_root "${INDEX_DB}" \
    --embeding_model "${EMBEDING_MODEL}" \
    --test_csv_file  ../tmp/qwen3_8b_instruction_${DETECT_DATE}_test2.csv \
    --ori_input_csv ../tmp/${DETECT_DATE}_test2.csv \
    --caption_paths "../data/train/Black/Caption/*" "..//data/2026forgery/data/train/White/Caption/*" \
    --output_path ../tmp/qwen3_8b_instruction_${DETECT_DATE}_with_index_test2.csv

# ================ 3. Enhanced generation ================================
VLM_MODEL=../models/qwen3_vl_8b_v3_sft_merged_dici
python3 ../src/llmexp/vlm_inference_vllm_cls.py --csv_file ../tmp/qwen3_8b_instruction_${DETECT_DATE}_with_index_test2.csv \
    --vlm_model "${VLM_MODEL}" \
    --vector_db_root "${INDEX_DB}" \
    --image_path ../tmp/vit_dinov3_idmbvit_${DETECT_DATE}_test2 \
    --output_path ./qwen3_8b_lorasftv3_instruction_${DETECT_DATE}_submit.csv
