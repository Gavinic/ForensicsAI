conda run -n train python ./src/merge_model.py
conda run -n infer python ./src/infre_vllm_merge_fusai.py
conda run -n infer python ./src/data_process_post_fusai.py
