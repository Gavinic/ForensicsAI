conda run -n infer python ./src/merge_model.py
conda run -n infer python ./src/infre_vllm_merge_chusai.py
conda run -n infer python ./src/data_process_post_chusai.py
