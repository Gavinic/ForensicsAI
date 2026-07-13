import argparse
import glob
import os

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm


def make_db(model, caption_path, args):
    if os.path.exists(os.path.join(args.vector_db_root, "embedding_db.npy")):
        print("The vector database already exists; skip building it")
        return
    os.makedirs(args.vector_db_root, exist_ok=True)
    #  Read the data
    md_files = []
    for pp in caption_path:
        md_files.extend(glob.glob(pp))
    print(f"Read {len(md_files)} md files")
    all_captions = []
    for p in md_files:
        data = {}
        with open(p, "r") as f:
            lines = f.readlines()[0]
        if not lines:
            print(p)
        t_path = p.replace(".md", ".jpg").replace(
            "Caption", "Image"
        )  # replace 'Caption' with 'Image' in the path to find the image
        if not os.path.exists(t_path):
            t_path = t_path.replace(".jpg", ".png")
        if not os.path.exists(t_path):
            print(t_path)
        data["image_path"] = t_path
        data["caption"] = lines
        all_captions.append(data)
    all_captions = pl.from_dicts(all_captions)
    ### =============Build the vector database==============
    all_lenght = len(all_captions)
    bs = 16
    iters = all_lenght // bs + 1
    ## Vector database
    embedding_db = np.zeros((all_lenght, 4096), dtype=np.float32)
    sentences = all_captions["caption"].to_list()
    image_paths = all_captions["image_path"].to_list()
    for ind in tqdm(range(iters)):
        temp_sentens = sentences[ind * bs : (ind + 1) * bs]
        if len(temp_sentens) == 0:
            break
        # 3. Generate vectors
        embeddings = model.encode(
            temp_sentens
        )  ## this model also supports multimodal and prompt indexing
        embedding_db[ind * bs : (ind + 1) * bs] = embeddings
    ## Save the vector database along with the indexed image names and caption content
    np.save(os.path.join(args.vector_db_root, "caption.npy"), sentences)
    np.save(os.path.join(args.vector_db_root, "image_path.npy"), image_paths)
    np.save(os.path.join(args.vector_db_root, "embedding_db.npy"), embedding_db)


def search_docmentIndex(model, args):
    # Compute the similarity between each sentence and the categories
    train_embedding = np.load(
        os.path.join(args.vector_db_root, "embedding_db.npy")
    ).astype(np.float32)
    test_vlm = pl.read_csv(args.test_csv_file)
    bs = 4
    test_lenght = len(test_vlm)
    iters = test_lenght // bs + 1
    test_sentens = test_vlm["explanation"]
    search_index = []
    for ind in tqdm(range(iters)):
        temp_sentens = test_sentens[ind * bs : (ind + 1) * bs]
        # print([len(i) for i in temp_sentens])
        if len(temp_sentens) == 0:
            break
        # 3. Generate vectors
        embeddings = model.encode(temp_sentens)

        cos_scores = util.cos_sim(embeddings, train_embedding)
        values, indexs = cos_scores.topk(
            args.topk, dim=-1
        )  # keep the indices of the top-k largest values
        for perindex in indexs:
            search_index.append(",".join([str(i.item()) for i in perindex]))
    test_vlm = test_vlm.with_columns(pl.Series(search_index).alias("indexs"))
    return test_vlm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector_db_root", type=str, default=""
    )  # vector database location
    parser.add_argument(
        "--output_path", type=str, default=""
    )  # final result output path
    parser.add_argument(
        "--test_csv_file", type=str, default=""
    )  # preliminary VLM results; used to index captions
    parser.add_argument(
        "--ori_input_csv", type=str, default=""
    )  # original results, used for next-step enhanced generation
    parser.add_argument(
        "--embeding_model", type=str, default="Qwen/Qwen3-Embedding-8B"
    )  # embedding model
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--caption_paths",
        type=str,
        nargs="+",
        default=["data/train/Black/Caption/*", "data/train/White/Caption/*"],
        help="list of caption data paths, space-separated",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    ## Vector database path
    # vector_db_root = "<BASE_PATH>/data/2026forgery/H-16_2026-03-11-12-43-06_indexdb"
    # output_path = "<BASE_PATH>/data/2026forgery/qwen3_8b_instruction_H-16_2026-03-11-12-43-06_with_index.csv"
    # test_csv_file = "<BASE_PATH>/data/2026forgery/qwen3_8b_instruction_H-16_2026-03-11-12-43-06.csv"
    # ori_input_csv = "<BASE_PATH>/data/2026forgery/baseline/training/H-16_2026-03-11-12-43-06.csv" # mainly used to concatenate the original input ori_
    ### ===========Read the data to be indexed========
    ## Read all captions, expressed with wildcards
    caption_path = args.caption_paths
    # ['<BASE_PATH>/data/2026forgery/data/train/Black/Caption/*',
    #                 '<BASE_PATH>/data/2026forgery/data/train/White/Caption/*']
    print("query csv:", args.test_csv_file)
    # 1. Load the Qwen3 embedding model
    # Available sizes: 0.6B (lightweight), 4B, 8B (high performance)
    model = SentenceTransformer(
        args.embeding_model,
        trust_remote_code=True,
        prompts={
            "clustering": "Find the sentence most similar to this attribution description in logic, semantics, and scene"
        },
    )
    ### ===============2. Build the database from the caption data==========================
    make_db(model, caption_path, args)

    ### =============3. Retrieve top-k indices from the vector database based on the query==============
    test_vlm_with_indexs = search_docmentIndex(model, args)

    ## Join the original input, used for the second VLM's preliminary detection conclusion
    ori_detect = pl.read_csv(args.ori_input_csv)
    test_vlm_with_indexs = test_vlm_with_indexs.join(
        ori_detect["image_name", "explanation"],
        how="left",
        on="image_name",
        suffix="_ori",
    )
    ## Output the csv with vector indices
    test_vlm_with_indexs.write_csv(args.output_path)
