import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mmdet.apis import inference_detector, init_detector
from pycocotools import mask as maskUtils
from tqdm import tqdm


def merge_masks(masks):
    """Merge RLE masks"""
    if not masks:
        return ""
    merged = maskUtils.merge(masks)
    merged["counts"] = merged["counts"].decode("utf-8")
    return json.dumps(merged)


def main():
    # create argument parser
    parser = argparse.ArgumentParser(description="Input and output paths")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Input dataset path"
    )
    # parser.add_argument('--output_path', type=str, required=True,
    #                    help='Output file path')

    # parse arguments
    args = parser.parse_args()

    # get paths
    root_dir = Path(__file__).parent.parent.absolute()

    input_path = os.path.join(root_dir, args.input_path)
    # output_path = os.path.join(root_dir, args.output_path)
    cls_ckpt = os.path.join(root_dir, "models/cls/cls.pth")
    cls_cfg = os.path.join(
        root_dir, "src/codetr/forgery_configs/co_dino_swinl_m1920_ok.py"
    )
    cls_model = init_detector(cls_cfg, cls_ckpt, device="cuda:0")

    det_ckpt = os.path.join(root_dir, "models/det/swinl.pth")
    det_cfg = os.path.join(root_dir, "src/codetr/forgery_configs/co_dino_swinl_m1920.py")
    det_model = init_detector(det_cfg, det_ckpt, device="cuda:0")

    # red_dir = os.path.join((root_dir,"data/LLM/RedImage")
    out_dir = os.path.join(root_dir, "tmp_dir")
    os.makedirs(out_dir, exist_ok=True)
    red_dir = f"{out_dir}/Image"
    os.makedirs(red_dir, exist_ok=True)

    # single-image inference
    data = {"image_name": [], "label": [], "location": [], "explanation": []}
    cleaned_annotations = []
    for img_name in tqdm(os.listdir(input_path)):
        print(img_name)
        # classification inference
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)

        cls_pred = inference_detector(cls_model, img_path)
        if cls_pred[1][1][0][0] > 0.03:
            label = 1
        else:
            label = 0

        # segmentation inference
        seg_pred = inference_detector(det_model, img_path)
        rles = []

        for bbox, mask, score in zip(
            seg_pred[0][0], seg_pred[1][0][0], seg_pred[1][1][0]
        ):
            mask = mask.astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(mask))
            if score > 0.2:
                x1, y1, x2, y2 = [float(tmp) for tmp in bbox[:4]]

                padding = 5
                x1_expanded = max(0, x1 - padding)
                y1_expanded = max(0, y1 - padding)
                x2_expanded = min(img.shape[1], x2 + padding)  # img.shape[1] is width
                y2_expanded = min(img.shape[0], y2 + padding)  # img.shape[0] is height

                # convert to integers (OpenCV requires integer coordinates)
                x1, y1, x2, y2 = (
                    int(x1_expanded),
                    int(y1_expanded),
                    int(x2_expanded),
                    int(y2_expanded),
                )

                # draw a red box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cleaned_annotations.append(
                    {"image_id": img_name, "bbox": [x1, y1, x2 - x1, y2 - y1]}
                )
                rles.append(rle)

        if len(rles) == 0:
            for bbox, mask, score in zip(
                seg_pred[0][0], seg_pred[1][0][0], seg_pred[1][1][0]
            ):
                # print()
                mask = mask.astype(np.uint8)
                rle = maskUtils.encode(np.asfortranarray(mask))
                if score > 0.1:
                    rles.append(rle)

        location = merge_masks(rles)

        data["image_name"].append(img_name)
        data["label"].append(label)
        data["location"].append(location)
        data["explanation"].append("")

        cv2.imwrite(f"{red_dir}/{img_name}", img)

    # save results
    df = pd.DataFrame(data)
    df.to_csv(f"{out_dir}/tmp.csv", index=False)

    json.dump(cleaned_annotations, open(f"{out_dir}/cleaned_annotations.json", "w"))
    print(f"Done! Saved {len(df)} records to tmp.csv")
    print(
        f"Positive samples: {(df['label'] == 1).sum()}, Negative samples: {(df['label'] == 0).sum()}"
    )


if __name__ == "__main__":
    main()
