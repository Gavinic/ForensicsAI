import numpy as np
from sklearn import metrics


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Val best metric ================================ \n"
    for k, v in metric_dict.items():
        str = str + f" {k}={v} "
    str = str + "| \n"
    str += "============================================================================================="
    return str


def get_test_metrics(pred_dict, true_dict):
    metrics_dict = {}
    avg_mae = 0
    for name, pred in pred_dict.items():
        label = true_dict[name].detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        mae = metrics.mean_absolute_error(label, pred)
        metrics_dict[f"mae_{name}"] = mae
        avg_mae += mae
    metrics_dict["avg_mae"] = avg_mae / len(pred_dict)
    return metrics_dict


# ======== Weighted R² ========
WEIGHT_DICT = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


def weighted_r2_score(pred_dict, true_dict):
    """
    y_true, y_pred: shape (N, 5)
    """
    ###
    metrics_dict = {}
    pred_dict = {
        k: v.detach().cpu().float().numpy()
        for k, v in pred_dict.items()
        if k not in ["image_path"]
    }
    true_dict = {
        k: v.detach().cpu().float().numpy()
        for k, v in true_dict.items()
        if k not in ["image_path"]
    }
    all_wr2 = 0
    all_w = 0
    for name, wi in WEIGHT_DICT.items():
        y_t = true_dict[name]
        y_p = np.clip(pred_dict[name], 0, None)
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)  ## the mean here may be incorrect
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        metrics_dict[f"{name}_r2"] = r2
        all_wr2 += wi * r2
        all_w += wi
    metrics_dict["weighted_r2"] = all_wr2 / all_w

    return metrics_dict


from sklearn.metrics import f1_score, precision_score, recall_score


def pixel_label_f1(pred_dict, true_dict):
    """
    pred_dict: {'pred_label':shape(B,), 'pred_mask':(B,1,H,W)}
    true_dict: {'label':shape(B,), 'gt_mask':(B,1,H,W)}
    """
    ###
    # Data preprocessing (convert to CPU numpy)
    pred_label = pred_dict["pred_label"].detach().float().sigmoid().cpu().numpy() > 0.5

    gt_label = true_dict["label"].detach().float().cpu().numpy() > 0.5

    pred_mask = pred_dict["pred_mask"].detach().float().sigmoid().cpu().numpy() > 0.5
    gt_mask = true_dict["gt_mask"].detach().float().cpu().numpy() > 0.5

    # 1. Compute S_Det (sample-level F1)
    s_det = f1_score(gt_label, pred_label, zero_division=0)

    # 2. Compute S_Loc (pixel-level F1)
    # .ravel() flattens the multi-dimensional mask to 1D so sklearn can compute the pixel overlap
    s_loc = f1_score(gt_mask.ravel(), pred_mask.ravel(), zero_division=0)

    return {
        "S_Det": s_det,
        "S_Loc": s_loc,
        "Precision_Det": precision_score(gt_label, pred_label, zero_division=0),
        "Recall_Det": recall_score(gt_label, pred_label, zero_division=0),
        "S_Fin": 0.45 * s_det + 0.25 * s_loc,
    }
