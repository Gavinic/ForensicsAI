import numpy as np


def f1_score(y_true, y_pred):
    e = 1e-8
    gp = np.sum(y_true)
    tp = np.sum(y_true * y_pred)
    pp = np.sum(y_pred)
    p = tp / (pp + e)
    r = tp / (gp + e)
    f1 = (2 * p * r) / (p + r + e)
    return f1


def cal_f1(outputs, gt):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    precision = true_positives / (true_positives + false_positives)
    recall    = true_positives / (true_positives + false_negatives)

    :param outputs: Network Output Mask
    :param gt:      Mask GroudTruth
    :return:
    """
    trans_output = outputs.flatten()
    trans_gt = gt.flatten()
    return f1_score(trans_gt, trans_output)


def computeMetrics(pred, gt):
    # Remove extra dimensions (if needed)
    pred = pred.squeeze()  # (256, 256)
    gt = gt.squeeze()  # (256, 256)

    # Compute TP, FP, FN, TN
    TP = np.sum((pred == 1) & (gt == 1))  # True Positive
    FP = np.sum((pred == 1) & (gt == 0))  # False Positive
    FN = np.sum((pred == 0) & (gt == 1))  # False Negative
    TN = np.sum((pred == 0) & (gt == 0))  # True Negative
    return TP, FP, FN, TN


def computeF1(FP, TP, FN, TN):
    return 2 * TP / np.maximum((2 * TP + FN + FP), 1e-32)


def cal_permute_f1(pred, gt):
    TP, FP, FN, TN = computeMetrics(pred, gt)
    f1 = computeF1(FP, TP, FN, TN)
    f1i = computeF1(TN, FN, TP, FP)
    P_F1 = max(f1, f1i)

    return P_F1


def cal_image_level_acc(probs, gt, threshold):
    image_pred = int(probs.max() > threshold)
    image_label = int(gt.max() > threshold)
    return float(image_pred == image_label)
