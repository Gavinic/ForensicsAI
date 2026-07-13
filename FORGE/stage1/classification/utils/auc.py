import numpy as np
from sklearn.metrics import roc_auc_score


def cal_auc(outputs, gt):
    trans_output = outputs.flatten()
    trans_gt = gt.flatten().astype(np.int32)
    try:
        roc_auc = roc_auc_score(trans_gt, trans_output)
    except ValueError:
        roc_auc = 0
        # pass  ## or other definitions, e.g. roc_auc=0
    auc = roc_auc

    return auc


if __name__ == "__main__":
    # outputs = np.random.randint(0, 2, (1, 10))
    # print(outputs)
    # gt = np.random.randint(0, 2, (1, 10))
    # print(gt)
    outputs = np.array([[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]], dtype=np.int32)
    gt = np.array([[0, 0, 1, 1, 0, 1, 1, 1, 1, 0]], dtype=np.int32)
    outputs = outputs.reshape((1, 2, 5))
    gt = gt.reshape((1, 2, 5))
    print(outputs.shape, gt.shape)
    print(cal_auc(outputs, gt))
