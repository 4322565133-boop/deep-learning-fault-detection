import numpy as np

def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def confusion_from_binary(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(np.uint8).reshape(-1)
    gt = gt.astype(np.uint8).reshape(-1)
    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    return tp, tn, fp, fn

def accuracy(tp, tn, fp, fn):
    return _safe_div(tp + tn, tp + tn + fp + fn)

def precision(tp, tn, fp, fn):
    return _safe_div(tp, tp + fp)

def recall(tp, tn, fp, fn):
    return _safe_div(tp, tp + fn)

def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray):
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score)
    scores_sorted = y_score[order]
    true_sorted = y_true[order]

    ranks = np.empty_like(scores_sorted, dtype=np.float64)
    i = 0
    rank = 1.0
    N = len(scores_sorted)
    while i < N:
        j = i
        while j + 1 < N and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        ranks[i:j+1] = avg
        rank += (j - i + 1)
        i = j + 1

    sum_ranks_pos = ranks[true_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def dice_coeff(pred: np.ndarray, gt: np.ndarray, smooth=1.0):
    pred = pred.astype(np.float32).reshape(-1)
    gt = gt.astype(np.float32).reshape(-1)
    inter = (pred * gt).sum()
    return float((2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth))

def iou_coeff(pred: np.ndarray, gt: np.ndarray, smooth=1.0):
    pred = pred.astype(np.float32).reshape(-1)
    gt = gt.astype(np.float32).reshape(-1)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float((inter + smooth) / (union + smooth))
