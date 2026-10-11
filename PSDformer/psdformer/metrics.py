"""NumPy evaluation metrics with explicit missing-label conventions.

Forecast scores pool all observed scalar targets. ``rse`` follows the supplied
manuscript's residual-standard-error equation, sqrt(SSE / (n - 2)); it is not
the alternative relative-squared-error metric sometimes given the same name.
"""

from __future__ import annotations

import numpy as np


def forecasting_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, int | float]:
    """Score equal-shaped arrays over finite targets and an optional mask.

    ``mask`` must have the exact target shape and contain booleans or 0/1.
    Nonfinite predictions at observed targets are errors rather than silently
    omitted samples. All scores are NaN for no observations; RSE additionally
    needs more than two observations, and R2 needs a nonconstant target.
    """
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical shapes.")
    observed = np.isfinite(target)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != target.shape:
            raise ValueError("mask must have the same shape as target.")
        if mask.dtype.kind not in "bui" or not np.isin(mask, [0, 1]).all():
            raise ValueError("mask must contain boolean or integer 0/1 values.")
        observed &= mask.astype(bool)
    y, pred = target[observed], prediction[observed]
    if not np.isfinite(pred).all():
        raise ValueError("Predictions at observed targets must be finite.")
    n = int(y.size)
    result = {"n_valid": n, **dict.fromkeys(("mae", "mse", "rmse", "rse", "r2"), float("nan"))}
    if not n:
        return result
    residual = pred - y
    sse = float(np.sum(np.square(residual), dtype=np.float64))
    mse = sse / n
    sst = float(np.sum(np.square(y - np.mean(y)), dtype=np.float64))
    result.update(
        mae=float(np.mean(np.abs(residual))),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        rse=float(np.sqrt(sse / (n - 2))) if n > 2 else float("nan"),
        r2=1.0 - sse / sst if sst > 0 else float("nan"),
    )
    return result


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    """Score class IDs [N] and class probabilities [N,C], ignoring y=-1.

    C must be at least one. Macro precision, recall and F1 include *every*
    configured class, including unsupported classes, with zero division = 0.
    The confusion matrix has true classes as rows and predictions as columns.
    Unlabelled rows are omitted, including their probability validation. Empty
    labelled data has NaN accuracy/NLL and zero macro scores/support. For NLL,
    exact zero probability is clipped to the smallest positive float64 value.
    """
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError("probabilities must have shape [N,C] with C >= 1.")
    n, classes = probabilities.shape
    if y.shape != (n,):
        raise ValueError("y must have shape [N], matching the probability rows.")
    if y.dtype.kind not in "iuf" or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
        raise ValueError("y must contain integer class IDs, with -1 for missing labels.")
    if np.any(y < -1) or np.any(y >= classes):
        raise ValueError("y must be -1 or a class ID in [0,C).")
    labelled = y >= 0
    labels = y[labelled].astype(np.int64)
    probs = probabilities[labelled]
    if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("Labelled probabilities must be finite values in [0,1].")
    if not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("Each labelled probability row must sum to one.")
    count = int(labels.size)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    if count:
        predicted = np.argmax(probs, axis=1)
        np.add.at(confusion, (labels, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(true_positive, predicted_support, out=np.zeros(classes), where=predicted_support > 0)
    recall = np.divide(true_positive, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(classes), where=(precision + recall) > 0)
    nll = float("nan")
    if count:
        correct_probs = probs[np.arange(count), labels]
        nll = float(-np.log(np.clip(correct_probs, np.finfo(np.float64).tiny, 1.0)).mean())
    return {
        "n_labelled": count,
        "accuracy": float(true_positive.sum() / count) if count else float("nan"),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "nll": nll,
        "confusion_matrix": confusion.tolist(),
        "support": support.tolist(),
    }
