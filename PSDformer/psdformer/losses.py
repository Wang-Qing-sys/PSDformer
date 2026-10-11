"""Masked forecast MSE and optional, explicitly labelled classification CE."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F


def _graph_zero(tensor: Tensor) -> Tensor:
    """Finite graph-connected zero, even if unused predictions contain NaN/Inf.

    Multiply each sanitized element before summing to avoid sum overflow.
    """
    return (torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0) * 0.0).sum()


def multitask_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    task: str = "joint",
    forecast_weight: float = 1.0,
    classification_weight: float = 1.0,
) -> dict[str, Tensor | int]:
    """Return task means, sums and counts for exact epoch aggregation.

    ``output['forecast']`` is floating point [B,K,O]; an enabled classification
    task also needs ``output['logits']`` [B,C], C >= 1. Forecast supervision is
    ``batch['target']`` of the same shape, with nonfinite entries missing.
    Classification supervision is integer ``batch['y']`` [B], with -1 missing.
    No labels are inferred from forecasts or other labels.

    Task means are unweighted; ``total`` combines them using the supplied
    weights. A task excluded by ``task`` or assigned weight zero is disabled:
    its target is not required, its count is zero, and its mean/sum is a
    graph-connected zero. Enabled tasks with no supervision also return zero.
    At least one selected task must have a positive weight. The sums and counts
    let callers aggregate across uneven batches without averaging batch means.
    """
    if task not in {"forecast", "classification", "joint"}:
        raise ValueError("task must be 'forecast', 'classification', or 'joint'.")
    for name, weight in (("forecast_weight", forecast_weight),
                         ("classification_weight", classification_weight)):
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be a finite, nonnegative number.")
    use_forecast = task in {"forecast", "joint"} and forecast_weight > 0
    use_classification = task in {"classification", "joint"} and classification_weight > 0
    if not (use_forecast or use_classification):
        raise ValueError("At least one selected task must have a positive weight.")
    if "forecast" not in output or not isinstance(output["forecast"], Tensor):
        raise ValueError("output requires a forecast tensor with shape [B,K,O].")
    forecast = output["forecast"]
    if forecast.ndim != 3 or min(forecast.shape[1:]) < 1:
        raise ValueError("forecast must have shape [B,K,O] with positive K and O.")
    if not forecast.is_floating_point():
        raise TypeError("forecast must have a real floating point dtype.")
    zero = _graph_zero(forecast)
    forecast_sum = zero
    classification_sum = zero
    n_forecast = n_classification = 0

    if use_forecast:
        target = batch.get("target")
        if not isinstance(target, Tensor) or target.shape != forecast.shape:
            raise ValueError("batch target must be a tensor matching forecast shape [B,K,O].")
        if not target.is_floating_point():
            raise TypeError("batch target must have a real floating point dtype.")
        if target.device != forecast.device:
            raise ValueError("forecast and target must be on the same device.")
        observed = torch.isfinite(target)
        n_forecast = int(observed.sum().item())
        selected_forecast = forecast[observed]
        if not torch.isfinite(selected_forecast).all().item():
            raise ValueError("Forecasts at observed targets must be finite.")
        if n_forecast:
            # Select BEFORE subtraction: multiplying a NaN residual by zero
            # afterwards would still poison the loss and its derivatives.
            forecast_sum = F.mse_loss(selected_forecast, target[observed], reduction="sum")

    if use_classification:
        logits = output.get("logits")
        if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[0] != forecast.shape[0] or logits.shape[1] < 1:
            raise ValueError("An enabled classification task requires logits [B,C], C >= 1.")
        if not logits.is_floating_point():
            raise TypeError("logits must have a real floating point dtype.")
        if logits.device != forecast.device:
            raise ValueError("logits and forecast must be on the same device.")
        labels = batch.get("y")
        if not isinstance(labels, Tensor) or labels.shape != (forecast.shape[0],):
            raise ValueError("batch y must be an integer tensor with shape [B].")
        if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("batch y must have an integer dtype; use -1 for missing labels.")
        if labels.device != logits.device:
            raise ValueError("logits and y must be on the same device.")
        # Widen before comparing with -1 or C: narrow/unsigned integer scalar
        # comparisons can otherwise wrap their bounds (for example uint8 -1).
        labels = labels.long()
        if ((labels < -1) | (labels >= logits.shape[1])).any().item():
            raise ValueError("y must be -1 or a class ID in [0,C).")
        labelled = labels >= 0
        n_classification = int(labelled.sum().item())
        selected_logits = logits[labelled]
        if not torch.isfinite(selected_logits).all().item():
            raise ValueError("Logits for labelled examples must be finite.")
        classification_sum = _graph_zero(logits) + zero
        if n_classification:
            classification_sum = F.cross_entropy(selected_logits, labels[labelled], reduction="sum")

    forecast_loss = forecast_sum / n_forecast if n_forecast else forecast_sum
    classification_loss = classification_sum / n_classification if n_classification else classification_sum
    total = forecast_weight * forecast_loss + classification_weight * classification_loss
    return {
        "total": total,
        "forecast": forecast_loss,
        "classification": classification_loss,
        "forecast_sum": forecast_sum,
        "classification_sum": classification_sum,
        "n_forecast": n_forecast,
        "n_classification": n_classification,
    }
