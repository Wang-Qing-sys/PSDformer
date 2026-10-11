"""Frozen-checkpoint evaluation on the saved participant-disjoint partition."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import WindowDataset, dataset_fingerprint, load_npz, partition_indices, validate_split
from .metrics import classification_metrics, forecasting_metrics
from .preprocessing import StandardNormalizer
from .utils import canonical_task, json_safe, resolve_device, write_json


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    from .model import PSDformer, PSDformerConfig
    device = resolve_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != "PSDformer":
        raise ValueError("Not a supported PSDformer reference checkpoint.")
    model = PSDformer(PSDformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    normalizers = {"x": StandardNormalizer.from_dict(checkpoint["normalizers"]["x"]),
                   "target": StandardNormalizer.from_dict(checkpoint["normalizers"]["target"])}
    return model, checkpoint, normalizers


def load_model_data(data_path: str | Path, config: dict) -> dict[str, np.ndarray]:
    return load_npz(data_path, input_len=config["input_len"], input_dim=config["input_dim"],
                    pred_len=config["pred_len"], output_dim=config["output_dim"],
                    num_classes=config.get("num_classes"))


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device: torch.device,
                        target_normalizer: StandardNormalizer | None = None) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list] = {name: [] for name in ("forecast", "target", "y", "index", "subject_ids")}
    for batch in loader:
        output = model(batch["x"].to(device), batch["mask"].to(device))
        forecast = output["forecast"].detach().cpu().numpy()
        if not np.isfinite(forecast).all():
            raise FloatingPointError("Model produced nonfinite forecasts.")
        target = batch["target"].numpy()
        if target_normalizer is not None:
            forecast, target = target_normalizer.inverse_transform(forecast), target_normalizer.inverse_transform(target)
        collected["forecast"].append(forecast)
        collected["target"].append(target)
        if output.get("logits") is not None:
            if not torch.isfinite(output["logits"]).all():
                raise FloatingPointError("Model produced nonfinite logits.")
            collected.setdefault("probs", []).append(torch.softmax(output["logits"], -1).cpu().numpy())
        for name in ("y", "index"):
            collected[name].append(batch[name].numpy())
        collected["subject_ids"].append(np.asarray(batch["subject_id"]))
    if not collected["y"]:
        raise ValueError("Requested partition contains no windows.")
    return {name: np.concatenate(values, axis=0) for name, values in collected.items()}


def summarize_predictions(predictions: dict[str, np.ndarray], task: str = "joint") -> dict[str, Any]:
    task = canonical_task(task)
    result = {"n_windows": len(predictions["y"]), "n_subjects": len(np.unique(predictions["subject_ids"]))}
    if task in ("forecast", "joint"):
        result["forecasting"] = forecasting_metrics(predictions["target"], predictions["forecast"])
        result["forecasting_per_output"] = [forecasting_metrics(predictions["target"][..., c], predictions["forecast"][..., c])
                                            for c in range(predictions["target"].shape[-1])]
        result["forecasting_per_horizon"] = [forecasting_metrics(predictions["target"][:, k], predictions["forecast"][:, k])
                                             for k in range(predictions["target"].shape[1])]
    if task in ("classification", "joint"):
        if "probs" not in predictions:
            raise ValueError("Classification task requires a classification head.")
        result["classification"] = classification_metrics(predictions["y"], predictions["probs"])
    return result


def checkpoint_indices(data: dict, checkpoint: dict, partition: str) -> np.ndarray:
    if partition == "all":
        return np.arange(len(data["x"]))
    if dataset_fingerprint(data) != checkpoint["dataset_fingerprint"]:
        raise ValueError("Data differs from the training NPZ. Saved-partition evaluation requires original data; use predict --partition all for new data.")
    split = validate_split(checkpoint["split"], data["subject_ids"])
    return partition_indices(data["subject_ids"], split, partition)


def evaluate_checkpoint(data_path: str | Path, checkpoint_path: str | Path,
                        output_path: str | Path, device: str = "cpu", batch_size: int = 32,
                        partition: str = "test") -> dict[str, Any]:
    if partition not in ("train", "val", "test"):
        raise ValueError("Evaluation requires an explicit train/val/test partition.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    device_obj = resolve_device(device)
    model, checkpoint, normalizers = load_checkpoint(checkpoint_path, device_obj)
    data = load_model_data(data_path, checkpoint["model_config"])
    indices = checkpoint_indices(data, checkpoint, partition)
    loader = DataLoader(WindowDataset(data, indices, normalizers["x"], normalizers["target"]), batch_size=batch_size, shuffle=False)
    predictions = collect_predictions(model, loader, device_obj, normalizers["target"])
    result = {"partition": partition, "checkpoint": str(checkpoint_path), "task": checkpoint["task"],
        "synthetic_demo": bool(checkpoint.get("synthetic_demo", False)), "subject_ids": checkpoint["split"][partition],
        "metrics": summarize_predictions(predictions, checkpoint["task"]),
        "units": "original target units after inverse training-only normalization",
        "metric_conventions": {"rse": "sqrt(SSE/(n_valid-2)); undefined when n_valid<=2",
                               "r2": "1-SSE/SST; undefined for constant target",
                               "undefined": "JSON null", "aggregate": "micro over observed target cells",
                               "classification": "macro over all configured classes; zero division=0"}}
    result = json_safe(result)
    write_json(output_path, result)
    return result
