"""Batch inference using checkpoint preprocessing; accepts unlabeled new windows."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from .data import WindowDataset
from .evaluate import checkpoint_indices, collect_predictions, load_checkpoint, load_model_data
from .utils import resolve_device, write_json


def predict_file(data_path: str | Path, checkpoint_path: str | Path, output_path: str | Path,
                 device: str = "cpu", batch_size: int = 32, partition: str = "all") -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if partition not in ("all", "train", "val", "test"):
        raise ValueError("partition must be all, train, val, or test.")
    device_obj = resolve_device(device)
    model, checkpoint, normalizers = load_checkpoint(checkpoint_path, device_obj)
    data = load_model_data(data_path, checkpoint["model_config"])
    indices = checkpoint_indices(data, checkpoint, partition)
    loader = DataLoader(WindowDataset(data, indices, normalizers["x"], normalizers["target"]), batch_size=batch_size, shuffle=False)
    predictions = collect_predictions(model, loader, device_obj, normalizers["target"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    task = checkpoint["task"]
    forecasting = task in ("forecast", "joint")
    classification = task in ("classification", "joint")
    if output.suffix.lower() == ".npz":
        values = {"index": predictions["index"], "subject_ids": predictions["subject_ids"]}
        if forecasting:
            values["forecast"] = predictions["forecast"]
        if classification:
            values["probs"] = predictions["probs"]
            values["predicted_class"] = predictions["probs"].argmax(-1)
        np.savez_compressed(output, **values)
    elif output.suffix.lower() == ".csv":
        headers = ["index", "subject_id"]
        if forecasting:
            _, horizon, outputs = predictions["forecast"].shape
            headers += [f"forecast_t{k+1}_o{o}" for k in range(horizon) for o in range(outputs)]
        if classification:
            headers += ["predicted_class"] + [f"prob_class_{c}" for c in range(predictions["probs"].shape[1])]
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            for i in range(len(predictions["index"])):
                row = [int(predictions["index"][i]), str(predictions["subject_ids"][i])]
                if forecasting:
                    row.extend(predictions["forecast"][i].ravel().tolist())
                if classification:
                    row.extend([int(predictions["probs"][i].argmax()), *predictions["probs"][i].tolist()])
                writer.writerow(row)
    else:
        raise ValueError("Prediction output extension must be .csv or .npz.")
    write_json(output.with_suffix(output.suffix+".json"), {"checkpoint": str(checkpoint_path), "partition": partition,
        "task": task, "n_windows": len(indices), "forecast_units": "original target units", "input_order_preserved": True,
        "synthetic_demo_checkpoint": bool(checkpoint.get("synthetic_demo", False)),
        "warning": "Only heads trained for the saved task are exported. Predictions are not clinical diagnoses."})
    return output
