from __future__ import annotations
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import WindowDataset, dataset_fingerprint, partition_indices, split_subjects, validate_split
from .evaluate import collect_predictions, load_model_data
from .losses import multitask_loss
from .metrics import classification_metrics
from .preprocessing import StandardNormalizer
from .utils import canonical_task, json_safe, read_config, resolve_device, seed_everything, write_json


def _move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _epoch(model, loader, device, task, weights, optimizer=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    sums = {"forecast": 0.0, "classification": 0.0}
    counts = {"forecast": 0, "classification": 0}
    steps, max_norm = 0, 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = _move(batch, device)
            output = model(batch["x"], batch["mask"])
            losses = multitask_loss(output, batch, task=task, **weights)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite training or validation loss.")
            has_supervision = losses["n_forecast"] > 0 or losses["n_classification"] > 0
            if training and has_supervision:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                max_norm = max(max_norm, float(norm))
                optimizer.step()
                steps += 1
            for name in sums:
                sums[name] += float(losses[f"{name}_sum"].detach())
                counts[name] += int(losses[f"n_{name}"])
    means = {name: sums[name]/counts[name] if counts[name] else 0.0 for name in sums}
    means["total"] = weights["forecast_weight"]*means["forecast"] + weights["classification_weight"]*means["classification"]
    means.update({f"n_{name}": counts[name] for name in counts})
    means.update(optimizer_steps=steps, max_unclipped_gradient_norm=max_norm)
    return means


def train_model(data_path: str | Path, config_path: str | Path, output_dir: str | Path,
                device: str = "cpu", split_path: str | Path | None = None,
                synthetic_demo: bool = False) -> dict[str, Any]:
    from .model import PSDformer, PSDformerConfig
    config = read_config(config_path)
    # Preserve explicit synthetic provenance when a generated NPZ is trained
    # separately through the CLI, not just through the one-command demo.
    metadata_path = Path(data_path).with_suffix(".json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            metadata = {}
        if isinstance(metadata, dict) and metadata.get("kind") == "synthetic_smoke_test_only":
            synthetic_demo = True
    training = config["training"]
    task = canonical_task(training.get("task", "forecast"))
    training["task"] = task
    seed = int(training.get("seed", 42))
    epochs, batch_size = int(training.get("epochs", 50)), int(training.get("batch_size", 32))
    threads = int(training.get("cpu_threads", 2))
    lr, decay, clip = float(training.get("learning_rate", .001)), float(training.get("weight_decay", 0)), float(training.get("gradient_clip", 1.0))
    if epochs < 1 or batch_size < 1 or threads < 1 or not np.isfinite([lr, decay, clip]).all() or lr <= 0 or decay < 0 or clip <= 0:
        raise ValueError("Require positive epochs/batch_size/threads/lr/gradient_clip and nonnegative weight_decay.")
    selection = training.get("selection_metric", "val_total_loss")
    allowed = {"val_total_loss", "val_forecast_loss", "val_classification_loss", "val_macro_f1"}
    if selection not in allowed or (task == "forecast" and selection in {"val_classification_loss", "val_macro_f1"}) or (task == "classification" and selection == "val_forecast_loss"):
        raise ValueError("selection_metric must be a validation metric available for this task.")
    weights = {"forecast_weight": float(training.get("forecast_weight", 1.0)) if task in ("forecast", "joint") else 0.0,
               "classification_weight": float(training.get("classification_weight", 1.0)) if task in ("classification", "joint") else 0.0}
    if not np.isfinite(list(weights.values())).all() or any(v < 0 for v in weights.values()) or not any(v > 0 for v in weights.values()):
        raise ValueError("Loss weights must be finite/nonnegative and one active task weight must be positive.")
    # Requiring positive weights for both named tasks prevents claiming an untrained head.
    if task == "joint" and any(v <= 0 for v in weights.values()):
        raise ValueError("Joint training requires positive weights for both tasks; otherwise select a single task.")
    training.update(epochs=epochs, batch_size=batch_size, cpu_threads=threads, learning_rate=lr,
        weight_decay=decay, gradient_clip=clip, selection_metric=selection, seed=seed, **weights)
    training.setdefault("val_fraction", .15)
    training.setdefault("test_fraction", .15)
    seed_everything(seed)
    torch.set_num_threads(threads)
    device_obj = resolve_device(device)
    model_config = PSDformerConfig(**config["model"])
    if task in ("classification", "joint") and not model_config.num_classes:
        raise ValueError("Classification/joint training requires a positive num_classes.")
    model = PSDformer(model_config).to(device_obj)
    # Persist resolved dataclass defaults, not just user-provided partial fields.
    config["model"] = asdict(model_config)
    data = load_model_data(data_path, config["model"])
    split = validate_split(json.loads(Path(split_path).read_text(encoding="utf-8")), data["subject_ids"]) if split_path else split_subjects(
        data["subject_ids"], seed, float(training.get("val_fraction", .15)), float(training.get("test_fraction", .15)))
    indices = {part: partition_indices(data["subject_ids"], split, part) for part in ("train", "val", "test")}
    for part in ("train", "val"):
        if task in ("forecast", "joint") and not np.isfinite(data["target"][indices[part]]).any():
            raise ValueError(f"{part} partition needs observed continuous targets for {task} training.")
        if task in ("classification", "joint") and not np.any(data["y"][indices[part]] >= 0):
            raise ValueError(f"{part} partition needs true class labels for {task} training.")
    x_normalizer = StandardNormalizer.fit(data["x"][indices["train"]], data["mask"][indices["train"]])
    target_normalizer = StandardNormalizer.fit(data["target"][indices["train"]]) if task in ("forecast", "joint") else StandardNormalizer.identity(model_config.output_dim)
    generator = torch.Generator().manual_seed(seed)
    loaders = {part: DataLoader(WindowDataset(data, indices[part], x_normalizer, target_normalizer), batch_size=batch_size,
        shuffle=part == "train", num_workers=0, generator=generator if part == "train" else None) for part in ("train", "val")}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"split.json", split)
    write_json(output/"config.json", config)
    fingerprint = dataset_fingerprint(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    history, global_step = [], 0
    best_score = -float("inf") if selection == "val_macro_f1" else float("inf")
    best_epoch = 0
    for epoch in range(1, epochs+1):
        train_stats = _epoch(model, loaders["train"], device_obj, task, weights, optimizer, clip)
        global_step += train_stats["optimizer_steps"]
        validation = _epoch(model, loaders["val"], device_obj, task, weights)
        if selection == "val_macro_f1":
            prediction = collect_predictions(model, loaders["val"], device_obj)
            validation["macro_f1"] = classification_metrics(prediction["y"], prediction["probs"])["macro_f1"]
            score = validation["macro_f1"]
        else:
            score = validation[{"val_total_loss": "total", "val_forecast_loss": "forecast", "val_classification_loss": "classification"}[selection]]
        if not math.isfinite(score):
            raise FloatingPointError("Checkpoint selection requires a finite validation metric.")
        improved = score > best_score if selection == "val_macro_f1" else score < best_score
        if improved:
            best_score, best_epoch = score, epoch
        row = {"epoch": epoch, "global_step": global_step, "train": train_stats, "validation": validation,
               "selection_metric": selection, "selection_value": score, "selected": improved}
        history.append(row)
        checkpoint = {"format_version": 1, "architecture": "PSDformer", "model_state": model.state_dict(),
            "model_config": asdict(model_config), "config": config, "task": task,
            "normalizers": {"x": x_normalizer.to_dict(), "target": target_normalizer.to_dict()},
            "normalizer_fit_partition": "train", "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "global_step": global_step, "best_epoch": best_epoch,
            "selection_metric": selection, "best_validation_value": best_score, "split": split,
            "seed": seed, "dataset_fingerprint": fingerprint, "synthetic_demo": bool(synthetic_demo),
            "history": history, "torch_rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state(),
            "torch_version": str(torch.__version__)}
        if improved:
            torch.save(checkpoint, output/"best.pt")
        torch.save(checkpoint, output/"last.pt")
        write_json(output/"history.json", history)
        print(json.dumps(json_safe({"epoch": epoch, "train_loss": train_stats["total"], "validation_loss": validation["total"],
            "selection_metric": selection, "selection_value": score}), allow_nan=False), flush=True)
    result = {"checkpoint": str(output/"best.pt"), "last_checkpoint": str(output/"last.pt"),
        "best_epoch": best_epoch, "selection_metric": selection, "best_validation_value": best_score,
        "global_step": global_step, "split": split, "task": task, "synthetic_demo": bool(synthetic_demo)}
    write_json(output/"training_summary.json", result)
    return result
