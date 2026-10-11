"""Validated NPZ windows, participant-disjoint splits, and explicit CSV conversion."""
from __future__ import annotations
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import StandardNormalizer


def load_npz(path: str | Path, input_len: int | None = None, input_dim: int | None = None,
             pred_len: int | None = None, output_dim: int | None = None,
             num_classes: int | None = None) -> dict[str, np.ndarray]:
    """Read safe arrays: x[N,T,D], subject_ids[N], optional target[N,K,O], y[N].

    mask has x's shape, True=observed. Nonfinite x is missing, replaced by zero.
    Missing target entries are NaN; missing classification labels are -1.
    Object arrays are forbidden. All dimensions remain explicit (no squeezing).
    """
    with np.load(path, allow_pickle=False) as archive:
        if not {"x", "subject_ids"}.issubset(archive.files):
            raise ValueError("NPZ requires x and subject_ids arrays.")
        data = {key: archive[key] for key in archive.files}
    x = np.asarray(data["x"], dtype=np.float32)
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError(f"x must have nonempty shape [N,T,D]; got {x.shape}.")
    n, t, d = x.shape
    if (input_len is not None and t != input_len) or (input_dim is not None and d != input_dim):
        raise ValueError(f"x shape {x.shape} disagrees with configured input_len/input_dim.")
    ids = np.asarray(data["subject_ids"])
    if ids.shape != (n,) or ids.dtype.kind not in "USiu":
        raise ValueError("subject_ids must be [N] strings or integers, never object dtype.")
    ids = ids.astype(str)
    if np.any(np.char.str_len(np.char.strip(ids)) == 0):
        raise ValueError("Empty subject IDs are not allowed.")
    raw_mask = np.asarray(data.get("mask", np.isfinite(x)))
    if raw_mask.shape != x.shape or raw_mask.dtype.kind not in "bui" or not np.isin(raw_mask, [0, 1]).all():
        raise ValueError("mask must match x and contain boolean or integer 0/1 values.")
    mask = raw_mask.astype(bool) & np.isfinite(x)
    raw_y = np.asarray(data.get("y", np.full(n, -1)))
    if raw_y.dtype.kind not in "iuf" or raw_y.shape != (n,) or not np.isfinite(raw_y).all() or not (raw_y == np.floor(raw_y)).all():
        raise ValueError("y must be [N] integer class IDs; use -1 for missing labels.")
    y = raw_y.astype(np.int64)
    if np.any(y < -1) or (num_classes is not None and num_classes > 0 and np.any(y >= num_classes)):
        raise ValueError("y must be -1 or in [0,num_classes).")
    target = np.asarray(data.get("target", np.full((n, pred_len or 1, output_dim or 1), np.nan)), np.float32)
    if target.ndim != 3 or target.shape[0] != n or min(target.shape) < 1:
        raise ValueError("target must have explicit shape [N,K,O]; use NaN for missing values.")
    if (pred_len is not None and target.shape[1] != pred_len) or (output_dim is not None and target.shape[2] != output_dim):
        raise ValueError("target dimensions disagree with configured pred_len/output_dim.")
    result = {"x": np.where(mask, x, 0).astype(np.float32), "mask": mask, "y": y,
              "target": np.where(np.isfinite(target), target, np.nan), "subject_ids": ids}
    for name in ("session_ids", "timestamp"):
        if name in data:
            if data[name].shape != (n,) or data[name].dtype.kind not in "USiuf":
                raise ValueError(f"{name} must be a non-object [N] array.")
            result[name] = np.asarray(data[name])
    return result


def validate_split(split: dict[str, list[str]], subject_ids: np.ndarray) -> dict[str, list[str]]:
    if not isinstance(split, dict) or set(split) != {"train", "val", "test"}:
        raise ValueError("Split must contain exactly train, val, test subject-ID lists.")
    if any(not isinstance(values, list) for values in split.values()):
        raise ValueError("Each split value must be a subject-ID list.")
    normalized = {part: [str(s) for s in values] for part, values in split.items()}
    for part, values in normalized.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{part} must contain unique, nonempty subject IDs.")
    groups = [set(normalized[p]) for p in ("train", "val", "test")]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Subject leakage: train/val/test overlap.")
    if set.union(*groups) != set(np.asarray(subject_ids).astype(str)):
        raise ValueError("Split must assign every data subject exactly once, with no unknown IDs.")
    return normalized


def split_subjects(subject_ids: np.ndarray, seed: int = 42, val_fraction: float = .15,
                   test_fraction: float = .15) -> dict[str, list[str]]:
    subjects = np.unique(np.asarray(subject_ids).astype(str))
    if len(subjects) < 3:
        raise ValueError("At least 3 subjects are needed for separate train, validation and test sets.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1.")
    subjects = np.random.default_rng(seed).permutation(subjects)
    nv, nt = max(1, round(len(subjects)*val_fraction)), max(1, round(len(subjects)*test_fraction))
    if nv + nt >= len(subjects):
        raise ValueError("Requested fractions leave no training subjects.")
    return validate_split({"train": subjects[nv+nt:].tolist(), "val": subjects[:nv].tolist(),
                           "test": subjects[nv:nv+nt].tolist()}, subject_ids)


def partition_indices(subject_ids: np.ndarray, split: dict[str, list[str]], partition: str) -> np.ndarray:
    if partition not in split:
        raise ValueError(f"Unknown partition: {partition}")
    return np.flatnonzero(np.isin(np.asarray(subject_ids).astype(str), split[partition]))


def dataset_fingerprint(data: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(data):
        values = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(str(values.dtype).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


class WindowDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray], indices: np.ndarray,
                 x_normalizer: StandardNormalizer, target_normalizer: StandardNormalizer | None = None):
        self.data, self.indices = data, np.asarray(indices, np.int64)
        self.x = x_normalizer.transform(data["x"][self.indices], data["mask"][self.indices])
        target_normalizer = target_normalizer or StandardNormalizer.identity(data["target"].shape[-1])
        self.target = target_normalizer.transform(data["target"][self.indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        original = self.indices[item]
        return {"x": torch.from_numpy(self.x[item]), "mask": torch.from_numpy(self.data["mask"][original]),
                "target": torch.from_numpy(self.target[item]), "y": torch.tensor(int(self.data["y"][original])),
                "index": int(original), "subject_id": str(self.data["subject_ids"][original])}


def make_demo_data(path: str | Path, subjects: int = 6, windows_per_subject: int = 5,
                   input_len: int = 24, pred_len: int = 3, input_dim: int = 5,
                   output_dim: int = 1, num_classes: int | None = 3, seed: int = 42) -> Path:
    """Synthetic trajectories and synthetic regime labels, never real stress truth."""
    if subjects < 3 or min(windows_per_subject, input_len, pred_len, input_dim, output_dim) < 1:
        raise ValueError("Demo requires >=3 subjects and positive data dimensions.")
    if num_classes is not None and (isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 0):
        raise ValueError("num_classes must be None or a nonnegative integer.")
    rng = np.random.default_rng(seed)
    xs, masks, targets, labels, ids = [], [], [], [], []
    time = np.arange(input_len + pred_len, dtype=np.float32)
    for subject in range(subjects):
        offsets = rng.normal(0, .1, input_dim)
        for window in range(windows_per_subject):
            regime = window % (num_classes or 1)
            frequency, phase = .08 + .015*regime, rng.uniform(0, 2*np.pi)
            latent = np.sin(frequency*time+phase) + .012*time + rng.normal(0, .015, len(time))
            full = np.stack([(1+.15*c)*latent + .2*np.cos(.04*(c+1)*time+phase)+offsets[c] for c in range(input_dim)], -1)
            full += rng.normal(0, .03, full.shape)
            x = full[:input_len].astype(np.float32)
            mask = rng.random(x.shape) > .015
            target = np.stack([full[input_len:, c % input_dim] for c in range(output_dim)], -1).astype(np.float32)
            if window == 1:
                target[-1, 0] = np.nan
            xs.append(np.where(mask, x, 0)); masks.append(mask); targets.append(target)
            labels.append(regime if num_classes else -1); ids.append(f"synthetic_subject_{subject:02d}")
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Demo output must have .npz extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.stack(xs), mask=np.stack(masks), target=np.stack(targets),
                        y=np.asarray(labels, np.int64), subject_ids=np.asarray(ids))
    path.with_suffix(".json").write_text(json.dumps({"kind": "synthetic_smoke_test_only", "seed": seed,
        "subjects": subjects, "windows_per_subject": windows_per_subject,
        "warning": "Simulated trajectories and regime labels; not real stress annotations, paper data or scientific replication."}, indent=2), encoding="utf-8")
    return path


def _number(value: str | None) -> float:
    if value is None or not value.strip():
        return float("nan")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Expected numeric feature/target value, got {value!r}.") from exc
    return result if np.isfinite(result) else float("nan")


def _time_key(value: str) -> tuple[str, float]:
    if not value or not value.strip():
        raise ValueError("CSV timestamps must not be empty.")
    try:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("CSV timestamps must be finite.")
        return "numeric", numeric
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Timestamp {value!r} must be numeric or ISO-8601.") from exc
        # Naive times have an explicit uniform convention rather than OS local TZ.
        from datetime import timezone
        kind = "aware_datetime" if parsed.tzinfo else "naive_datetime"
        return kind, parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()


def prepare_csv(csv_path: str | Path, output_path: str | Path, feature_columns: list[str],
                target_columns: list[str], subject_column: str, session_column: str,
                time_column: str, input_len: int = 24, pred_len: int = 3,
                stride: int = 1, label_column: str | None = None,
                num_classes: int | None = None) -> Path:
    """Window within (subject,session), sorting time; no resampling or label invention.

    x rows [start,start+T); target rows [start+T,start+T+K). Classification
    labels are read only at the LAST INPUT row. Every named target is explicit.
    Irregular sampling is preserved; horizons count rows, not seconds.
    """
    if min(input_len, pred_len, stride) < 1 or not feature_columns:
        raise ValueError("Positive lengths/stride and explicit feature columns are required.")
    if not target_columns and label_column is None:
        raise ValueError("Specify target columns, a label column, or both.")
    if len(feature_columns) != len(set(feature_columns)) or len(target_columns) != len(set(target_columns)):
        raise ValueError("Feature and target columns must not contain duplicates.")
    needed = set(feature_columns + target_columns + [subject_column, session_column, time_column])
    if label_column:
        needed.add(label_column)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"CSV is missing required columns: {sorted(needed-set(reader.fieldnames or []))}.")
        for row in reader:
            subject, session = row[subject_column].strip(), row[session_column].strip()
            if not subject or not session:
                raise ValueError("Every CSV row requires nonempty subject and session IDs.")
            row["__time_key"] = _time_key(row[time_column])
            groups.setdefault((subject, session), []).append(row)
    xs, targets, labels, ids, sessions, timestamps = [], [], [], [], [], []
    skipped = 0
    for (subject, session), rows in sorted(groups.items()):
        kinds = {row["__time_key"][0] for row in rows}
        if len(kinds) != 1:
            raise ValueError("Use a consistent timestamp representation within each session.")
        rows.sort(key=lambda r: r["__time_key"][1])
        keys = [row["__time_key"][1] for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate timestamps in subject/session {subject}/{session}.")
        values = np.asarray([[_number(r[c]) for c in feature_columns] for r in rows], np.float32)
        regression = np.asarray([[_number(r[c]) for c in target_columns] for r in rows], np.float32)
        # Classification-only does not require unused future rows.
        future_rows = pred_len if target_columns else 0
        made = 0
        for start in range(0, len(rows)-input_len-future_rows+1, stride):
            end = start+input_len
            xs.append(values[start:end])
            if target_columns:
                targets.append(regression[end:end+pred_len])
            else:
                targets.append(np.full((pred_len, 1), np.nan, np.float32))
            label = -1
            if label_column:
                raw = _number(rows[end-1][label_column])
                if np.isfinite(raw):
                    if raw != int(raw) or raw < -1 or (num_classes is not None and raw >= num_classes):
                        raise ValueError("CSV labels must be integer class IDs in [0,num_classes), or -1/blank.")
                    label = int(raw)
            labels.append(label); ids.append(subject); sessions.append(session)
            timestamps.append(rows[end-1][time_column]); made += 1
        skipped += int(made == 0)
    if not xs:
        raise ValueError("No complete windows: check input_len, pred_len, and session lengths.")
    x = np.stack(xs)
    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("CSV conversion output must have .npz extension.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=np.where(np.isfinite(x), x, 0), mask=np.isfinite(x),
        target=np.stack(targets), y=np.asarray(labels, np.int64), subject_ids=np.asarray(ids),
        session_ids=np.asarray(sessions), timestamp=np.asarray(timestamps))
    output.with_suffix(".json").write_text(json.dumps({"source": str(csv_path), "feature_columns": feature_columns,
        "target_columns": target_columns, "subject_column": subject_column, "session_column": session_column,
        "time_column": time_column, "label_column": label_column, "label_policy": "last input row",
        "input_len": input_len, "pred_len": pred_len, "stride": stride, "windows": len(xs),
        "short_sessions_skipped": skipped, "sampling_policy": "chronological observed rows; no interpolation/resampling",
        "warning": "Verify feature causality, synchronization, target units and true stress annotation externally."}, indent=2), encoding="utf-8")
    return output
