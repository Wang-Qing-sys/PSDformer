"""Train-partition-only standardization; the final axis contains variables."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StandardNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray | None = None) -> "StandardNormalizer":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 2 or values.size == 0:
            raise ValueError("Normalizer requires nonempty [...,variables] data.")
        observed = np.isfinite(values)
        if mask is not None:
            if np.shape(mask) != values.shape:
                raise ValueError("Normalization mask must match data shape.")
            observed &= np.asarray(mask, dtype=bool)
        axes = tuple(range(values.ndim - 1))
        count = observed.sum(axis=axes)
        if np.any(count == 0):
            raise ValueError("Every normalized variable needs at least one observed training value.")
        mean = np.where(observed, values, 0).sum(axis=axes) / count
        variance = np.where(observed, (values-mean)**2, 0).sum(axis=axes) / count
        scale = np.sqrt(variance)
        scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    @classmethod
    def identity(cls, variables: int) -> "StandardNormalizer":
        return cls(np.zeros(variables, np.float32), np.ones(variables, np.float32))

    def transform(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != len(self.mean):
            raise ValueError("Normalizer variable dimension does not match data.")
        transformed = (values-self.mean) / self.scale
        if mask is not None:
            if np.shape(mask) != values.shape:
                raise ValueError("Normalization mask must match data shape.")
            return np.where(np.asarray(mask, bool) & np.isfinite(values), transformed, 0).astype(np.float32)
        return np.where(np.isfinite(values), transformed, np.nan).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != len(self.mean):
            raise ValueError("Normalizer variable dimension does not match data.")
        return values*self.scale + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "method": "standard_population_std"}

    @classmethod
    def from_dict(cls, data: dict) -> "StandardNormalizer":
        mean, scale = np.asarray(data["mean"], np.float32), np.asarray(data["scale"], np.float32)
        if mean.ndim != 1 or not len(mean) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid saved normalization statistics.")
        return cls(mean, scale)
