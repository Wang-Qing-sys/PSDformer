"""PSDformer reproduction from the uploaded manuscript.

This implementation follows the architectural statements and equations that are
explicitly available in the manuscript:
  * MDSW: per-dimension non-overlapping segmentation + linear projection + 2-D PE.
  * TDE: shared cross-time MHSA followed by router-based cross-dimension MHSA.
  * EDM: hierarchical segment merging encoder + multi-scale decoder.
  * DGC: densely connected convolution blocks with 1x3 convolution and 1xq pooling.
  * Fig. 9: two branches, cross-modal matching, feature regularization, learned
    feature fusion, and output-consistency regularization.

The manuscript does NOT specify several implementation hyperparameters or exact
loss formulas (e.g., d_model, number of heads/layers, router count, feature-
regularization formula, cross-modal matching loss, fusion-weight training rule).
Those pieces are therefore implemented with standard, configurable choices and
are marked in code as engineering completions rather than claimed paper facts.

Tested with PyTorch >= 2.1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# -----------------------------------------------------------------------------
# Reproducibility and basic utilities
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardScaler:
    """Numpy z-score scaler fitted on training data only."""

    def __init__(self, eps: float = 1e-8):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = np.nanmean(x, axis=0, keepdims=True)
        self.std_ = np.nanstd(x, axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted first.")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted first.")
        return x * self.std_ + self.mean_


def moving_average_filter(x: np.ndarray, window: int = 1) -> np.ndarray:
    """Optional simple filtering hook for Fig. 8 preprocessing.

    The paper shows a filtering stage but does not specify its exact filter, so
    window=1 (identity) is the default. Increase only if this matches your data
    acquisition pipeline.
    """
    if window <= 1:
        return x
    out = np.empty_like(x, dtype=np.float64)
    for d in range(x.shape[1]):
        s = pd.Series(x[:, d])
        out[:, d] = s.rolling(window, center=True, min_periods=1).mean().to_numpy()
    return out


def fill_missing(x: np.ndarray) -> np.ndarray:
    """Linear interpolation + edge fill, matching the generic Fig. 8 step."""
    df = pd.DataFrame(x)
    df = df.interpolate(limit_direction="both").ffill().bfill()
    return df.to_numpy(dtype=np.float64)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class PSDformerConfig:
    input_dim: int
    target_dim: int = 1
    task: str = "regression"  # "regression" or "classification"
    num_classes: int = 3

    # Paper-stated training default: lookback = 24.
    lookback: int = 24
    horizon: int = 1

    # MDSW / Transformer hyperparameters (not numerically specified by paper).
    segment_len: int = 4
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    tde_layers_part1: int = 2
    edm_levels: int = 3
    router_tokens: int = 1

    # DGC: paper states 2 dense blocks; conv kernel 1x3; pool kernel 1xq.
    dgc_blocks: int = 2
    dgc_layers_per_block: int = 2
    dgc_growth: int = 16
    dgc_pool_q: int = 2

    # Figure-9 regularization/fusion terms. Exact coefficients are not given.
    aux_task_weight: float = 0.25
    alignment_weight: float = 0.05
    feature_reg_weight: float = 0.05
    consistency_weight: float = 0.05

    # Paper-stated optimization defaults.
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    @property
    def max_segments(self) -> int:
        return int(math.ceil(self.lookback / self.segment_len))

    @property
    def output_channels(self) -> int:
        return self.num_classes if self.task == "classification" else self.target_dim

    def validate(self) -> None:
        if self.task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.lookback <= 0 or self.segment_len <= 0:
            raise ValueError("lookback and segment_len must be positive")
        if self.edm_levels <= 0:
            raise ValueError("edm_levels must be positive")
        if self.router_tokens <= 0:
            raise ValueError("router_tokens must be positive")


# -----------------------------------------------------------------------------
# Dataset and data preparation
# -----------------------------------------------------------------------------

class SlidingWindowDataset(Dataset):
    """Multivariate lookback -> multi-step target dataset."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lookback: int,
        horizon: int,
        task: str,
    ):
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of time points")
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.task = task
        self.n = len(x) - lookback - horizon + 1
        if self.n <= 0:
            raise ValueError("Not enough samples for lookback + horizon")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        xs = self.x[idx : idx + self.lookback]
        ys = self.y[idx + self.lookback : idx + self.lookback + self.horizon]
        xt = torch.from_numpy(xs)
        if self.task == "classification":
            # One categorical target per forecast step.
            if ys.ndim > 1 and ys.shape[-1] == 1:
                ys = ys[..., 0]
            yt = torch.as_tensor(ys, dtype=torch.long)
        else:
            ys = np.asarray(ys, dtype=np.float32)
            if ys.ndim == 1:
                ys = ys[:, None]
            yt = torch.from_numpy(ys)
        return xt, yt


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    n = len(x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n_train <= 0 or n_val <= 0 or n_train + n_val >= n:
        raise ValueError("Invalid split ratios for dataset length")
    return (
        (x[:n_train], y[:n_train]),
        (x[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        (x[n_train + n_val :], y[n_train + n_val :]),
    )


def make_balanced_sampler(dataset: SlidingWindowDataset) -> Optional[WeightedRandomSampler]:
    """Optional classification balancing corresponding to Fig. 8.

    Returns None for regression. The paper does not specify a balancing method;
    inverse-frequency weighted sampling is used as a transparent implementation.
    """
    if dataset.task != "classification":
        return None
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y.reshape(-1)[0].item()))
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    inv = {int(c): 1.0 / float(n) for c, n in zip(classes, counts)}
    weights = np.asarray([inv[int(v)] for v in labels], dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# -----------------------------------------------------------------------------
# Positional encoding and MDSW
# -----------------------------------------------------------------------------

def sinusoidal_encoding(length: int, d_model: int, device=None, dtype=None) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype or torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype or torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device, dtype=dtype or torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class MDSW(nn.Module):
    """Multi-dimensional segmentation and embedding (paper Eqs. 1-3).

    Input:  x [B, T, D]
    Output: W [B, D, L, E]
    """

    def __init__(self, cfg: PSDformerConfig):
        super().__init__()
        self.cfg = cfg
        self.segment_len = cfg.segment_len
        self.projection = nn.Linear(cfg.segment_len, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"MDSW expects [B,T,D], got {tuple(x.shape)}")
        b, t, d = x.shape
        if d != self.cfg.input_dim:
            raise ValueError(f"Expected input_dim={self.cfg.input_dim}, got {d}")
        pad = (-t) % self.segment_len
        if pad:
            # Repeat the final observation only to complete the final segment.
            x = torch.cat([x, x[:, -1:, :].expand(b, pad, d)], dim=1)
        t_pad = x.shape[1]
        l = t_pad // self.segment_len

        # [B,T,D] -> [B,D,T] -> [B,D,L,Ls]
        seg = x.transpose(1, 2).unfold(dimension=-1, size=self.segment_len, step=self.segment_len)
        w = self.projection(seg)

        # Fixed 2-D positional encoding: segment-position + dimension-position.
        pe_t = sinusoidal_encoding(l, self.cfg.d_model, device=x.device, dtype=x.dtype)
        pe_d = sinusoidal_encoding(d, self.cfg.d_model, device=x.device, dtype=x.dtype)
        w = w + pe_t[None, None, :, :] + pe_d[None, :, None, :]
        return self.norm(w)


# -----------------------------------------------------------------------------
# TDE: cross-time + router-based cross-dimension dependency extraction
# -----------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TDELayer(nn.Module):
    """Two-stage dependency extraction from paper Eqs. 4-10.

    Stage 1: temporal MHSA shared across all dimensions.
    Stage 2: router aggregation/distribution shared across all segments.
    """

    def __init__(self, cfg: PSDformerConfig):
        super().__init__()
        self.cfg = cfg
        self.temporal_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        # One MHSA module is reused for router aggregate and distribute operations.
        self.dimension_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.router = nn.Parameter(
            torch.randn(cfg.max_segments, cfg.router_tokens, cfg.d_model) * 0.02
        )

        self.temporal_ln1 = nn.LayerNorm(cfg.d_model)
        self.temporal_ln2 = nn.LayerNorm(cfg.d_model)
        self.temporal_ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

        self.dimension_ln1 = nn.LayerNorm(cfg.d_model)
        self.dimension_ln2 = nn.LayerNorm(cfg.d_model)
        self.dimension_ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [B,D,L,E]
        b, d, l, e = f.shape
        if l > self.cfg.max_segments:
            raise ValueError(
                f"TDE received {l} segments, exceeding max_segments={self.cfg.max_segments}. "
                "Increase lookback or change segment_len in the config before model creation."
            )

        # Eq. 4-5: shared cross-temporal attention within each dimension.
        z = f.reshape(b * d, l, e)
        a, _ = self.temporal_attn(z, z, z, need_weights=False)
        z = self.temporal_ln1(z + a)
        z = self.temporal_ln2(z + self.temporal_ffn(z))
        ft = z.reshape(b, d, l, e)

        # Eq. 6-9: router-based cross-dimensional interaction per segment.
        xi = ft.permute(0, 2, 1, 3).reshape(b * l, d, e)  # [B*L,D,E]
        r = self.router[:l]  # [L,R,E]
        r = r.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * l, self.cfg.router_tokens, e)

        # Router queries all dimensions to aggregate information (Eq. 6).
        eps, _ = self.dimension_attn(r, xi, xi, need_weights=False)
        # Dimension tokens query the aggregated router messages (Eq. 7).
        dist, _ = self.dimension_attn(xi, eps, eps, need_weights=False)
        y = self.dimension_ln1(xi + dist)  # Eq. 8
        y = self.dimension_ln2(y + self.dimension_ffn(y))  # Eq. 9
        return y.reshape(b, l, d, e).permute(0, 2, 1, 3).contiguous()


class TDEStack(nn.Module):
    def __init__(self, cfg: PSDformerConfig, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([TDELayer(cfg) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------------------------------------------------------
# Cross-modal matching and feature regularization from Fig. 9
# -----------------------------------------------------------------------------

class CrossModalMatch(nn.Module):
    """Bidirectional MHSA match between aligned/projected token streams.

    Fig. 9 depicts two MHSAs and feature similarity, but the manuscript does not
    provide equations. This is a standard symmetric cross-attention completion.
    """

    def __init__(self, cfg: PSDformerConfig):
        super().__init__()
        self.attn_a = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.attn_b = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln_a = nn.LayerNorm(cfg.d_model)
        self.ln_b = nn.LayerNorm(cfg.d_model)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # [B,D,L,E] -> [B,D*L,E]
        shape = a.shape
        af = a.flatten(1, 2)
        bf = b.flatten(1, 2)
        da, _ = self.attn_a(af, bf, bf, need_weights=False)
        db, _ = self.attn_b(bf, af, af, need_weights=False)
        am = self.ln_a(af + da)
        bm = self.ln_b(bf + db)

        pa = F.normalize(am.mean(dim=1), dim=-1)
        pb = F.normalize(bm.mean(dim=1), dim=-1)
        alignment_loss = (1.0 - (pa * pb).sum(dim=-1)).mean()
        return am.reshape(shape), bm.reshape(shape), alignment_loss


# -----------------------------------------------------------------------------
# EDM: hierarchical encoder-decoder with segment merging
# -----------------------------------------------------------------------------

class SegmentMerge(nn.Module):
    """Merge two adjacent temporal segments by concatenation + learned projection."""

    def __init__(self, d_model: int):
        super().__init__()
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, l, e = x.shape
        if l % 2:
            x = torch.cat([x, x[:, :, -1:, :]], dim=2)
            l = l + 1
        a = x[:, :, 0:l:2, :]
        c = x[:, :, 1:l:2, :]
        return self.norm(self.merge(torch.cat([a, c], dim=-1)))


class EDMDecoderLayer(nn.Module):
    """Decoder layer matching the paper's Eqs. 14-18 at implementation level."""

    def __init__(self, cfg: PSDformerConfig):
        super().__init__()
        self.self_tde = TDEStack(cfg, depth=1)
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, dec: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        dec = self.self_tde(dec)
        b, d, lq, e = dec.shape
        lk = enc.shape[2]
        q = dec.reshape(b * d, lq, e)
        kv = enc.reshape(b * d, lk, e)
        ca, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q = self.ln1(q + ca)
        q = self.ln2(q + self.ffn(q))
        return q.reshape(b, d, lq, e)


class EDM(nn.Module):
    """Hierarchical multi-scale encoder-decoder from paper Eqs. 11-18."""

    def __init__(self, cfg: PSDformerConfig):
        super().__init__()
        self.cfg = cfg
        self.levels = cfg.edm_levels
        self.encoder_tde = nn.ModuleList([TDEStack(cfg, depth=1) for _ in range(self.levels)])
        self.mergers = nn.ModuleList([SegmentMerge(cfg.d_model) for _ in range(self.levels - 1)])
        self.decoder_layers = nn.ModuleList([EDMDecoderLayer(cfg) for _ in range(self.levels)])
        self.decoder_pos = nn.Parameter(
            torch.randn(self.levels, cfg.input_dim, cfg.max_segments, cfg.d_model) * 0.02
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)

    @staticmethod
    def _resize_segments(x: torch.Tensor, target_l: int) -> torch.Tensor:
        if x.shape[2] == target_l:
            return x
        # Linear interpolation along segment axis, applied independently to D and E.
        b, d, l, e = x.shape
        y = x.permute(0, 1, 3, 2).reshape(b * d, e, l)
        y = F.interpolate(y, size=target_l, mode="linear", align_corners=False)
        return y.reshape(b, d, e, target_l).permute(0, 1, 3, 2).contiguous()

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # Encoder: level 1 = theta; later levels pairwise merge then TDE.
        enc: List[torch.Tensor] = []
        x = theta
        for level in range(self.levels):
            if level > 0:
                x = self.mergers[level - 1](x)
            x = self.encoder_tde[level](x)
            enc.append(x)

        # Decoder: coarse -> fine. Initial decoder state is learnable positional embedding.
        dec_outputs: List[torch.Tensor] = [None] * self.levels  # type: ignore
        dec: Optional[torch.Tensor] = None
        for rev_idx, level in enumerate(range(self.levels - 1, -1, -1)):
            target_l = enc[level].shape[2]
            if dec is None:
                pos = self.decoder_pos[level, :, :target_l, :]
                dec = pos.unsqueeze(0).expand(theta.shape[0], -1, -1, -1)
            else:
                dec = self._resize_segments(dec, target_l)
                pos = self.decoder_pos[level, :, :target_l, :]
                dec = dec + pos.unsqueeze(0)
            dec = self.decoder_layers[rev_idx](dec, enc[level])
            dec_outputs[level] = dec

        # Paper states that projected decoder outputs from all scales are summed.
        finest_l = enc[0].shape[2]
        summed = torch.zeros_like(enc[0])
        for item in dec_outputs:
            summed = summed + self._resize_segments(item, finest_l)
        return self.out_norm(summed)


# -----------------------------------------------------------------------------
# DGC: densely connected convolution module
# -----------------------------------------------------------------------------

class DenseConvLayer(nn.Module):
    def __init__(self, in_channels: int, growth: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth, kernel_size=(1, 3), padding=(0, 1))
        self.norm = nn.BatchNorm2d(growth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new = F.gelu(self.norm(self.conv(x)))
        return torch.cat([x, new], dim=1)


class DenseConvBlock(nn.Module):
    def __init__(self, in_channels: int, growth: int, layers: int):
        super().__init__()
        mods = []
        c = in_channels
        for _ in range(layers):
            mods.append(DenseConvLayer(c, growth))
            c += growth
        self.layers = nn.ModuleList(mods)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DGC(nn.Module):
    """Dense graph-convolution-inspired feature extractor from paper Eq. 19/Fig. 13.

    The manuscript specifies dense connectivity, a 1x3 convolution and 1xq max
    pooling, but does not provide an adjacency operator. We therefore implement
    the explicitly described dense convolutional transformation on the D x L
    token grid rather than inventing an undocumented graph adjacency matrix.
    """

    def __init__(self, cfg: PSDformerConfig):
        super().__init__()
        blocks = []
        transitions = []
        c = cfg.d_model
        for bi in range(cfg.dgc_blocks):
            block = DenseConvBlock(c, cfg.dgc_growth, cfg.dgc_layers_per_block)
            blocks.append(block)
            c = block.out_channels
            transitions.append(
                nn.Sequential(
                    nn.Conv2d(c, cfg.d_model, kernel_size=(1, 3), padding=(0, 1)),
                    nn.GELU(),
                    nn.MaxPool2d(
                        kernel_size=(1, cfg.dgc_pool_q),
                        stride=(1, cfg.dgc_pool_q),
                        ceil_mode=True,
                    ),
                )
            )
            c = cfg.d_model
        self.blocks = nn.ModuleList(blocks)
        self.transitions = nn.ModuleList(transitions)
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,D,L,E] -> [B,E,D,L]
        y = x.permute(0, 3, 1, 2).contiguous()
        for block, trans in zip(self.blocks, self.transitions):
            y = block(y)
            y = trans(y)
        y = F.adaptive_avg_pool2d(y, output_size=(1, 1)).flatten(1)
        return self.proj(y)


# -----------------------------------------------------------------------------
# Full PSDformer
# -----------------------------------------------------------------------------

class ForecastHead(nn.Module):
    def __init__(self, cfg: PSDformerConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.horizon * cfg.output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y.view(x.shape[0], self.cfg.horizon, self.cfg.output_channels)


class PSDformer(nn.Module):
    """End-to-end two-part PSDformer based on Fig. 9 and Sec. III."""

    def __init__(self, cfg: PSDformerConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.mdsw = MDSW(cfg)

        # Part-II "Projected Tokens" path in Fig. 9.
        self.part2_projection = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        self.cross_modal_match = CrossModalMatch(cfg)

        # Part I: MDSW -> TDE -> Head.
        self.part1_tde = TDEStack(cfg, depth=cfg.tde_layers_part1)
        self.part1_norm = nn.LayerNorm(cfg.d_model)
        self.head1 = ForecastHead(cfg)

        # Part II: EDM -> DGC -> Head.
        self.edm = EDM(cfg)
        self.dgc = DGC(cfg)
        self.head2 = ForecastHead(cfg)

        # Learned fusion weights; initialize equally. These can be loaded/frozen
        # if external pretrained weights are available.
        self.fusion_logits = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        aligned = self.mdsw(x)
        projected = self.part2_projection(aligned)
        aligned, projected, alignment_loss = self.cross_modal_match(aligned, projected)

        # Part I.
        h1_tokens = self.part1_tde(aligned)
        h1_tokens = self.part1_norm(h1_tokens + aligned)
        h1 = h1_tokens.mean(dim=(1, 2))
        pred1 = self.head1(h1)

        # Part II.
        h2_tokens = self.edm(projected)
        h2 = self.dgc(h2_tokens)
        pred2 = self.head2(h2)

        # Feature regularization from Fig. 9: align normalized branch features.
        h1n = F.normalize(h1, dim=-1)
        h2n = F.normalize(h2, dim=-1)
        feature_reg_loss = F.mse_loss(h1n, h2n)

        weights = torch.softmax(self.fusion_logits, dim=0)
        pred = weights[0] * pred1 + weights[1] * pred2

        if self.cfg.task == "classification":
            p1 = torch.softmax(pred1, dim=-1)
            p2 = torch.softmax(pred2, dim=-1)
            consistency_loss = F.mse_loss(p1, p2)
        else:
            consistency_loss = F.mse_loss(pred1, pred2)

        return {
            "prediction": pred,
            "part1_prediction": pred1,
            "part2_prediction": pred2,
            "fusion_weights": weights,
            "alignment_loss": alignment_loss,
            "feature_reg_loss": feature_reg_loss,
            "consistency_loss": consistency_loss,
            "part1_feature": h1,
            "part2_feature": h2,
        }


def task_loss(cfg: PSDformerConfig, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if cfg.task == "classification":
        # pred [B,H,C], target [B,H]
        return F.cross_entropy(pred.reshape(-1, cfg.num_classes), target.reshape(-1))
    return F.mse_loss(pred, target)


def compute_total_loss(
    cfg: PSDformerConfig,
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    fused = task_loss(cfg, output["prediction"], target)
    aux1 = task_loss(cfg, output["part1_prediction"], target)
    aux2 = task_loss(cfg, output["part2_prediction"], target)
    aux = 0.5 * (aux1 + aux2)
    total = (
        fused
        + cfg.aux_task_weight * aux
        + cfg.alignment_weight * output["alignment_loss"]
        + cfg.feature_reg_weight * output["feature_reg_loss"]
        + cfg.consistency_weight * output["consistency_loss"]
    )
    return {
        "total": total,
        "task": fused,
        "aux_task": aux,
        "alignment": output["alignment_loss"],
        "feature_reg": output["feature_reg_loss"],
        "consistency": output["consistency_loss"],
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    err = yt - yp
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    # Paper prints RSE with sqrt(sum error^2/(n-2)); retain that convention.
    n = max(len(yt), 3)
    rse = float(np.sqrt(np.sum(err ** 2) / (n - 2)))
    denom = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    return {"MSE": mse, "MAE": mae, "RSE": rse, "R2": r2}


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1).astype(np.int64)
    yp = logits.argmax(axis=-1).reshape(-1).astype(np.int64)
    classes = np.unique(np.concatenate([yt, yp]))
    acc = float((yt == yp).mean())
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return {
        "Accuracy": acc,
        "Precision_macro": float(np.mean(precisions)),
        "Recall_macro": float(np.mean(recalls)),
        "F1_macro": float(np.mean(f1s)),
    }


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: PSDformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    sums: Dict[str, float] = {}
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        losses = compute_total_loss(model.cfg, out, y)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs = x.shape[0]
        count += bs
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().item()) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(
    model: PSDformer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        losses = compute_total_loss(model.cfg, out, y)
        bs = x.shape[0]
        count += bs
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach().item()) * bs
        preds.append(out["prediction"].cpu().numpy())
        targets.append(y.cpu().numpy())
    loss_avg = {k: v / max(count, 1) for k, v in sums.items()}
    yp = np.concatenate(preds, axis=0)
    yt = np.concatenate(targets, axis=0)
    if model.cfg.task == "classification":
        metric = classification_metrics(yt, yp)
    else:
        metric = regression_metrics(yt, yp)
    return loss_avg, metric


def fit_model(
    model: PSDformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    checkpoint: Optional[Path] = None,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model.cfg.learning_rate,
        weight_decay=model.cfg.weight_decay,
    )
    model.to(device)
    best = float("inf")
    history = {"train_total": [], "val_total": []}
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va, met = evaluate(model, val_loader, device)
        history["train_total"].append(tr["total"])
        history["val_total"].append(va["total"])
        metrics_str = ", ".join(f"{k}={v:.5f}" for k, v in met.items())
        print(
            f"Epoch {epoch:03d} | train={tr['total']:.6f} | "
            f"val={va['total']:.6f} | {metrics_str}"
        )
        if va["total"] < best:
            best = va["total"]
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.state_dict(), "config": asdict(model.cfg), "epoch": epoch},
                    checkpoint,
                )
    return history


# -----------------------------------------------------------------------------
# CSV workflow
# -----------------------------------------------------------------------------

def load_csv_arrays(
    csv_path: str,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    task: str,
    filter_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, int]]]:
    df = pd.read_csv(csv_path)
    missing = [c for c in [*feature_cols, *target_cols] if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing columns: {missing}")

    x = fill_missing(df[list(feature_cols)].to_numpy(dtype=np.float64))
    x = moving_average_filter(x, filter_window)

    label_map = None
    if task == "classification":
        if len(target_cols) != 1:
            raise ValueError("Classification mode expects exactly one target column")
        s = df[target_cols[0]]
        if not np.issubdtype(s.dtype, np.number):
            cats = pd.Categorical(s)
            y = cats.codes.astype(np.int64)
            label_map = {str(cat): int(i) for i, cat in enumerate(cats.categories)}
        else:
            y = s.to_numpy(dtype=np.int64)
        y = y[:, None]
    else:
        y = fill_missing(df[list(target_cols)].to_numpy(dtype=np.float64))
    return x, y, label_map


def build_loaders_from_csv(
    cfg: PSDformerConfig,
    csv_path: str,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    filter_window: int = 1,
    balanced_sampling: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    x, y, label_map = load_csv_arrays(
        csv_path, feature_cols, target_cols, cfg.task, filter_window=filter_window
    )
    (xtr, ytr), (xv, yv), (xte, yte) = chronological_split(x, y)

    x_scaler = StandardScaler().fit(xtr)
    xtr, xv, xte = map(x_scaler.transform, (xtr, xv, xte))

    y_scaler = None
    if cfg.task == "regression":
        y_scaler = StandardScaler().fit(ytr)
        ytr, yv, yte = map(y_scaler.transform, (ytr, yv, yte))

    train_ds = SlidingWindowDataset(xtr, ytr, cfg.lookback, cfg.horizon, cfg.task)
    val_ds = SlidingWindowDataset(xv, yv, cfg.lookback, cfg.horizon, cfg.task)
    test_ds = SlidingWindowDataset(xte, yte, cfg.lookback, cfg.horizon, cfg.task)

    sampler = make_balanced_sampler(train_ds) if balanced_sampling else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    meta: Dict[str, object] = {
        "feature_cols": list(feature_cols),
        "target_cols": list(target_cols),
        "label_map": label_map,
        "x_mean": None if x_scaler.mean_ is None else x_scaler.mean_.tolist(),
        "x_std": None if x_scaler.std_ is None else x_scaler.std_.tolist(),
        "y_mean": None if y_scaler is None or y_scaler.mean_ is None else y_scaler.mean_.tolist(),
        "y_std": None if y_scaler is None or y_scaler.std_ is None else y_scaler.std_.tolist(),
    }
    return train_loader, val_loader, test_loader, meta


# -----------------------------------------------------------------------------
# Demo / CLI
# -----------------------------------------------------------------------------

def make_synthetic_demo(task: str = "regression", n: int = 1200, d: int = 5):
    rng = np.random.default_rng(7)
    t = np.arange(n, dtype=np.float64)
    x = np.stack(
        [
            np.sin(2 * np.pi * t / (35 + 4 * k))
            + 0.35 * np.sin(2 * np.pi * t / (9 + k))
            + 0.12 * rng.standard_normal(n)
            for k in range(d)
        ],
        axis=1,
    )
    if task == "classification":
        score = 0.6 * x[:, 0] + 0.3 * x[:, 1] - 0.2 * x[:, 2]
        y = np.digitize(score, [-0.25, 0.25]).astype(np.int64)[:, None]
    else:
        y = (
            0.55 * x[:, [0]]
            + 0.25 * x[:, [1]]
            - 0.15 * x[:, [2]]
            + 0.05 * rng.standard_normal((n, 1))
        )
    return x, y


def demo_run(task: str, epochs: int, device: torch.device) -> None:
    x, y = make_synthetic_demo(task=task)
    (xtr, ytr), (xv, yv), (xte, yte) = chronological_split(x, y)
    xs = StandardScaler().fit(xtr)
    xtr, xv, xte = map(xs.transform, (xtr, xv, xte))
    if task == "regression":
        ys = StandardScaler().fit(ytr)
        ytr, yv, yte = map(ys.transform, (ytr, yv, yte))

    cfg = PSDformerConfig(
        input_dim=x.shape[1],
        target_dim=1,
        task=task,
        num_classes=3,
        lookback=24,
        horizon=1,
        segment_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        edm_levels=2,
        dgc_growth=8,
        batch_size=32,
    )
    tr = SlidingWindowDataset(xtr, ytr, cfg.lookback, cfg.horizon, task)
    va = SlidingWindowDataset(xv, yv, cfg.lookback, cfg.horizon, task)
    te = SlidingWindowDataset(xte, yte, cfg.lookback, cfg.horizon, task)
    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=cfg.batch_size)
    test_loader = DataLoader(te, batch_size=cfg.batch_size)
    model = PSDformer(cfg)

    # Shape smoke test before training.
    xb, yb = next(iter(train_loader))
    with torch.no_grad():
        o = model(xb.to(device)) if next(model.parameters()).device == device else model.to(device)(xb.to(device))
    print("Smoke-test input:", tuple(xb.shape))
    print("Smoke-test output:", tuple(o["prediction"].shape))
    print("Initial fusion weights:", o["fusion_weights"].detach().cpu().numpy())

    fit_model(model, train_loader, val_loader, device, epochs=epochs)
    losses, metrics = evaluate(model, test_loader, device)
    print("Test losses:", json.dumps(losses, indent=2))
    print("Test metrics:", json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSDformer manuscript reproduction")
    p.add_argument("--demo", action="store_true", help="Run a synthetic smoke/training demo")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--features", type=str, default=None, help="Comma-separated feature columns")
    p.add_argument("--targets", type=str, default=None, help="Comma-separated target columns")
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--segment-len", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--filter-window", type=int, default=1)
    p.add_argument("--balanced-sampling", action="store_true")
    p.add_argument("--checkpoint", type=str, default="psdformer_best.pt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.demo or args.csv is None:
        demo_run(args.task, epochs=max(1, min(args.epochs, 5)), device=device)
        return

    if not args.features or not args.targets:
        raise SystemExit("--features and --targets are required with --csv")
    features = [s.strip() for s in args.features.split(",") if s.strip()]
    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    cfg = PSDformerConfig(
        input_dim=len(features),
        target_dim=len(targets) if args.task == "regression" else 1,
        task=args.task,
        num_classes=args.num_classes,
        lookback=args.lookback,
        horizon=args.horizon,
        segment_len=args.segment_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    train_loader, val_loader, test_loader, meta = build_loaders_from_csv(
        cfg,
        args.csv,
        features,
        targets,
        filter_window=args.filter_window,
        balanced_sampling=args.balanced_sampling,
    )
    model = PSDformer(cfg)
    checkpoint = Path(args.checkpoint)
    fit_model(model, train_loader, val_loader, device, epochs=args.epochs, checkpoint=checkpoint)

    # Load best validation checkpoint for final test.
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
    test_losses, test_metrics = evaluate(model, test_loader, device)
    print("Test losses:", json.dumps(test_losses, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))

    meta_path = checkpoint.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({"config": asdict(cfg), "data": meta}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved checkpoint:", checkpoint)
    print("Saved metadata:", meta_path)


if __name__ == "__main__":
    main()
