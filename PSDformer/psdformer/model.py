"""Configurable, independent PyTorch reference implementation of PSDformer.

This is not official code and does not establish an exact reproduction. The
manuscript specifies the attention/merging scaffold, but leaves architectural
hyperparameters and DGC shape handling unspecified. The defaults below are
illustrative engineering choices. In particular, the classification head is an
optional extension: its label count, targets and loss are not given by the
manuscript.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn

from .blocks import (
    DecoderLayer,
    DenseGraphConvolution,
    MultiDimensionalSegmentEmbedding,
    SegmentMerging,
    TwoStageDependencyExtractor,
)


@dataclass
class PSDformerConfig:
    """Model settings. Only ``input_len=24`` is reported in the manuscript.

    These default dimensions are examples, not recovered study settings.
    ``num_classes=None`` or ``0`` omits the optional classification head.
    Two DGC blocks are fixed by the paper; their depth, growth and pooling size
    are configurable because the paper does not supply their numeric values.
    """

    input_dim: int = 5
    input_len: int = 24
    pred_len: int = 3
    output_dim: int = 1
    num_classes: int | None = 3
    d_model: int = 64
    n_heads: int = 4
    segment_len: int = 4
    router_count: int = 4
    tde_layers: int = 1
    encoder_levels: int = 3
    dropout: float = 0.1
    ff_multiplier: int = 4
    dgc_growth_rate: int = 16
    dgc_layers_per_block: int = 2
    dgc_pool_size: int = 2

    def __post_init__(self) -> None:
        positive_fields = (
            "input_dim", "input_len", "pred_len", "output_dim", "d_model",
            "n_heads", "segment_len", "router_count", "tde_layers",
            "encoder_levels", "ff_multiplier", "dgc_growth_rate",
            "dgc_layers_per_block", "dgc_pool_size",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be a number in [0, 1).")
        if self.num_classes is not None and (
            isinstance(self.num_classes, bool)
            or not isinstance(self.num_classes, int)
            or self.num_classes < 0
        ):
            raise ValueError("num_classes must be None or a nonnegative integer.")

    def to_dict(self) -> dict:
        return asdict(self)


class PSDformer(nn.Module):
    """MDSW -> TDE -> hierarchical EDM -> two DGC blocks -> task heads.

    Args:
        config: A validated ``PSDformerConfig``.

    Forward:
        x: Floating point tensor [B,input_len,input_dim].
        mask: Optional boolean tensor of the same shape. True means observed.
            Non-finite samples are always treated as unobserved. Missing entries
            are replaced by zero before embedding; the mask is not an attention
            mask or a learned missing-data model. Preprocess/normalize observed
            samples with training-set statistics outside this module.

    Returns:
        A dictionary with ``forecast`` [B,pred_len,output_dim], ``features``
        [B,d_model], ``edm_forecast`` [B,pred_len,input_dim], and
        ``scale_forecasts`` [B,encoder_levels,pred_len,input_dim]. The last two
        expose the exact multiscale sum before DGC. ``logits`` [B,num_classes]
        is included only when the optional classification head is enabled.

    Boundary choices: zero-pad the last input segment, duplicate the final
    token for odd encoder merges, crop padded decoder predictions to pred_len.
    Encoder memories are consumed fine-to-coarse at matching decoder levels.
    """

    def __init__(self, config: PSDformerConfig | None = None):
        super().__init__()
        self.config = config if config is not None else PSDformerConfig()
        if not isinstance(self.config, PSDformerConfig):
            raise TypeError("config must be a PSDformerConfig instance.")
        # Revalidate in case a caller modified the dataclass after creation.
        self.config.__post_init__()
        c = self.config
        self.input_segments = math.ceil(c.input_len / c.segment_len)
        self.output_segments = math.ceil(c.pred_len / c.segment_len)
        self.embedding = MultiDimensionalSegmentEmbedding(
            c.input_dim, c.input_len, c.segment_len, c.d_model
        )
        self.pre_encoder = nn.ModuleList(
            [self._make_tde(self.input_segments) for _ in range(c.tde_layers)]
        )
        # Algorithm 1 uses the standalone TDE output directly as encoder level 1.
        self.encoder_merges = nn.ModuleList()
        self.encoder_tdes = nn.ModuleList()
        self.encoder_segment_counts = [self.input_segments]
        segments = self.input_segments
        for _ in range(1, c.encoder_levels):
            segments = math.ceil(segments / 2)
            self.encoder_segment_counts.append(segments)
            self.encoder_merges.append(SegmentMerging(c.d_model))
            self.encoder_tdes.append(self._make_tde(segments))
        self.decoder_queries = nn.Parameter(
            torch.empty(1, c.input_dim, self.output_segments, c.d_model)
        )
        nn.init.normal_(self.decoder_queries, std=0.02)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    c.d_model, c.n_heads, self.output_segments, c.segment_len,
                    c.router_count, c.ff_multiplier, c.dropout
                )
                for _ in range(c.encoder_levels)
            ]
        )
        self.dgc = DenseGraphConvolution(
            c.dgc_growth_rate, c.dgc_layers_per_block, c.dgc_pool_size
        )
        flattened_size = (
            self.dgc.output_channels * c.input_dim * self.dgc.output_length(c.pred_len)
        )
        # The paper does not specify this final flatten/projection head. Keeping
        # variable/time positions avoids imposing unreported graph aggregation.
        self.feature_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_size, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
        )
        self.forecast_head = nn.Linear(c.d_model, c.pred_len * c.output_dim)
        self.classifier = (
            nn.Linear(c.d_model, c.num_classes) if c.num_classes else None
        )

    def _make_tde(self, segments: int) -> TwoStageDependencyExtractor:
        c = self.config
        return TwoStageDependencyExtractor(
            c.d_model, c.n_heads, segments, c.router_count,
            c.ff_multiplier, c.dropout
        )

    def _sanitize(self, x: Tensor, mask: Tensor | None) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor.")
        expected = (self.config.input_len, self.config.input_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"x must have shape [B,{expected[0]},{expected[1]}]; got {tuple(x.shape)}."
            )
        if x.shape[0] < 1:
            raise ValueError("x must contain at least one batch element.")
        if not x.is_floating_point():
            raise TypeError("x must have a real floating point dtype.")
        observed = torch.isfinite(x)
        if mask is not None:
            if not isinstance(mask, Tensor) or mask.shape != x.shape:
                raise ValueError("mask must be a tensor with the same shape as x.")
            if mask.dtype != torch.bool:
                raise TypeError("mask must have dtype torch.bool; True means observed.")
            if mask.device != x.device:
                raise ValueError("mask and x must be on the same device.")
            observed = observed & mask
        # Replacing NaN and +/-Inf explicitly avoids all-missing softmax masks.
        return torch.where(observed, x, torch.zeros((), dtype=x.dtype, device=x.device))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        x = self._sanitize(x, mask)
        encoded = self.embedding(x)
        for layer in self.pre_encoder:
            encoded = layer(encoded)
        memories = [encoded]
        for merge, tde in zip(self.encoder_merges, self.encoder_tdes):
            encoded = tde(merge(encoded))
            memories.append(encoded)
        decoded = self.decoder_queries.expand(x.shape[0], -1, -1, -1)
        predictions = []
        for decoder, memory in zip(self.decoder_layers, memories):
            decoded, prediction = decoder(decoded, memory)
            predictions.append(prediction[:, :, : self.config.pred_len])
        # [B,levels,D,K] -> exact Eq. (18) text's multiscale projected sum.
        scale_predictions = torch.stack(predictions, dim=1)
        edm_series = scale_predictions.sum(dim=1)
        features = self.feature_head(self.dgc(edm_series))
        forecast = self.forecast_head(features).reshape(
            x.shape[0], self.config.pred_len, self.config.output_dim
        )
        output = {
            "forecast": forecast,
            "features": features,
            "edm_forecast": edm_series.transpose(1, 2).contiguous(),
            "scale_forecasts": scale_predictions.permute(0, 1, 3, 2).contiguous(),
        }
        if self.classifier is not None:
            output["logits"] = self.classifier(features)
        return output
