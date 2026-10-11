"""Building blocks for an independent, equation-guided PSDformer reference.

Internal attention tensors use [batch, variable, segment, embedding]. This
module follows the manuscript's Algorithm 1 and Eqs. (1)--(19), rather than
claiming to recover an unpublished implementation. Details not specified in
the manuscript are identified explicitly below.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    """Deterministic position codes, including odd and one-dimensional widths."""
    if width == 0:
        return torch.empty(length, 0)
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10000.0) / max(width, 1))
    )
    codes = torch.zeros(length, width)
    codes[:, 0::2] = torch.sin(positions * frequencies)
    codes[:, 1::2] = torch.cos(positions * frequencies[: width // 2])
    return codes


def fixed_2d_positions(input_dim: int, segments: int, d_model: int) -> Tensor:
    """Return [1,D,L,E] fixed codes identifying variable and segment positions.

    The paper specifies fixed two-dimensional positions but not their formula.
    We concatenate sinusoidal variable and segment codes. A one-channel model
    uses their sum, since concatenation cannot represent both axes at width 1.
    """
    if d_model == 1:
        variable = _sinusoidal_positions(input_dim, 1)[:, None, :]
        temporal = _sinusoidal_positions(segments, 1)[None, :, :]
        return (variable + temporal).unsqueeze(0)
    variable_width = d_model // 2
    temporal_width = d_model - variable_width
    variable = _sinusoidal_positions(input_dim, variable_width)
    temporal = _sinusoidal_positions(segments, temporal_width)
    return torch.cat(
        (
            variable[:, None, :].expand(input_dim, segments, variable_width),
            temporal[None, :, :].expand(input_dim, segments, temporal_width),
        ),
        dim=-1,
    ).unsqueeze(0)


class MultiDimensionalSegmentEmbedding(nn.Module):
    """MDSW: separate variables, non-overlapping segments, shared projection.

    The final incomplete input segment is right-padded with zeros. This padding
    rule is an implementation choice; it is not specified by the paper.
    """

    def __init__(self, input_dim: int, input_len: int, segment_len: int, d_model: int):
        super().__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.segment_len = segment_len
        self.num_segments = math.ceil(input_len / segment_len)
        self.projection = nn.Linear(segment_len, d_model)
        self.register_buffer(
            "position_encoding",
            fixed_2d_positions(input_dim, self.num_segments, d_model),
            persistent=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # [B,T,D] -> [B,D,L,S]. No flattening mixes variables at tokenization.
        x = x.transpose(1, 2)
        pad = self.num_segments * self.segment_len - self.input_len
        if pad:
            x = F.pad(x, (0, pad))
        x = x.reshape(x.shape[0], self.input_dim, self.num_segments, self.segment_len)
        tokens = self.projection(x)
        return tokens + self.position_encoding.to(dtype=tokens.dtype)


class FeedForward(nn.Module):
    """Positionwise MLP; GELU, hidden width and dropout are reference choices."""

    def __init__(self, d_model: int, ff_multiplier: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TwoStageDependencyExtractor(nn.Module):
    """TDE Eqs. (4)--(9): temporal attention, then routed variable attention.

    Each segment has learned router queries [R,E]. The collection and
    distribution attention weights are shared across segment positions, as
    described in the paper. Residual connections use post-LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        segments: int,
        router_count: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.segments = segments
        self.d_model = d_model
        self.router_count = router_count
        self.routers = nn.Parameter(torch.empty(segments, router_count, d_model))
        nn.init.normal_(self.routers, std=0.02)
        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.router_collect = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.router_distribute = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm1 = nn.LayerNorm(d_model)
        self.temporal_norm2 = nn.LayerNorm(d_model)
        self.dimension_norm1 = nn.LayerNorm(d_model)
        self.dimension_norm2 = nn.LayerNorm(d_model)
        self.temporal_ff = FeedForward(d_model, ff_multiplier, dropout)
        self.dimension_ff = FeedForward(d_model, ff_multiplier, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, d, l, e = x.shape
        if l != self.segments or e != self.d_model:
            raise ValueError(
                f"TDE expects segment/embedding shape ({self.segments}, "
                f"{self.d_model}); got ({l}, {e})."
            )
        # A separate temporal sequence for each input variable: [B*D,L,E].
        temporal = x.reshape(b * d, l, e)
        attended = self.temporal_attention(
            temporal, temporal, temporal, need_weights=False
        )[0]
        temporal = self.temporal_norm1(temporal + self.dropout(attended))
        temporal = self.temporal_norm2(temporal + self.temporal_ff(temporal))
        # A separate variable sequence for each segment: [B*L,D,E].
        dimensional = temporal.reshape(b, d, l, e).permute(0, 2, 1, 3)
        dimensional = dimensional.reshape(b * l, d, e)
        routers = self.routers.unsqueeze(0).expand(b, l, self.router_count, e)
        routers = routers.reshape(b * l, self.router_count, e)
        messages = self.router_collect(
            routers, dimensional, dimensional, need_weights=False
        )[0]
        distributed = self.router_distribute(
            dimensional, messages, messages, need_weights=False
        )[0]
        dimensional = self.dimension_norm1(dimensional + self.dropout(distributed))
        dimensional = self.dimension_norm2(
            dimensional + self.dimension_ff(dimensional)
        )
        return dimensional.reshape(b, l, d, e).permute(0, 2, 1, 3).contiguous()


class SegmentMerging(nn.Module):
    """Eq. (12): concatenate each adjacent temporal pair, then project 2E->E.

    If a level has an odd number of segments, replicate the final segment so
    every merge has two inputs. The manuscript does not define the odd case.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(2 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[2] % 2:
            x = torch.cat((x, x[:, :, -1:, :]), dim=2)
        pairs = torch.cat((x[:, :, 0::2, :], x[:, :, 1::2, :]), dim=-1)
        return self.projection(pairs)


class DecoderLayer(nn.Module):
    """Eqs. (14)--(18): TDE, per-variable cross-attention, residual MLP.

    A shared E->S linear projection maps every output segment to S time samples.
    Predictions retain all input variables; horizon padding is cropped by the
    caller. Decoder scale outputs are summed, not averaged.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        output_segments: int,
        segment_len: int,
        router_count: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tde = TwoStageDependencyExtractor(
            d_model, n_heads, output_segments, router_count, ff_multiplier, dropout
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_multiplier, dropout)
        self.dropout = nn.Dropout(dropout)
        self.prediction = nn.Linear(d_model, segment_len)

    def forward(self, query: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        query = self.tde(query)
        b, d, l, e = query.shape
        q = query.reshape(b * d, l, e)
        kv = memory.reshape(b * d, memory.shape[2], e)
        attended = self.cross_attention(q, kv, kv, need_weights=False)[0]
        q = self.norm1(q + self.dropout(attended))
        q = self.norm2(q + self.ff(q))
        decoded = q.reshape(b, d, l, e)
        prediction = self.prediction(decoded).flatten(start_dim=2)
        return decoded, prediction


class SameLengthMaxPool(nn.Module):
    """Temporal max pooling with explicit SAME padding and stride 1."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        total = self.kernel_size - 1
        left = total // 2
        right = total - left
        if total:
            x = F.pad(x, (left, right, 0, 0), value=float("-inf"))
        return F.max_pool2d(x, kernel_size=(1, self.kernel_size), stride=1)


class DenseConvolutionBlock(nn.Module):
    """Eq. (19) dense feature reuse using 1x3 convolution and 1xq pooling.

    The paper does not specify how pooled maps with different lengths can be
    concatenated. Our explicit reference policy keeps each dense layer's pooling
    stride at 1 with SAME padding, then compresses time once at the block exit.
    ``growth_rate`` means new convolution channels, while ``pool_size`` is q;
    the manuscript conflates these quantities. No graph adjacency is invented.
    """

    def __init__(self, input_channels: int, growth_rate: int, layers: int, pool_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_channels + i * growth_rate, growth_rate,
                              kernel_size=(1, 3), padding=(0, 1)),
                    SameLengthMaxPool(pool_size),
                )
                for i in range(layers)
            ]
        )
        self.pool_size = pool_size
        self.output_channels = input_channels + layers * growth_rate

    def forward(self, x: Tensor) -> Tensor:
        maps = [x]
        for layer in self.layers:
            maps.append(layer(torch.cat(maps, dim=1)))
        merged = torch.cat(maps, dim=1)
        # ceil_mode preserves at least one sample even for tiny horizons.
        return F.max_pool2d(
            merged, kernel_size=(1, self.pool_size),
            stride=(1, self.pool_size), ceil_mode=True
        )


class DenseGraphConvolution(nn.Module):
    """The manuscript's two-block DGC, implemented as its described dense CNN.

    Input is the summed EDM series [B,D,K], treated as [B,1,D,K]. Outputs are
    [B,C,D,K_compressed]. The manuscript provides no adjacency matrix or graph
    convolution operator, so the historical DGC name does not imply either.
    """

    def __init__(self, growth_rate: int = 16, layers_per_block: int = 2, pool_size: int = 2):
        super().__init__()
        first = DenseConvolutionBlock(1, growth_rate, layers_per_block, pool_size)
        second = DenseConvolutionBlock(
            first.output_channels, growth_rate, layers_per_block, pool_size
        )
        self.blocks = nn.ModuleList([first, second])
        self.output_channels = second.output_channels
        self.pool_size = pool_size

    def output_length(self, input_length: int) -> int:
        for _ in self.blocks:
            input_length = math.ceil(input_length / self.pool_size)
        return input_length

    def forward(self, series: Tensor) -> Tensor:
        x = series.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return x
