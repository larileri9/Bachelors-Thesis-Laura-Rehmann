from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BaskervilleBuildError(ValueError):
    """Raised when a Baskerville config cannot be mapped to a Torch model."""


# ---------------------------------------------------------------------------
# Initializer helpers
# ---------------------------------------------------------------------------


def _tf_initializer_name(spec: Any, default: str) -> str:
    if spec is None:
        return default
    if isinstance(spec, str):
        return spec.lower()
    if isinstance(spec, dict):
        class_name = spec.get("class_name") or spec.get("name")
        if isinstance(class_name, str):
            return class_name.lower()
    raise BaskervilleBuildError(f"Unsupported initializer spec {spec!r}")


def _fan_in_out(tensor: torch.Tensor) -> tuple[float, float]:
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    return float(fan_in), float(fan_out)


def _apply_tf_weight_initializer(tensor: torch.Tensor, initializer: str) -> None:
    name = initializer.lower()
    if name in {"he_normal", "kaiming_normal", "he_normal_v2"}:
        nn.init.kaiming_normal_(tensor, mode="fan_in", nonlinearity="relu")
        return
    if name in {"he_uniform", "kaiming_uniform", "he_uniform_v2"}:
        nn.init.kaiming_uniform_(tensor, mode="fan_in", nonlinearity="relu")
        return
    if name in {"glorot_uniform", "xavier_uniform"}:
        nn.init.xavier_uniform_(tensor)
        return
    if name in {"glorot_normal", "xavier_normal"}:
        nn.init.xavier_normal_(tensor)
        return
    if name == "lecun_normal":
        fan_in, _ = _fan_in_out(tensor)
        std = math.sqrt(1.0 / max(fan_in, 1.0))
        nn.init.normal_(tensor, mean=0.0, std=std)
        return
    if name == "lecun_uniform":
        fan_in, _ = _fan_in_out(tensor)
        bound = math.sqrt(3.0 / max(fan_in, 1.0))
        nn.init.uniform_(tensor, -bound, bound)
        return
    if name in {"zeros", "zero"}:
        nn.init.zeros_(tensor)
        return
    if name in {"ones", "one"}:
        nn.init.ones_(tensor)
        return
    if name == "orthogonal":
        nn.init.orthogonal_(tensor)
        return
    if name in {"random_normal", "normal"}:
        nn.init.normal_(tensor, mean=0.0, std=0.05)
        return
    if name in {"truncated_normal", "truncatednormal"}:
        nn.init.trunc_normal_(tensor, mean=0.0, std=0.05, a=-0.1, b=0.1)
        return
    raise BaskervilleBuildError(f"Unsupported TF initializer '{initializer}'")


def _init_linear_or_conv(module: nn.Module, initializer: str) -> None:
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        _apply_tf_weight_initializer(module.weight, initializer)
    if (
        hasattr(module, "bias")
        and isinstance(module.bias, torch.Tensor)
        and module.bias is not None
    ):
        nn.init.zeros_(module.bias)


def _init_norm(module: nn.Module) -> None:
    if (
        hasattr(module, "weight")
        and isinstance(module.weight, torch.Tensor)
        and module.weight is not None
    ):
        nn.init.ones_(module.weight)
    if (
        hasattr(module, "bias")
        and isinstance(module.bias, torch.Tensor)
        and module.bias is not None
    ):
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Activation / Norm helpers
# ---------------------------------------------------------------------------


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU(approximate="tanh")
    if key in {"linear", "identity", "none"}:
        return nn.Identity()
    if key == "softmax":
        return nn.Softmax(dim=-1)
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "softplus":
        return nn.Softplus()
    raise BaskervilleBuildError(f"Unsupported activation '{name}'")


def _same_padding(kernel_size: int) -> int:
    return kernel_size // 2


class _ChannelsFirstLayerNorm(nn.Module):
    """LayerNorm that operates on the channel dim of [B, C, L] tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _maybe_norm(norm_type: str | None, channels: int, bn_momentum_tf: float = 0.99) -> nn.Module:
    if norm_type is None:
        return nn.Identity()
    key = norm_type.lower()
    if key.startswith("batch"):
        return nn.BatchNorm1d(channels, eps=1e-3, momentum=1.0 - bn_momentum_tf)
    if key == "layer":
        return _ChannelsFirstLayerNorm(channels)
    raise BaskervilleBuildError(f"Unsupported norm_type '{norm_type}'")


# ---------------------------------------------------------------------------
# Scale (learnable, zero-initialized)
# ---------------------------------------------------------------------------


class _Scale1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.view(1, -1, 1)


# ---------------------------------------------------------------------------
# ConvDNA
# ---------------------------------------------------------------------------


class ConvDNA(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cfg: dict[str, Any], global_cfg: dict[str, Any]):
        super().__init__()
        kernel_size = int(cfg.get("kernel_size", 15))
        stride = int(cfg.get("stride", 1))
        norm_type = cfg.get("norm_type", global_cfg.get("norm_type"))
        activation = cfg.get("activation", global_cfg.get("activation", "relu"))
        dropout = float(cfg.get("dropout", 0.0))
        pool_size = int(cfg.get("pool_size", 1))
        bn_momentum = float(cfg.get("bn_momentum", global_cfg.get("bn_momentum", 0.99)))

        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=_same_padding(kernel_size),
            bias=norm_type is None,
        )
        self.norm = _maybe_norm(norm_type, out_ch, bn_momentum)
        self.act = _activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = (
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
            if pool_size > 1
            else nn.Identity()
        )
        self.kernel_initializer = _tf_initializer_name(
            cfg.get("kernel_initializer"),
            _tf_initializer_name(global_cfg.get("kernel_initializer"), "he_normal"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.pool(y)
        return y

    def apply_tf_initialization(self) -> None:
        _init_linear_or_conv(self.conv, self.kernel_initializer)
        if not isinstance(self.norm, nn.Identity):
            _init_norm(self.norm)


# ---------------------------------------------------------------------------
# ResidualTower
# ---------------------------------------------------------------------------


class ResidualTower(nn.Module):
    def __init__(self, in_ch: int, cfg: dict[str, Any], global_cfg: dict[str, Any]):
        super().__init__()
        repeat = int(cfg.get("repeat", 1))
        num_convs = int(cfg.get("num_convs", 2))
        kernel_size = int(cfg.get("kernel_size", 1))
        pool_size = int(cfg.get("pool_size", 2))
        dropout = float(cfg.get("dropout", 0.0))
        norm_type = cfg.get("norm_type", global_cfg.get("norm_type"))
        activation = cfg.get("activation", global_cfg.get("activation", "relu"))
        bn_momentum = float(cfg.get("bn_momentum", global_cfg.get("bn_momentum", 0.99)))
        kernel_initializer = _tf_initializer_name(
            cfg.get("kernel_initializer"),
            _tf_initializer_name(global_cfg.get("kernel_initializer"), "he_normal"),
        )

        filters_init = int(cfg.get("filters_init", in_ch))
        filters_end = int(cfg.get("filters_end", filters_init))
        divisible_by = int(cfg.get("divisible_by", 1))

        def round_div(x: float) -> int:
            return int(round(x / divisible_by) * divisible_by)

        if repeat > 1:
            mult = math.exp(math.log(filters_end / filters_init) / (repeat - 1))
        else:
            mult = 1.0

        def conv_nac(in_channels: int, out_channels: int, ksize: int) -> nn.Module:
            return nn.Sequential(
                _maybe_norm(norm_type, in_channels, bn_momentum),
                _activation(activation),
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=ksize,
                    padding=_same_padding(ksize),
                    bias=True,
                ),
            )

        self.blocks = nn.ModuleList()
        self.kernel_initializer = kernel_initializer
        self.repr_channels: list[int] = []
        ch = in_ch
        rep_filters = float(filters_init)
        for _ in range(repeat):
            out_ch = round_div(rep_filters)
            self.repr_channels.append(out_ch)
            first = conv_nac(ch, out_ch, kernel_size)

            convs = nn.ModuleList()
            for _ci in range(1, num_convs):
                convs.append(conv_nac(out_ch, out_ch, 1))

            down = (
                nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
                if pool_size > 1
                else nn.Identity()
            )
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "first": first,
                        "convs": convs,
                        "scale": _Scale1D(out_ch),
                        "drop": nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                        "down": down,
                    }
                )
            )
            ch = out_ch
            rep_filters *= mult

        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        reprs: list[torch.Tensor] = []
        y = x
        for block in self.blocks:
            out = block["first"](y)
            out0 = out
            for conv in block["convs"]:
                out = conv(out)
            out = block["drop"](out)
            if len(block["convs"]) > 0:
                out = out0 + block["scale"](out)
            reprs.append(out)
            y = block["down"](out)
        return y, reprs

    def apply_tf_initialization(self) -> None:
        for block in self.blocks:
            for seq in [block["first"]] + list(block["convs"]):
                for layer in seq:
                    if isinstance(layer, nn.Conv1d):
                        _init_linear_or_conv(layer, self.kernel_initializer)
                    elif isinstance(layer, (nn.BatchNorm1d, _ChannelsFirstLayerNorm)):
                        _init_norm(layer if isinstance(layer, nn.BatchNorm1d) else layer.norm)


# ---------------------------------------------------------------------------
# Relative positional encoding (matching baskerville/layers.py)
# ---------------------------------------------------------------------------


def _positional_features_central_mask(
    positions: torch.Tensor, feature_size: int, seq_length: int
) -> torch.Tensor:
    pow_rate = np.exp(np.log(seq_length + 1) / feature_size).astype(np.float32)
    center_widths = pow_rate ** torch.arange(
        1, feature_size + 1, dtype=torch.float32, device=positions.device
    )
    center_widths = center_widths - 1
    outputs = (center_widths > positions.abs().unsqueeze(-1)).float()
    return outputs


def _positional_features(
    positions: torch.Tensor, feature_size: int, seq_length: int, *, symmetric: bool = False
) -> torch.Tensor:
    num_components = 1 if symmetric else 2
    num_basis = feature_size // num_components
    embeddings = _positional_features_central_mask(positions, num_basis, seq_length)
    if not symmetric:
        embeddings = torch.cat([embeddings, positions.sign().unsqueeze(-1) * embeddings], dim=-1)
    return embeddings


def _relative_shift(x: torch.Tensor) -> torch.Tensor:
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat([to_pad, x], dim=-1)
    B, H, t1, t2 = x.shape
    x = x.reshape(B, H, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(B, H, t1, t2 - 1)
    x = x[..., : (t2 + 1) // 2]
    return x


# ---------------------------------------------------------------------------
# MultiheadAttention (faithful to baskerville/layers.py)
# ---------------------------------------------------------------------------


class BaskervilleMultiheadAttention(nn.Module):
    """Matches baskerville.layers.MultiheadAttention with relative position bias."""

    def __init__(
        self,
        value_size: int,
        key_size: int,
        heads: int,
        num_position_features: int,
        *,
        attention_dropout: float = 0.05,
        positional_dropout: float = 0.01,
        content_position_bias: bool = True,
        zero_initialize: bool = True,
        initializer: str = "he_normal",
    ):
        super().__init__()
        self._value_size = value_size
        self._key_size = key_size
        self._heads = heads
        self._attention_dropout = attention_dropout
        self._positional_dropout = positional_dropout
        self._content_position_bias = content_position_bias
        self._num_position_features = num_position_features

        key_proj_size = key_size * heads
        embedding_size = value_size * heads

        self.q_layer = nn.Linear(embedding_size, key_proj_size, bias=False)
        self.k_layer = nn.Linear(embedding_size, key_proj_size, bias=False)
        self.v_layer = nn.Linear(embedding_size, embedding_size, bias=False)

        self.r_k_layer = nn.Linear(num_position_features, key_proj_size, bias=False)
        self.r_w_bias = nn.Parameter(torch.zeros(1, heads, 1, key_size))
        self.r_r_bias = nn.Parameter(torch.zeros(1, heads, 1, key_size))

        w_init = "zeros" if zero_initialize else initializer
        self.embedding_layer = nn.Linear(embedding_size, embedding_size, bias=True)

        self._initializer = initializer
        self._embedding_initializer = w_init

    def _multihead_output(self, linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> linear -> [B, H, T, K_or_V]"""
        out = linear(x)
        B, T, _ = out.shape
        kv = out.shape[-1] // self._heads
        out = out.reshape(B, T, self._heads, kv)
        return out.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self._multihead_output(self.q_layer, x)
        k = self._multihead_output(self.k_layer, x)
        v = self._multihead_output(self.v_layer, x)

        q = q * (self._key_size**-0.5)

        content_logits = torch.matmul(q + self.r_w_bias, k.transpose(-1, -2))

        if self._num_position_features > 0:
            distances = torch.arange(-(T - 1), T, dtype=torch.float32, device=x.device).unsqueeze(0)
            pos_enc = _positional_features(
                distances, self._num_position_features, T, symmetric=False
            )

            if self.training and self._positional_dropout > 0:
                pos_enc = F.dropout(pos_enc, p=self._positional_dropout, training=True)

            r_k = self._multihead_output(self.r_k_layer, pos_enc)

            if self._content_position_bias:
                relative_logits = torch.matmul(q + self.r_r_bias, r_k.transpose(-1, -2))
            else:
                relative_logits = torch.matmul(self.r_r_bias, r_k.transpose(-1, -2))
                relative_logits = relative_logits.expand(B, -1, T, -1)

            relative_logits = _relative_shift(relative_logits)
            logits = content_logits + relative_logits
        else:
            logits = content_logits

        weights = torch.softmax(logits, dim=-1)

        if self.training and self._attention_dropout > 0:
            weights = F.dropout(weights, p=self._attention_dropout, training=True)

        output = torch.matmul(weights, v)
        output = output.permute(0, 2, 1, 3).reshape(B, T, -1)
        output = self.embedding_layer(output)
        return output

    def apply_tf_initialization(self) -> None:
        _init_linear_or_conv(self.q_layer, self._initializer)
        _init_linear_or_conv(self.k_layer, self._initializer)
        _init_linear_or_conv(self.v_layer, self._initializer)
        _init_linear_or_conv(self.r_k_layer, self._initializer)
        _apply_tf_weight_initializer(self.r_w_bias.data, self._initializer)
        _apply_tf_weight_initializer(self.r_r_bias.data, self._initializer)
        _init_linear_or_conv(self.embedding_layer, self._embedding_initializer)


# ---------------------------------------------------------------------------
# Transformer block + tower (matching baskerville/blocks.py transformer())
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer block matching baskerville.blocks.transformer()."""

    def __init__(
        self,
        channels: int,
        *,
        key_size: int = 64,
        heads: int = 4,
        num_position_features: int = 32,
        dense_expansion: float = 2.0,
        dropout: float = 0.2,
        attention_dropout: float = 0.05,
        position_dropout: float = 0.01,
        content_position_bias: bool = True,
        mha_initializer: str = "he_normal",
        kernel_initializer: str = "he_normal",
    ):
        super().__init__()
        assert channels % heads == 0
        value_size = channels // heads

        self.norm1 = nn.LayerNorm(channels, eps=1e-3)
        self.mha = BaskervilleMultiheadAttention(
            value_size=value_size,
            key_size=key_size,
            heads=heads,
            num_position_features=num_position_features,
            attention_dropout=attention_dropout,
            positional_dropout=position_dropout,
            content_position_bias=content_position_bias,
            zero_initialize=True,
            initializer=mha_initializer,
        )
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        expansion_filters = int(dense_expansion * channels)
        self.norm2 = nn.LayerNorm(channels, eps=1e-3)
        self.dense1 = nn.Linear(channels, expansion_filters)
        self.drop_dense1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn_act = nn.ReLU()
        self.dense2 = nn.Linear(expansion_filters, channels)
        self.drop_dense2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._kernel_initializer = kernel_initializer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.mha(h)
        h = self.drop1(h)
        x = x + h

        if self.dense1 is not None:
            h = self.norm2(x)
            h = self.dense1(h)
            h = self.drop_dense1(h)
            h = self.ffn_act(h)
            h = self.dense2(h)
            h = self.drop_dense2(h)
            x = x + h

        return x

    def apply_tf_initialization(self) -> None:
        _init_norm(self.norm1)
        self.mha.apply_tf_initialization()
        _init_norm(self.norm2)
        _init_linear_or_conv(self.dense1, self._kernel_initializer)
        _init_linear_or_conv(self.dense2, self._kernel_initializer)


class TransformerTower(nn.Module):
    def __init__(self, channels: int, cfg: dict[str, Any], global_cfg: dict[str, Any]):
        super().__init__()
        repeat = int(cfg.get("repeat", 1))
        key_size = int(cfg.get("key_size", 64))
        heads = int(cfg.get("heads", 4))
        dropout = float(cfg.get("dropout", 0.2))
        attention_dropout = float(cfg.get("attention_dropout", 0.05))
        position_dropout = float(cfg.get("position_dropout", 0.01))
        dense_expansion = float(cfg.get("dense_expansion", 2.0))
        content_position_bias = bool(cfg.get("content_position_bias", True))

        num_position_features = cfg.get("num_position_features")
        if num_position_features is None:
            value_size = channels // heads
            divisible_by = 2
            num_position_features = (value_size // divisible_by) * divisible_by
        else:
            num_position_features = int(num_position_features)

        ki_resolved = cfg.get(
            "kernel_initializer", global_cfg.get("kernel_initializer", "he_normal")
        )
        kernel_initializer = _tf_initializer_name(ki_resolved, "he_normal")
        mha_initializer = _tf_initializer_name(cfg.get("mha_initializer"), "he_normal")

        self.layers = nn.ModuleList()
        for _ in range(repeat):
            self.layers.append(
                TransformerBlock(
                    channels,
                    key_size=key_size,
                    heads=heads,
                    num_position_features=num_position_features,
                    dense_expansion=dense_expansion,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    position_dropout=position_dropout,
                    content_position_bias=content_position_bias,
                    mha_initializer=mha_initializer,
                    kernel_initializer=kernel_initializer,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        for layer in self.layers:
            y = layer(y)
        return y.transpose(1, 2)

    def apply_tf_initialization(self) -> None:
        for layer in self.layers:
            layer.apply_tf_initialization()


# ---------------------------------------------------------------------------
# UnetConv (matching baskerville/blocks.py unet_conv())
# ---------------------------------------------------------------------------


class UnetConv(nn.Module):
    """Faithful port of baskerville.blocks.unet_conv().

    TF data format: [B, L, C]. PyTorch uses [B, C, L] in the trunk.
    All internal operations are done in channels-first format.
    """

    def __init__(
        self,
        channels: int,
        skip_channels: int,
        cfg: dict[str, Any],
        global_cfg: dict[str, Any],
    ):
        super().__init__()
        kernel_size = int(cfg.get("kernel_size", 1))
        stride = int(cfg.get("stride", 2))
        activation = cfg.get("activation", global_cfg.get("activation", "relu"))
        norm_type = cfg.get("norm_type", global_cfg.get("norm_type"))
        bn_momentum = float(cfg.get("bn_momentum", global_cfg.get("bn_momentum", 0.99)))
        upsample_conv = bool(cfg.get("upsample_conv", False))

        kernel_initializer = _tf_initializer_name(
            cfg.get("kernel_initializer"),
            _tf_initializer_name(global_cfg.get("kernel_initializer"), "he_normal"),
        )

        self.norm_main = _maybe_norm(norm_type, channels, bn_momentum)
        self.norm_skip = _maybe_norm(norm_type, skip_channels, bn_momentum)
        self.act_main = _activation(activation)
        self.act_skip = _activation(activation)

        self.proj_main = nn.Conv1d(channels, channels, kernel_size=1) if upsample_conv else None
        self.proj_skip = nn.Conv1d(skip_channels, channels, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

        self.sep_depth = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=_same_padding(kernel_size),
            groups=channels,
            bias=False,
        )
        self.sep_point = nn.Conv1d(channels, channels, kernel_size=1, bias=True)

        self.kernel_initializer = kernel_initializer
        self.stride = stride

    def forward(self, x: torch.Tensor, skip_repr: torch.Tensor) -> torch.Tensor:
        cur = self.norm_main(x)
        cur = self.act_main(cur)
        if self.proj_main is not None:
            cur = self.proj_main(cur)

        sk = self.norm_skip(skip_repr)
        sk = self.act_skip(sk)
        sk = self.proj_skip(sk)

        cur = self.upsample(cur)

        min_len = min(cur.shape[-1], sk.shape[-1])
        if cur.shape[-1] != sk.shape[-1]:
            cur = cur[..., :min_len]
            sk = sk[..., :min_len]

        y = cur + sk
        y = self.sep_point(self.sep_depth(y))
        return y

    def apply_tf_initialization(self) -> None:
        if self.proj_main is not None:
            _init_linear_or_conv(self.proj_main, self.kernel_initializer)
        _init_linear_or_conv(self.proj_skip, self.kernel_initializer)
        _init_linear_or_conv(self.sep_depth, self.kernel_initializer)
        _init_linear_or_conv(self.sep_point, self.kernel_initializer)
        for norm in [self.norm_main, self.norm_skip]:
            if not isinstance(norm, nn.Identity):
                target = norm.norm if isinstance(norm, _ChannelsFirstLayerNorm) else norm
                _init_norm(target)


# ---------------------------------------------------------------------------
# FinalHead
# ---------------------------------------------------------------------------


class FinalHead(nn.Module):
    def __init__(self, in_ch: int, units: int, activation: str, kernel_initializer: str):
        super().__init__()
        self.proj = nn.Linear(in_ch, units)
        self.act = _activation(activation)
        self.kernel_initializer = kernel_initializer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.proj(y)
        y = self.act(y)
        return y

    def apply_tf_initialization(self) -> None:
        _init_linear_or_conv(self.proj, self.kernel_initializer)


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


@dataclass
class ModelPlan:
    seq_length: int
    num_features: int
    trunk_blocks: int
    head_name: str
    output_units: int


class BaskervilleTorchModel(nn.Module):
    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        self.model_cfg = model_cfg
        self.seq_length = int(model_cfg["seq_length"])
        self.num_features = int(model_cfg.get("num_features", 4))

        if "trunk" not in model_cfg or not isinstance(model_cfg["trunk"], list):
            raise BaskervilleBuildError("model.trunk must be a list")

        trunk_activation = model_cfg.get("activation", "relu")
        self.trunk_final_act = _activation(trunk_activation)

        self.trunk = nn.ModuleList()

        channels = self.num_features
        spatial = self.seq_length
        repr_entries: list[tuple[int, int]] = []

        for block in model_cfg["trunk"]:
            if not isinstance(block, dict) or "name" not in block:
                raise BaskervilleBuildError("Each trunk block must be an object with 'name'")
            name = str(block["name"])

            if name == "conv_dna":
                out_ch = int(block.get("filters", channels))
                pool_size = int(block.get("pool_size", 1))
                self.trunk.append(ConvDNA(channels, out_ch, block, model_cfg))
                channels = out_ch
                if pool_size > 1:
                    spatial = math.ceil(spatial / pool_size)

            elif name == "res_tower":
                tower = ResidualTower(channels, block, model_cfg)
                pool_size = int(block.get("pool_size", 2))
                tower_sp = spatial
                for rep_ch in tower.repr_channels:
                    repr_entries.append((tower_sp, rep_ch))
                    if pool_size > 1:
                        tower_sp = math.ceil(tower_sp / pool_size)
                spatial = tower_sp
                channels = tower.out_channels
                self.trunk.append(tower)

            elif name == "transformer_tower":
                self.trunk.append(TransformerTower(channels, block, model_cfg))

            elif name == "unet_conv":
                stride = int(block.get("stride", 2))
                target_spatial = spatial * stride
                skip_ch = channels
                for sp, ch in reversed(repr_entries):
                    if sp == target_spatial:
                        skip_ch = ch
                        break
                self.trunk.append(UnetConv(channels, skip_ch, block, model_cfg))
                spatial = target_spatial

            else:
                raise BaskervilleBuildError(f"Unsupported trunk block '{name}'")

        head_name = next((k for k in model_cfg.keys() if k.startswith("head")), None)
        if head_name is None:
            raise BaskervilleBuildError("No head* section found in model config")
        head_cfg = model_cfg[head_name]
        if not isinstance(head_cfg, dict):
            raise BaskervilleBuildError(f"{head_name} must be an object")
        if head_cfg.get("name") != "final":
            raise BaskervilleBuildError("Only final head is currently supported")
        units = int(head_cfg["units"])
        activation = str(head_cfg.get("activation", "linear"))
        head_kernel_initializer = _tf_initializer_name(
            head_cfg.get("kernel_initializer"),
            _tf_initializer_name(model_cfg.get("kernel_initializer"), "he_normal"),
        )
        self.head = FinalHead(channels, units, activation, head_kernel_initializer)

        self.plan = ModelPlan(
            seq_length=self.seq_length,
            num_features=self.num_features,
            trunk_blocks=len(model_cfg["trunk"]),
            head_name=head_name,
            output_units=units,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise BaskervilleBuildError("Expected input tensor shape [batch, seq_len, channels]")
        y = x.transpose(1, 2)
        reprs: list[torch.Tensor] = []

        for block in self.trunk:
            if isinstance(block, ConvDNA):
                y = block(y)
            elif isinstance(block, ResidualTower):
                y, tower_reprs = block(y)
                reprs.extend(tower_reprs)
            elif isinstance(block, TransformerTower):
                y = block(y)
            elif isinstance(block, UnetConv):
                target = None
                for rep in reversed(reprs):
                    if rep.shape[-1] >= y.shape[-1] * 2:
                        target = rep
                        break
                if target is None and reprs:
                    target = reprs[-1]
                if target is None:
                    raise BaskervilleBuildError(
                        "unet_conv requires saved representations from previous tower"
                    )
                y = block(y, target)
            else:
                raise BaskervilleBuildError(f"Unsupported block instance {type(block)!r}")

        y = self.trunk_final_act(y)
        return self.head(y)

    def apply_tf_initialization(self) -> None:
        for block in self.trunk:
            if hasattr(block, "apply_tf_initialization"):
                block.apply_tf_initialization()
        self.head.apply_tf_initialization()


def build_torch_model_from_baskerville_params(model_cfg: dict[str, Any]) -> BaskervilleTorchModel:
    return BaskervilleTorchModel(model_cfg)
