"""Load TF Baskerville H5 weights into the PyTorch model.

Keras H5 files store weights grouped by layer name. Layer names are
auto-incremented in construction order (``conv1d``, ``conv1d_1``, …).
This module rebuilds the expected TF layer sequence from the model config,
then maps each variable to the corresponding PyTorch state-dict key with
the correct axis transpositions.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)

SHORKIE_LM_WEIGHTS_URL = "https://storage.googleapis.com/seqnn-share/shorkie_lm/train/model_best.h5"

TransposeFn = Callable[[np.ndarray], np.ndarray]
WeightEntry = tuple[str, str, str, TransposeFn]

# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------


def get_cached_weights(
    url: str = SHORKIE_LM_WEIGHTS_URL,
    cache_dir: Path | str | None = None,
) -> Path:
    """Download weights lazily; reuse cached file on the same node."""
    if cache_dir is None:
        cache_dir = Path("/tmp/genome_lm_weights_cache")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    filename = f"{url_hash}_{Path(url).name}"
    cached = cache_dir / filename

    if cached.exists():
        logger.info("Using cached weights: %s", cached)
        return cached

    tmp = cached.with_suffix(".downloading")
    logger.info("Downloading weights from %s …", url)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(cached)
    logger.info("Saved weights to %s", cached)
    return cached


# ---------------------------------------------------------------------------
# H5 weight extraction  (lazy-import h5py so the module loads without it)
# ---------------------------------------------------------------------------


def _extract_layer_weights(group: Any) -> dict[str, np.ndarray]:
    """Recursively collect every ``h5py.Dataset`` under *group*."""
    import h5py

    out: dict[str, np.ndarray] = {}
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            out[key] = item[:]
        elif isinstance(item, h5py.Group):
            for sub_key, arr in _extract_layer_weights(item).items():
                out[f"{key}/{sub_key}"] = arr
    return out


def _extract_all_weights(h5_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Return ``{layer_name: {weight_path: ndarray}}`` from a Keras H5 file."""
    import h5py

    result: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(str(h5_path), "r") as f:
        root = f["model_weights"] if "model_weights" in f else f
        for layer_name in root:
            group = root[layer_name]
            if isinstance(group, h5py.Group):
                result[layer_name] = _extract_layer_weights(group)
    return result


# ---------------------------------------------------------------------------
# Transpose helpers
# ---------------------------------------------------------------------------


def _identity(a: np.ndarray) -> np.ndarray:
    return a


def _conv1d_kernel(a: np.ndarray) -> np.ndarray:
    """TF [kw, in, out] → PyTorch [out, in, kw]."""
    return np.ascontiguousarray(np.transpose(a, [2, 1, 0]))


def _dense_kernel(a: np.ndarray) -> np.ndarray:
    """TF Dense [in, out] → PyTorch Linear [out, in]."""
    return np.ascontiguousarray(a.T)


def _dense_to_conv1d(a: np.ndarray) -> np.ndarray:
    """TF Dense [in, out] → PyTorch Conv1d(ks=1) [out, in, 1]."""
    return np.ascontiguousarray(a.T[:, :, np.newaxis])


def _depthwise_kernel(a: np.ndarray) -> np.ndarray:
    """TF SepConv depthwise [kw, ch, 1] → PyTorch [ch, 1, kw]."""
    return np.ascontiguousarray(np.transpose(a, [1, 2, 0]))


def _pointwise_kernel(a: np.ndarray) -> np.ndarray:
    """TF SepConv pointwise [1, in, out] → PyTorch [out, in, 1]."""
    return np.ascontiguousarray(np.transpose(a, [2, 1, 0]))


# ---------------------------------------------------------------------------
# TF auto-increment name tracker
# ---------------------------------------------------------------------------


class _TFNames:
    """Track Keras auto-incrementing layer names."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def next(self, base: str) -> str:
        idx = self._counts.get(base, 0)
        self._counts[base] = idx + 1
        return base if idx == 0 else f"{base}_{idx}"


# ---------------------------------------------------------------------------
# Mapping helpers  (one per layer type)
# ---------------------------------------------------------------------------


def _bn(m: list[WeightEntry], t: _TFNames, prefix: str) -> None:
    n = t.next("batch_normalization")
    m.extend(
        [
            (n, f"{n}/gamma:0", f"{prefix}.weight", _identity),
            (n, f"{n}/beta:0", f"{prefix}.bias", _identity),
            (n, f"{n}/moving_mean:0", f"{prefix}.running_mean", _identity),
            (n, f"{n}/moving_variance:0", f"{prefix}.running_var", _identity),
        ]
    )


def _ln(m: list[WeightEntry], t: _TFNames, prefix: str) -> None:
    n = t.next("layer_normalization")
    m.extend(
        [
            (n, f"{n}/gamma:0", f"{prefix}.weight", _identity),
            (n, f"{n}/beta:0", f"{prefix}.bias", _identity),
        ]
    )


def _conv(m: list[WeightEntry], t: _TFNames, prefix: str, bias: bool = True) -> None:
    n = t.next("conv1d")
    m.append((n, f"{n}/kernel:0", f"{prefix}.weight", _conv1d_kernel))
    if bias:
        m.append((n, f"{n}/bias:0", f"{prefix}.bias", _identity))


def _dense_linear(
    m: list[WeightEntry],
    t: _TFNames,
    prefix: str,
    bias: bool = True,
) -> None:
    n = t.next("dense")
    m.append((n, f"{n}/kernel:0", f"{prefix}.weight", _dense_kernel))
    if bias:
        m.append((n, f"{n}/bias:0", f"{prefix}.bias", _identity))


def _dense_conv1x1(
    m: list[WeightEntry],
    t: _TFNames,
    prefix: str,
    bias: bool = True,
) -> None:
    """TF Dense that corresponds to a PyTorch Conv1d(kernel_size=1)."""
    n = t.next("dense")
    m.append((n, f"{n}/kernel:0", f"{prefix}.weight", _dense_to_conv1d))
    if bias:
        m.append((n, f"{n}/bias:0", f"{prefix}.bias", _identity))


def _sepconv(
    m: list[WeightEntry],
    t: _TFNames,
    depth_pf: str,
    point_pf: str,
) -> None:
    n = t.next("separable_conv1d")
    m.extend(
        [
            (n, f"{n}/depthwise_kernel:0", f"{depth_pf}.weight", _depthwise_kernel),
            (n, f"{n}/pointwise_kernel:0", f"{point_pf}.weight", _pointwise_kernel),
            (n, f"{n}/bias:0", f"{point_pf}.bias", _identity),
        ]
    )


def _scale(m: list[WeightEntry], t: _TFNames, prefix: str) -> None:
    n = t.next("scale")
    m.append((n, f"{n}/scale:0", f"{prefix}.scale", _identity))


def _mha(m: list[WeightEntry], t: _TFNames, prefix: str) -> None:
    """Baskerville's custom MultiheadAttention."""
    n = t.next("multihead_attention")
    for proj in ("q_layer", "k_layer", "v_layer"):
        m.append(
            (
                n,
                f"{n}/{proj}/kernel:0",
                f"{prefix}.{proj}.weight",
                _dense_kernel,
            )
        )
    m.append(
        (
            n,
            f"{n}/r_k_layer/kernel:0",
            f"{prefix}.r_k_layer.weight",
            _dense_kernel,
        )
    )
    m.append((n, f"{n}/r_w_bias:0", f"{prefix}.r_w_bias", _identity))
    m.append((n, f"{n}/r_r_bias:0", f"{prefix}.r_r_bias", _identity))
    m.append(
        (
            n,
            f"{n}/embedding_layer/kernel:0",
            f"{prefix}.embedding_layer.weight",
            _dense_kernel,
        )
    )
    m.append(
        (
            n,
            f"{n}/embedding_layer/bias:0",
            f"{prefix}.embedding_layer.bias",
            _identity,
        )
    )


# ---------------------------------------------------------------------------
# Full mapping from model config
# ---------------------------------------------------------------------------


def build_weight_mapping(model_cfg: dict[str, Any]) -> list[WeightEntry]:
    """Build ``[(tf_layer, tf_path, pt_key, transpose_fn), …]`` from config."""
    t = _TFNames()
    m: list[WeightEntry] = []
    gnt = model_cfg.get("norm_type")

    trunk_idx = 0
    for bcfg in model_cfg["trunk"]:
        name = bcfg["name"]

        if name == "conv_dna":
            nt = bcfg.get("norm_type", gnt)
            _conv(m, t, f"trunk.{trunk_idx}.conv", bias=(nt is None))
            if nt and nt.startswith("batch"):
                _bn(m, t, f"trunk.{trunk_idx}.norm")

        elif name == "res_tower":
            repeat = int(bcfg.get("repeat", 1))
            num_convs = int(bcfg.get("num_convs", 2))
            nt = bcfg.get("norm_type", gnt)
            for r in range(repeat):
                pf = f"trunk.{trunk_idx}.blocks.{r}"
                if nt and nt.startswith("batch"):
                    _bn(m, t, f"{pf}.first.0")
                _conv(m, t, f"{pf}.first.2")
                for c in range(1, num_convs):
                    ci = c - 1
                    if nt and nt.startswith("batch"):
                        _bn(m, t, f"{pf}.convs.{ci}.0")
                    _conv(m, t, f"{pf}.convs.{ci}.2")
                if num_convs > 1:
                    _scale(m, t, f"{pf}.scale")

        elif name == "transformer_tower":
            repeat = int(bcfg.get("repeat", 1))
            for i in range(repeat):
                pf = f"trunk.{trunk_idx}.layers.{i}"
                _ln(m, t, f"{pf}.norm1")
                _mha(m, t, f"{pf}.mha")
                _ln(m, t, f"{pf}.norm2")
                _dense_linear(m, t, f"{pf}.dense1")
                _dense_linear(m, t, f"{pf}.dense2")

        elif name == "unet_conv":
            nt = bcfg.get("norm_type", gnt)
            uc = bool(bcfg.get("upsample_conv", False))
            pf = f"trunk.{trunk_idx}"
            if nt and nt.startswith("batch"):
                _bn(m, t, f"{pf}.norm_main")
                _bn(m, t, f"{pf}.norm_skip")
            if uc:
                _dense_conv1x1(m, t, f"{pf}.proj_main")
            _dense_conv1x1(m, t, f"{pf}.proj_skip")
            _sepconv(m, t, f"{pf}.sep_depth", f"{pf}.sep_point")

        trunk_idx += 1

    head_key = next((k for k in model_cfg if k.startswith("head")), None)
    if head_key:
        _dense_linear(m, t, "head.proj")

    return m


# ---------------------------------------------------------------------------
# Load weights into a PyTorch model
# ---------------------------------------------------------------------------


def load_tf_weights(
    model: torch.nn.Module,
    h5_path: str | Path,
    model_cfg: dict[str, Any],
    *,
    strict: bool = True,
) -> list[str]:
    """Transfer TF H5 weights into *model*, returning unmapped PyTorch keys."""
    h5_weights = _extract_all_weights(h5_path)
    mapping = build_weight_mapping(model_cfg)
    sd = model.state_dict()

    loaded: set[str] = set()
    errors: list[str] = []

    for tf_layer, tf_path, pt_key, transpose in mapping:
        lw = h5_weights.get(tf_layer)
        if lw is None:
            errors.append(f"H5 layer group '{tf_layer}' missing")
            continue
        if tf_path not in lw:
            errors.append(f"'{tf_path}' missing in layer '{tf_layer}'")
            continue
        if pt_key not in sd:
            errors.append(f"PyTorch key '{pt_key}' not in state_dict")
            continue

        arr = transpose(lw[tf_path])
        expected = tuple(sd[pt_key].shape)
        actual = tuple(arr.shape)
        if expected != actual:
            errors.append(
                f"Shape mismatch: pt '{pt_key}' expects {expected}, "
                f"got {actual} from tf '{tf_layer}/{tf_path}'"
            )
            continue

        sd[pt_key] = torch.from_numpy(arr)
        loaded.add(pt_key)

    if errors and strict:
        raise RuntimeError(f"{len(errors)} weight-loading errors:\n" + "\n".join(errors))

    model.load_state_dict(sd, strict=False)

    unmapped = sorted(k for k in sd if k not in loaded and "num_batches_tracked" not in k)
    if unmapped:
        logger.warning("Unmapped PyTorch keys: %s", unmapped)
    return unmapped
