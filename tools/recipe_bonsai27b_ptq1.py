"""NInfer conversion recipe: ternary Bonsai 2 27B from a llama.cpp PTQ1_0 GGUF + an mmproj.

The stock `bonsai2_27b_ternary` recipe (`src-tree/fusion-engine-src/tools/convert/ternary.py`) expects
PrismML's **PQ2_0** GGUF -- every ternary tensor typed `PQ2_0`, `qwen35.block_count = 64` -- and takes
Vision from a Qwen3.8-27B safetensors checkpoint. Neither holds for the files llama.cpp users have:

* the released GGUF carries **PTQ1_0** (ggml type 143), and the MTP-merged variant has one extra block
  `blk.64` holding 16 MTP tensors in Q6_K / Q4_K / Q8_0;
* the only vision weights available are llama.cpp's `mmproj-*.gguf`.

Both gaps are converter-side only: the engine already ships PTQ1_0 decode paths
(`src/ops/linear/t2/t2_ptq1_*`, `tools/convert/sources/gguf.py::decode_ptq1_0`) and an mmproj name map
(`qwen35_vision_name_map` + `HFAliasSource`, used by the `qwen3_8_27b_gguf` recipe). This module
re-implements the two pieces `bonsai2_27b_ternary` does differently and delegates the rest to it:

* `validate` -- same header / Hadamard / tensor contract, tolerating PTQ1_0 and `blk.64.*`;
* `_optional` -- the vision component reads the mmproj through the existing name map.

`pipeline.convert` never passes `identity` (which the v3 directory requires and `schema.py` enforces),
so `_install_identity` swaps in an `ArtifactWriter` subclass that supplies `{model_id, weights_id}`.
`convert()` looks the name up in `tools.convert.pipeline` at call time, so the patch takes effect.

Run it through `tools/convert_bonsai27b_ptq1.py`, or by hand with the engine source root on
`sys.path` and this file given as a path (a bare `module:function` name only resolves for the recipes
registered in the engine's own `RECIPES` dicts):

    cd <engine-src> && PYTHONPATH=. python -m tools.convert \
        --model <config-only dir from make_hf_config_bonsai27b.py> \
        --recipe <this file>:bonsai2_27b_ternary \
        --source ternary=<text>.gguf --source vision=<mmproj>.gguf \
        --components text,vision --name bonsai2-27b-ptq1 --out <artifact>.ninfer

Verified end to end: text tower + vision from the model's own GGUFs, no checkpoint download, 131,072
token KV pool on a 12 GiB card, needle retrieval at 66.5K tokens, vision description correct.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

try:
    import tools.convert.ternary as T
    from tools.convert.methods import cast_direct, grouped_absmax
    from tools.convert.official_recipes import Q4, Q5, Q6, Q8
    from tools.convert.sources.gguf_source import (
        GGUFSource,
        HFAliasSource,
        qwen35_vision_name_map,
    )
except ModuleNotFoundError as error:  # pragma: no cover - guidance beats a bare traceback
    raise SystemExit(
        f"cannot import the engine converter ({error}). Put the engine source root on PYTHONPATH "
        "(the directory that contains `tools/convert/`), e.g. run the converter from that root with "
        "`PYTHONPATH=.`, or use tools/convert_bonsai27b_ptq1.py which does it for you."
    ) from error

TERNARY_KINDS = ("PQ2_0", "PTQ1_0")
MTP_PREFIX = "blk.64."


def vision_format(name: str) -> str:
    """Mirror `gguf_blocks._vision_format`: the artifact's packed vision formats."""

    if name == "vision/patch_embedding":
        return Q6
    if name.startswith("vision/merger/"):
        return Q8
    if name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
        return Q4
    return Q5


def validate(gguf) -> None:
    """`T.validate` with PTQ1_0 tensors and the merged MTP block allowed."""

    header = dict(T.EXPECTED_HEADER)
    header["qwen35.block_count"] = 65  # 64 text layers + the merged MTP layer
    for key, expected in header.items():
        if gguf.kv.get(key) != expected:
            raise ValueError(f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}")

    widths = tuple(int(width) for width in gguf.kv.get("prism.hadamard.sign_widths", ()))
    if widths != T.SIGN_WIDTHS:
        raise ValueError(f"{gguf.path}: sign widths {widths}, expected {T.SIGN_WIDTHS}")
    values = np.asarray(gguf.kv.get("prism.hadamard.sign_values", ()), dtype=np.int64)
    if values.size != sum(T.SIGN_WIDTHS) or not np.all(np.abs(values) == 1):
        raise ValueError(f"{gguf.path}: sign values must be {sum(T.SIGN_WIDTHS)} entries of +-1")

    expected = T.expected_tensors()
    rotated = {name for name, (_, kind) in expected.items() if kind == "PQ2_0"} - {
        "token_embd.weight"
    }
    if set(gguf.kv.get("prism.hadamard.weight_names", ())) != rotated:
        raise ValueError(f"{gguf.path}: the rotated tensor set differs from the registered one")
    if tuple(gguf.kv.get("prism.hadamard.inverse_weight_names", ())) != ("token_embd.weight",):
        raise ValueError(f"{gguf.path}: only token_embd.weight may be stored inverse-rotated")

    allowed = set(expected) | {name for name in gguf.tensors if name.startswith(MTP_PREFIX)}
    if set(gguf.tensors) != allowed:
        missing = sorted(allowed - set(gguf.tensors))[:5]
        extra = sorted(set(gguf.tensors) - allowed)[:5]
        raise ValueError(f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})")

    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        accepted = TERNARY_KINDS if kind == "PQ2_0" else (kind,)
        if info.shape != shape or info.type_name not in accepted:
            raise ValueError(
                f"{gguf.path}: {name} is {info.type_name} {info.shape}, expected "
                f"{' or '.join(accepted)} {shape}"
            )

    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")


def make_optional(sources):
    """`_optional` that reads vision from the mmproj GGUF instead of the checkpoint."""

    def optional(model, recipe, *, method=grouped_absmax):
        vision = None
        if "vision" in model.components:
            mmproj = sources["vision"]
            if isinstance(mmproj, GGUFSource):
                config = model.components["vision"]["config"]
                names, transforms, shapes, composites = qwen35_vision_name_map(
                    config["depth"],
                    hidden_size=config["hidden_size"],
                    in_channels=config.get("in_channels", 3),
                    temporal_patch_size=config["temporal_patch_size"],
                    patch_size=config["patch_size"],
                )
                vision = HFAliasSource(
                    mmproj, names, transforms=transforms, shapes=shapes, composites=composites
                )
            else:
                vision = mmproj
        for name, parameter in model.parameters.items():
            if name.startswith("vision/"):
                source = model.source(name, vision)
                if parameter.projection:
                    recipe.assign(
                        name, format=vision_format(name), method=grouped_absmax, source=source
                    )
                else:
                    recipe.assign(
                        name, format=parameter.direct_format, method=cast_direct, source=source
                    )
            elif name.startswith(("mtp/", "dflash/", "dflash2/")) and parameter.projection:
                raise ValueError(
                    f"{name}: this recipe builds the text and vision components only; the merged "
                    "blk.64 MTP head is not wired yet (drop it from --components)"
                )

    return optional


def _weights_id(gguf) -> str:
    """A stable digest of the source GGUF: its first 8 MiB plus the data-section size."""

    path = Path(gguf.path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(8 * 1024 * 1024))
    digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def _install_identity(sources) -> None:
    """`pipeline.convert` never passes `identity`, which the v3 directory requires."""

    import tools.convert.pipeline as pipeline

    current = pipeline.ArtifactWriter
    if getattr(current, "_identity_injected", False):
        return
    identity = {
        "model_id": "bonsai2-27b-ptq1",
        "weights_id": _weights_id(sources["ternary"]),
    }

    class IdentifiedWriter(current):
        _identity_injected = True

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("identity", dict(identity))
            super().__init__(*args, **kwargs)

    pipeline.ArtifactWriter = IdentifiedWriter


def bonsai2_27b_ternary(model, recipe, sources):
    """Entry point for `--recipe <this file>:bonsai2_27b_ternary`."""

    T.validate = validate
    T._optional = make_optional(sources)
    _install_identity(sources)
    return T.bonsai2_27b_ternary(model, recipe, sources)


if __name__ == "__main__":  # pragma: no cover - the entry point is called by tools.convert
    sys.exit(
        "This file is a converter recipe, not a program: pass it as "
        "--recipe <path>:bonsai2_27b_ternary to `python -m tools.convert`, or run "
        "tools/convert_bonsai27b_ptq1.py."
    )
