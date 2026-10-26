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
re-implements the parts `bonsai2_27b_ternary` does differently and delegates the rest to it:

* `validate` -- same header / Hadamard / tensor contract, tolerating PTQ1_0 and `blk.64.*`;
* `_optional` -- the vision component reads the mmproj through the existing name map;
* `_optional` again -- the MTP head is not ternary, so its `blk.64` rows keep their stored ggml block
  types and are copied the way `qwen3_8_27b_gguf` copies a block GGUF (see `mtp_sources`).

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
        --components text,vision,mtp --name bonsai2-27b-ptq1 --out <artifact>.ninfer

Verified end to end: text tower + vision from the model's own GGUFs, no checkpoint download, 131,072
token KV pool on a 12 GiB card, needle retrieval at 66.5K tokens, vision description correct. The
`mtp` component turns the merged `blk.64` head into `--spec mtp` speculative decoding, the one part
of the release's GGUFs the text+vision artifact used to drop.

Measured here (12 GB GPU, temperature 0, 400 output tokens, 131,072-token pool, 93-token
prompt; `decode` from the engine's own request line, acceptance from `ninfer:draft_*_total`):

    --spec none                                                50.0 tok/s
    --spec mtp --draft-tokens 1                                70.4 tok/s   57.7%
    --spec mtp --draft-tokens 2                                91.1 tok/s   61.8%   <- best
    --spec mtp --draft-tokens 3                                72.8 tok/s   31.4%
    --spec mtp --draft-tokens 4                                68.3 tok/s   24.2%
    --spec mtp --draft-tokens 2 --mtp-attention-window 64      75.2 tok/s   42.0%
    --spec mtp --draft-tokens 4 --adaptive-mtp                 89.9 tok/s   49.2%

Three drafts is where acceptance collapses, so two is the setting to start from; the attention
window costs more acceptance than it saves. The head adds +335 MiB to the artifact and ~0.14 GiB of
runtime state (131,072-token pool: 469 MiB free with `--spec mtp`, 946 MiB without).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

try:
    import tools.convert.ternary as T
    from tools.convert import gguf_blocks as GB
    from tools.convert.methods import cast_direct, grouped_absmax, import_encoded
    from tools.convert.official_recipes import Q4, Q5, Q6, Q8
    from tools.convert.sources.gguf import BLOCK_FORMATS
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


def mtp_tensors() -> dict[str, tuple[tuple[int, ...], str]]:
    """Row-major shape of every `blk.64` tensor this recipe reads ("blocks" = any stored ggml type).

    One block: the same full-attention layer shape as `blk.3` (24 heads x 256 = 6144 query rows and
    a 1024-row KV), a 17408-wide FFN, and the four `nextn.*` tensors of the head itself. The MTP
    block is *not* Hadamard-rotated and *not* ternary -- llama.cpp's `-mtp` merge writes it as plain
    ggml blocks -- which is exactly why `T.expected_tensors()` cannot describe it.
    """

    head, inter = T.HIDDEN, T.INTERMEDIATE
    out = {
        MTP_PREFIX + "attn_norm.weight": ((head,), "F32"),
        MTP_PREFIX + "post_attention_norm.weight": ((head,), "F32"),
        MTP_PREFIX + "ffn_gate.weight": ((inter, head), "blocks"),
        MTP_PREFIX + "ffn_up.weight": ((inter, head), "blocks"),
        MTP_PREFIX + "ffn_down.weight": ((head, inter), "blocks"),
        MTP_PREFIX + "attn_q.weight": ((2 * T.ATTENTION_QUERY_ROWS, head), "blocks"),
        MTP_PREFIX + "attn_k.weight": ((T.ATTENTION_KV_ROWS, head), "blocks"),
        MTP_PREFIX + "attn_v.weight": ((T.ATTENTION_KV_ROWS, head), "blocks"),
        MTP_PREFIX + "attn_output.weight": ((head, T.ATTENTION_QUERY_ROWS), "blocks"),
        MTP_PREFIX + "attn_q_norm.weight": ((T.ATTENTION_HEAD_DIM,), "F32"),
        MTP_PREFIX + "attn_k_norm.weight": ((T.ATTENTION_HEAD_DIM,), "F32"),
        MTP_PREFIX + "nextn.eh_proj.weight": ((head, 2 * head), "blocks"),
        MTP_PREFIX + "nextn.enorm.weight": ((head,), "F32"),
        MTP_PREFIX + "nextn.hnorm.weight": ((head,), "F32"),
        MTP_PREFIX + "nextn.shared_head_norm.weight": ((head,), "F32"),
    }
    return out


def has_mtp(gguf) -> bool:
    return MTP_PREFIX + "nextn.eh_proj.weight" in gguf.tensors


# Read but never stored: the merged head's own token table. `bind_mtp`
# (`src/models/qwen3_5/load/mtp.cpp`) hands the MTP head the text token table, so the artifact only
# ever stores one table. The two are demonstrably different vectors -- sampled rows correlate at
# cosine 0.887 against the un-rotated `token_embd` rows -- so the ~715 MB here is the only weight this
# head carries that neither the engine's layout nor its draft step can use.
MTP_UNUSED = (MTP_PREFIX + "nextn.embed_tokens.weight",)


def validate(gguf) -> None:
    """`T.validate` with PTQ1_0 tensors and the merged MTP block allowed."""

    for key, expected in T.EXPECTED_HEADER.items():
        if key != "qwen35.block_count" and gguf.kv.get(key) != expected:
            raise ValueError(f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}")
    blocks = gguf.kv.get("qwen35.block_count")
    if blocks != T.LAYERS + (1 if has_mtp(gguf) else 0):
        raise ValueError(
            f"{gguf.path}: qwen35.block_count = {blocks!r}, expected {T.LAYERS} text layers"
            f"{' plus the merged MTP layer (65)' if has_mtp(gguf) else ' (64)'}"
        )

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

    merged = mtp_tensors() if has_mtp(gguf) else {}
    allowed = set(expected) | set(merged) | (set(MTP_UNUSED) if merged else set())
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

    for name, (shape, kind) in merged.items():
        info = gguf.tensors[name]
        stored = info.type_id in BLOCK_FORMATS
        if info.shape != shape or (kind == "blocks") != stored or (
            kind != "blocks" and info.type_name != kind
        ):
            raise ValueError(
                f"{gguf.path}: {name} is {info.type_name} {info.shape}, expected "
                f"{'a stored ggml block type' if kind == 'blocks' else kind} {shape}"
            )

    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")


def mtp_sources(gguf):
    """Encoded and direct sources of the merged `blk.64` head, in the ggml types it is stored in.

    `GB._layer` / `GB._attention` are the block-GGUF recipe's row maps: they undo llama.cpp's
    interleaved query/gate order and its `1 + w` norms, and leave every row in the type it was stored
    in (Q6_K for the projections and `eh_proj`, Q8_0 for K/V in this release), so `import_encoded`
    copies bytes instead of re-quantising a 2-bit artifact's drafter.
    """

    g, p = MTP_PREFIX, "mtp/layers/0/"
    encoded, direct = GB._layer(gguf, g, p)
    attention, norms = GB._attention(gguf, g, p + "attention/")
    encoded |= attention
    direct |= norms
    encoded["mtp/input_projection"] = GB.block_source(
        gguf, g + "nextn.eh_proj.weight", (T.HIDDEN, 2 * T.HIDDEN), T.rows()
    )
    for role, tensor in (
        ("embedding_norm", "nextn.enorm.weight"),
        ("hidden_norm", "nextn.hnorm.weight"),
        ("final_norm", "nextn.shared_head_norm.weight"),
    ):
        # `1 + w`, like every other norm PrismML's exporter writes for this model: the draft head's
        # acceptance fell from 61.8% to 29.2% (draft 2) when these three were taken as primal values,
        # so the offset is measured, not assumed.
        direct["mtp/" + role] = T._norm(gguf, g + tensor, True)
    return encoded, direct


def _group_same_format(recipe, names, formats) -> None:
    if len({formats[name] for name in names}) == 1:
        recipe.group(names)


def assign_mtp(model, recipe, gguf) -> None:
    """Assign the MTP head; `bonsai2_27b_ternary` never touches `mtp/*` itself."""

    if not has_mtp(gguf):
        raise ValueError(
            f"{gguf.path}: --components names mtp, but this GGUF carries no merged blk.64 head. Use "
            "the -mtp.gguf build, or leave mtp out of --components"
        )
    encoded, direct = mtp_sources(gguf)
    formats = {}
    for name, source in encoded.items():
        format = source.read_encoded(0, 1).format
        formats[name] = format
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=source,
            activation_policy="AllowA8",
        )
    for name, source in direct.items():
        recipe.assign(
            name,
            format=model.parameters[name].direct_format,
            method=cast_direct,
            source=source,
        )
    # The rule `qwen3_8_27b_gguf` applies to a block GGUF: group the parts of a fused op only when
    # they share a format. Here K/V are Q8_0 beside Q6_K query/gate, so those four stay separate and
    # `mtp_projection` runs them one at a time (`unpacked()` in execution/mtp.cpp); gate/up are both
    # Q6_K, so they group on the half-width input and the pair op still applies.
    a = "mtp/layers/0/attention/"
    _group_same_format(recipe, [a + "query", a + "key", a + "gate", a + "value"], formats)
    _group_same_format(recipe, ["mtp/layers/0/mlp/gate", "mtp/layers/0/mlp/up"], formats)


def make_optional(sources):
    """`_optional` that reads vision from the mmproj GGUF and MTP from the merged `blk.64` head."""

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
            elif name.startswith(("dflash/", "dflash2/")) and parameter.projection:
                raise ValueError(
                    f"{name}: this recipe builds the text, vision and merged-MTP components only; "
                    "drop the drafter from --components"
                )
        if "mtp" in model.components:
            assign_mtp(model, recipe, sources["ternary"])

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
