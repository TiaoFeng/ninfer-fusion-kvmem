#!/usr/bin/env python3
"""Generate the config-only HF directory the NInfer converter needs for Bonsai 2 27B.

`tools.convert` takes `--model <checkpoint>` and reads three things from it: the geometry in
`config.json`, six frontend resources, and the weights of any parameter a recipe does not source
elsewhere. `SafetensorsSource` accepts a directory that holds **no weights at all** ("A custom recipe
may supply every logical value from another source"), so a converter run that pulls the text tower
from a GGUF and Vision from an mmproj only needs the geometry and the resources.

This script writes exactly those seven files, derived from the two GGUFs themselves -- nothing is
downloaded and no HuggingFace checkpoint is required:

    config.json                    text tower geometry (qwen35.*) + vision_config (clip.*)
    tokenizer.json                 byte-level BPE built from tokenizer.ggml.tokens / .merges
    tokenizer_config.json          prefix semantics + added_tokens_decoder
    chat_template.jinja            tokenizer.chat_template, verbatim
    generation_config.json         bos / eos / pad ids
    preprocessor_config.json       image pipeline invariants (patch 16, temporal 2, merge 2)
    video_preprocessor_config.json the same plus fps / frame defaults the runtime fixes

Three details are not obvious and each one fails the engine or the converter when wrong:

* added tokens must be **disjoint from `model.vocab`** -- the runtime rejects an overlap -- so the
  special ids leave the vocabulary and come back as `added_tokens`. The union is still the
  contiguous id domain the converter insists on.
* `<think>` and `</think>` must be added tokens with `special: false`. The runtime encodes its
  canonical thinking-control suffix and decodes it with `skip_special_tokens`; a special `</think>`
  vanishes from that check and startup dies with "cannot present the canonical thinking control
  suffix".
* `pre_tokenizer` must be the exact Qwen Split+ByteLevel pair. The converter rewrites a drifted
  Split pattern, but the ByteLevel flags and the `model` invariants still have to hold.

Usage:

    python tools/make_hf_config_bonsai27b.py \
        --text-gguf   ~/Models/Ternary-Bonsai-2-27B-mtp/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
        --mmproj-gguf ~/Models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
        --out         ~/Models/Bonsai-2-27B-ninfer/hf
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# Qwen byte-level BPE split pattern. The runtime validates the stored Regex for exact equality against
# kQwenSplitPattern (src/models/qwen3_5/frontend/tokenizer.cpp); the converter rewrites a drifted one.
QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}


class GGUFFile:
    """Just enough GGUF v3 to read the metadata and the tensor directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open("rb") as handle:
            self._data = handle.read(64 * 1024 * 1024)
        if self._data[:4] != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        self._cursor = 4
        (version,) = self._unpack("I")
        if version != 3:
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        (tensor_count,) = self._unpack("Q")
        (kv_count,) = self._unpack("Q")
        self.kv = {}
        for _ in range(kv_count):
            key = self._string()
            (kind,) = self._unpack("I")
            self.kv[key] = self._value(kind)
        self.tensors = {}
        for _ in range(tensor_count):
            name = self._string()
            (rank,) = self._unpack("I")
            dims = self._unpack("Q" * rank)
            (kind,) = self._unpack("I")
            (offset,) = self._unpack("Q")
            # GGUF lists the fastest dimension first; HF/NInfer shapes are row-major.
            self.tensors[name] = (tuple(reversed(dims)), kind, offset)

    def _unpack(self, fmt: str):
        fmt = "<" + fmt
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, self._data, self._cursor)
        self._cursor += size
        return values

    def _string(self) -> str:
        (length,) = self._unpack("Q")
        raw = self._data[self._cursor : self._cursor + length]
        self._cursor += length
        return raw.decode("utf-8")

    def _value(self, kind: int):
        if kind == 8:
            return self._string()
        if kind == 9:
            (element,) = self._unpack("I")
            (count,) = self._unpack("Q")
            return [self._value(element) for _ in range(count)]
        if kind == 7:
            value = bool(self._data[self._cursor])
            self._cursor += 1
            return value
        if kind in _SCALARS:
            fmt = _SCALARS[kind]
            size = struct.calcsize(fmt)
            (value,) = struct.unpack_from(fmt, self._data, self._cursor)
            self._cursor += size
            return value
        raise ValueError(f"unsupported GGUF value type {kind}")


def text_config(kv: dict) -> dict:
    """The qwen3_5 text geometry the artifact stores, read from the text GGUF's own metadata."""

    head_dim = kv["qwen35.attention.key_length"]
    return {
        "hidden_size": kv["qwen35.embedding_length"],
        "vocab_size": len(kv["tokenizer.ggml.tokens"]),
        "num_hidden_layers": 64,
        "max_position_embeddings": kv["qwen35.context_length"],
        "tie_word_embeddings": False,
        "rms_norm_eps": kv["qwen35.attention.layer_norm_rms_epsilon"],
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "full_attention_interval": kv["qwen35.full_attention_interval"],
        "num_attention_heads": kv["qwen35.attention.head_count"],
        "num_key_value_heads": kv["qwen35.attention.head_count_kv"],
        "head_dim": head_dim,
        "intermediate_size": kv["qwen35.feed_forward_length"],
        "linear_num_key_heads": kv["qwen35.ssm.group_count"],
        "linear_key_head_dim": kv["qwen35.ssm.state_size"],
        "linear_num_value_heads": kv["qwen35.ssm.time_step_rank"],
        "linear_value_head_dim": kv["qwen35.ssm.state_size"],
        "linear_conv_kernel_dim": kv["qwen35.ssm.conv_kernel"],
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": kv["qwen35.rope.freq_base"],
            "partial_rotary_factor": kv["qwen35.rope.dimension_count"] / head_dim,
            "mrope_interleaved": True,
            "mrope_section": [
                int(v) for v in kv["qwen35.rope.dimension_sections"] if int(v) > 0
            ],
        },
    }


def vision_config(clip: dict, positions: int, output_width: int) -> dict:
    """The vision tower geometry, read from the mmproj's `clip.*` metadata.

    `temporal_patch_size` is not in the metadata; the exporter splits the Conv3d kernel into
    `v.patch_embd.weight` and `.weight.1`, which only happens for a temporal depth of two, and the
    name map refuses any other value.
    """

    return {
        "in_channels": 3,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [],
        "out_hidden_size": output_width,
        "depth": clip["clip.vision.block_count"],
        "hidden_size": clip["clip.vision.embedding_length"],
        "intermediate_size": clip["clip.vision.feed_forward_length"],
        "num_heads": clip["clip.vision.attention.head_count"],
        "patch_size": clip["clip.vision.patch_size"],
        "temporal_patch_size": 2,
        "spatial_merge_size": clip["clip.vision.spatial_merge_size"],
        "num_position_embeddings": positions,
    }


def generate(text_gguf: Path, mmproj_gguf: Path, out_dir: Path) -> dict:
    """Write the seven files and return a summary of what was derived."""

    text = GGUFFile(text_gguf)
    kv = text.kv
    mmproj = GGUFFile(mmproj_gguf)
    clip = mmproj.kv
    if "v.position_embd.weight" not in mmproj.tensors:
        raise ValueError(f"{mmproj_gguf}: no v.position_embd.weight; is this a Qwen mmproj?")
    # (num_position_embeddings, hidden) in HF order; the grid count is the long axis.
    positions = max(mmproj.tensors["v.position_embd.weight"][0])

    tokens = kv["tokenizer.ggml.tokens"]
    token_types = kv["tokenizer.ggml.token_type"]
    cfg_text = text_config(kv)
    cfg_vision = vision_config(clip, positions, cfg_text["hidden_size"])
    if cfg_vision["out_hidden_size"] != cfg_text["hidden_size"]:
        raise ValueError(
            f"{mmproj_gguf}: projection_dim {cfg_vision['out_hidden_size']} is not the text width "
            f"{cfg_text['hidden_size']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": cfg_text,
                "vision_config": cfg_vision,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # CONTROL (3) and USER_DEFINED (4) become added tokens; everything else stays in the vocabulary.
    reasoning = {i for i, token in enumerate(tokens) if token in ("<think>", "</think>")}
    special_ids = [i for i, kind in enumerate(token_types) if kind in (3, 4)]
    special_id_set = set(special_ids)
    vocab = {token: i for i, token in enumerate(tokens) if i not in special_id_set}
    added = [
        {
            "id": i,
            "content": tokens[i],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            # See the module docstring: the reasoning delimiters must survive skip_special_tokens.
            "special": i not in reasoning,
        }
        for i in special_ids
    ]
    merges = [pair.split(" ", 1) for pair in kv["tokenizer.ggml.merges"]]
    broken = [pair for pair in merges if len(pair) != 2]
    if broken:
        raise ValueError(f"{text_gguf}: {len(broken)} merges are not symbol pairs, e.g. {broken[:3]}")

    (out_dir / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": added,
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {
                    "type": "Sequence",
                    "pretokenizers": [
                        {
                            "type": "Split",
                            "pattern": {"Regex": QWEN_SPLIT},
                            "behavior": "Isolated",
                            "invert": False,
                        },
                        {
                            "type": "ByteLevel",
                            "add_prefix_space": False,
                            "trim_offsets": True,
                            "use_regex": False,
                        },
                    ],
                },
                "post_processor": None,
                "decoder": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
                "model": {
                    "type": "BPE",
                    "dropout": None,
                    "unk_token": None,
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": None,
                    "fuse_unk": False,
                    "byte_fallback": False,
                    "ignore_merges": False,
                    "vocab": vocab,
                    "merges": merges,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    template = kv["tokenizer.chat_template"]
    (out_dir / "chat_template.jinja").write_text(template, encoding="utf-8")

    decoder = {
        str(item["id"]): {
            "content": item["content"],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": item["special"],
        }
        for item in added
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "add_bos_token": False,
                "add_prefix_space": False,
                "bos_token": "<|endoftext|>",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "tokenizer_class": "Qwen2Tokenizer",
                "model_max_length": kv["qwen35.context_length"],
                "chat_template": template,
                "added_tokens_decoder": decoder,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": kv["tokenizer.ggml.bos_token_id"],
                "eos_token_id": [kv["tokenizer.ggml.eos_token_id"]],
                "pad_token_id": kv["tokenizer.ggml.padding_token_id"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # The compiled pixel pipeline fixes these: bicubic resample, 1/255 rescale, mean/std 0.5.
    shortest, longest = 32 * 32 * 4, 12845056
    pixels = {
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "do_convert_rgb": True,
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "size": {"shortest_edge": shortest, "longest_edge": longest},
        "patch_size": cfg_vision["patch_size"],
        "temporal_patch_size": cfg_vision["temporal_patch_size"],
        "merge_size": cfg_vision["spatial_merge_size"],
        "min_pixels": shortest,
        "max_pixels": longest,
        "processor_class": "Qwen3VLProcessor",
    }
    (out_dir / "preprocessor_config.json").write_text(
        json.dumps(pixels, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "video_preprocessor_config.json").write_text(
        json.dumps({**pixels, "fps": 2.0, "min_frames": 4, "max_frames": 768}, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "out": str(out_dir),
        "vocab": len(vocab),
        "added": len(added),
        "reasoning_plain": sorted(reasoning),
        "merges": len(merges),
        "vision": cfg_vision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the config-only HF directory for the Bonsai 2 27B NInfer conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text-gguf", type=Path, required=True,
                        help="Qwen3.8-27B ternary GGUF (PQ2_0 or PTQ1_0, MTP block allowed)")
    parser.add_argument("--mmproj-gguf", type=Path, required=True,
                        help="llama.cpp mmproj GGUF holding the vision tower (BF16 preferred)")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory to write; pass it to tools.convert --model")
    args = parser.parse_args(argv)

    for path in (args.text_gguf, args.mmproj_gguf):
        if not path.is_file():
            parser.error(f"no such file: {path}")

    summary = generate(args.text_gguf, args.mmproj_gguf, args.out)
    vision = summary["vision"]
    print(f"wrote {summary['out']}")
    print(f"  vocab {summary['vocab']} + added {summary['added']} "
          f"= {summary['vocab'] + summary['added']} ids, merges {summary['merges']}")
    print(f"  plain added tokens (survive skip_special_tokens): {summary['reasoning_plain']}")
    print(f"  vision depth {vision['depth']} hidden {vision['hidden_size']} "
          f"positions {vision['num_position_embeddings']} out {vision['out_hidden_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
