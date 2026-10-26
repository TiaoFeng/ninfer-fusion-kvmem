#!/usr/bin/env python3
"""One-shot: turn the Bonsai 2 27B GGUFs into a NInfer v3 artifact.

Two steps, no downloads and no HuggingFace checkpoint:

1. `tools/make_hf_config_bonsai27b.py` derives the config-only model directory (geometry + the six
   frontend resources) from the text GGUF and the mmproj metadata;
2. `python -m tools.convert` runs the local recipe (`tools/recipe_bonsai27b_ptq1.py`) with the text
   tower from the ternary GGUF, Vision from the mmproj, and -- when the GGUF is an `-mtp` build -- the
   merged `blk.64` MTP head. That last one is what makes `--spec mtp` speculative decoding possible;
   it is tiny next to the text tower (Q6_K/Q8_0, ~330 MB) because the head's own token table is not
   stored (the engine hands it the text table).

The engine source tree must already be configured for the conversion (Python 3.11 + torch with CUDA:
the converter quantizes on the GPU).

    python tools/convert_bonsai27b_ptq1.py \
        --engine-src  src-tree/fusion-engine-src \
        --text-gguf   ~/Models/Ternary-Bonsai-2-27B-mtp/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
        --mmproj-gguf ~/Models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-mmproj-BF16.gguf \
        --out         ~/Models/Bonsai-2-27B-ninfer/bonsai2_27b.ninfer

Use the BF16 mmproj, not the Q8_0 one: the artifact stores Vision in Q4/Q5/Q6/Q8 (`_vision_format` in
the recipe), so a Q8_0 source is decoded and re-quantized a second time.

The artifact lands next to the work directory (`<out>.work/hf` by default, overridable with
`--work-dir`), which stays reusable: rerunning with `--reuse-config` skips step 1.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_hf_config_bonsai27b as hf_config  # noqa: E402  (sibling script, same directory)

RECIPE = Path(__file__).resolve().parent / "recipe_bonsai27b_ptq1.py"
ENTRY = "bonsai2_27b_ternary"


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> int:
    """Stream a child process so the converter's per-object progress is visible."""

    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Bonsai 2 27B (ternary GGUF + mmproj) into a NInfer v3 artifact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--engine-src", type=Path, required=True,
                        help="engine source root, the directory holding tools/convert/")
    parser.add_argument("--text-gguf", type=Path, required=True,
                        help="Qwen3.8-27B ternary GGUF (PQ2_0 or PTQ1_0; MTP block tolerated)")
    parser.add_argument("--mmproj-gguf", type=Path, required=True,
                        help="llama.cpp mmproj GGUF with the vision tower (BF16 preferred)")
    parser.add_argument("--out", type=Path, required=True, help="artifact path to write")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="where the config-only model directory goes (default <out>.work)")
    parser.add_argument("--reuse-config", action="store_true",
                        help="skip step 1 and use the existing work directory as-is")
    parser.add_argument("--name", default="bonsai2-27b-ptq1", help="artifact metadata name")
    parser.add_argument("--components", default=None,
                        help="comma-separated components to build (default: text,vision,mtp when the "
                             "GGUF carries the merged blk.64 head, else text,vision)")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter holding torch with CUDA (default: the one running this "
                             "script); the converter quantizes on the GPU")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = parser.parse_args(argv)

    engine_src = args.engine_src.resolve()
    if not (engine_src / "tools" / "convert" / "__init__.py").is_file():
        parser.error(f"{engine_src} does not look like the engine source root (no tools/convert/)")
    for path in (args.text_gguf, args.mmproj_gguf):
        if not path.is_file():
            parser.error(f"no such file: {path}")
    if args.out.exists():
        parser.error(f"{args.out} already exists; the converter refuses to overwrite")
    if args.out.suffix != ".ninfer":
        print(f"note: {args.out.name} does not end in .ninfer; the engine does not care", flush=True)

    if args.components is None:
        # The merged head is the only optional component the GGUFs can supply, so pick it whenever it
        # is there: a text+vision artifact silently drops the one weight set llama.cpp's users have
        # that a speculative-decoding engine can use. GGUF header only -- no weights are read.
        sys.path.insert(0, str(engine_src))
        from tools.convert.sources.gguf import GGUFFile

        with GGUFFile(args.text_gguf) as probe:
            merged = "blk.64.nextn.eh_proj.weight" in probe.tensors
        args.components = "text,vision,mtp" if merged else "text,vision"
        print(f"components: {args.components} ({'blk.64 present' if merged else 'no blk.64'})",
              flush=True)

    # The converter quantizes on the GPU, so the interpreter must have torch built against CUDA.
    probe = subprocess.run(
        [args.python, "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        parser.error(
            f"{args.python} cannot import torch ({probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else 'no output'}); "
            "point --python at the environment that has it"
        )
    print(f"python {args.python}: torch {probe.stdout.strip()}", flush=True)

    work = args.work_dir.resolve() if args.work_dir else Path(str(args.out) + ".work")
    hf_dir = work / "hf"

    if args.dry_run:
        print(f"config dir: {hf_dir}")
        print(f"+ python {Path(__file__).name} --text-gguf ... --mmproj-gguf ... --out {hf_dir}")
        print("+ python -m tools.convert --model ... --recipe ...  (cwd=%s)" % engine_src)
        return 0

    if args.reuse_config:
        if not (hf_dir / "config.json").is_file():
            parser.error(f"--reuse-config was given but {hf_dir}/config.json is absent")
        print(f"reusing {hf_dir}")
    else:
        summary = hf_config.generate(args.text_gguf, args.mmproj_gguf, hf_dir)
        vision = summary["vision"]
        print(f"config dir {summary['out']}: vocab {summary['vocab']} + added "
              f"{summary['added']}, vision depth {vision['depth']} "
              f"out {vision['out_hidden_size']}", flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(engine_src)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    command = [
        args.python, "-m", "tools.convert",
        "--model", str(hf_dir),
        "--recipe", f"{RECIPE}:{ENTRY}",
        "--source", f"ternary={args.text_gguf.resolve()}",
        "--source", f"vision={args.mmproj_gguf.resolve()}",
        "--components", args.components,
        "--name", args.name,
        "--out", str(args.out.resolve()),
    ]
    return run(command, cwd=engine_src, env=env)


if __name__ == "__main__":
    sys.exit(main())
