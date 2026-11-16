#!/usr/bin/env python3
"""Build the native engine's view of the kortexa-ai Inkling-2-Small EXL3 export (the named-view pattern).

    scripts/make_view.py [--checkpoint DIR] [--view DIR] [--force]

The export keeps config.json, hf_quant_config.json, the tokenizer, the chat template and mtp.safetensors in
auxiliary/, and its dense tensors in dense/*.safetensors indexed by checkpoint.json's weight_map. The native inkling2
family reads a model folder (config.json + hf_quant_config.json + model.safetensors.index.json + mtp.safetensors at
the top; checkpoint.json present = EXL3 experts, read by exl3.zig from layers/). The view is that folder: symlinks into
the checkpoint (never a copy, never a write to it) and one generated file, model.safetensors.index.json (the dense and
vision weight map with paths under dense/). VIEW.json records the source and its checkpoint.json digest.

Refuses a checkpoint whose checkpoint.json is not complete or not format kortexa-inkling2-exl3-v1, or a view
directory that holds anything but its own files (unless --force rebuilds it). Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

FORMAT = "kortexa-inkling2-exl3-v1"
LINKS = {  # view name -> path in the checkpoint
    "config.json": "auxiliary/config.json",
    "hf_quant_config.json": "auxiliary/hf_quant_config.json",
    "tokenizer.json": "auxiliary/tokenizer.json",
    "tokenizer_config.json": "auxiliary/tokenizer_config.json",
    "special_tokens_map.json": "auxiliary/special_tokens_map.json",
    "chat_template.jinja": "auxiliary/chat_template.jinja",
    "processor_config.json": "auxiliary/processor_config.json",
    "mtp.safetensors": "auxiliary/mtp.safetensors",
    "checkpoint.json": "checkpoint.json",
    "dense": "dense",
    "layers": "layers",
}
OWN = set(LINKS) | {"model.safetensors.index.json", "VIEW.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=home / "models/inkling-2-small-exl3-2.59bpw")
    ap.add_argument("--view", type=Path, default=home / "inference/artifacts/views/inkling-2-small-exl3-2.59bpw")
    ap.add_argument("--force", action="store_true", help="replace an existing view's own files")
    a = ap.parse_args()
    ck = a.checkpoint.resolve()
    receipt = json.loads((ck / "checkpoint.json").read_text())
    if receipt.get("format") != FORMAT or receipt.get("status") != "complete":
        print(f"{ck}: checkpoint.json is not a complete {FORMAT} export", file=sys.stderr)
        return 1
    for name, rel in LINKS.items():
        if not (ck / rel).exists():
            print(f"{ck}: missing {rel}", file=sys.stderr)
            return 1
    view = a.view
    view.mkdir(parents=True, exist_ok=True)
    others = [p.name for p in view.iterdir() if p.name not in OWN]
    if others:
        print(f"{view} holds files that are not the view's: {others}", file=sys.stderr)
        return 1
    existing = [p.name for p in view.iterdir()]
    if existing and not a.force:
        print(f"{view} exists ({len(existing)} entries); --force rebuilds it", file=sys.stderr)
        return 1
    for p in view.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()
    for name, rel in LINKS.items():
        os.symlink(ck / rel, view / name)
    wm = receipt["weight_map"]
    bad = [f for f in set(wm.values()) if not f.startswith("dense/") or ".." in f]
    if bad:
        print(f"weight_map points outside dense/: {bad[:3]}", file=sys.stderr)
        return 1
    index = {"metadata": {"source": str(ck), "format": FORMAT, "note": "dense + vision tensors of the EXL3 export; "
                          "routed experts are read from layers/ by the native inkling2 family's exl3.zig"},
             "weight_map": dict(sorted(wm.items()))}
    (view / "model.safetensors.index.json").write_text(json.dumps(index, indent=1) + "\n")
    meta = {"checkpoint": str(ck), "checkpoint_json_sha256": sha256(ck / "checkpoint.json"),
            "total_bytes": receipt.get("total_bytes"), "format": FORMAT, "dense_tensors": len(wm),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generator": str(Path(__file__).resolve())}
    (view / "VIEW.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"view {view}: {len(LINKS)} links into {ck}, model.safetensors.index.json ({len(wm)} tensors), VIEW.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
