#!/usr/bin/env python3
"""Build the combined model view for the inkling2 engine (the directory the engine loads).

    scripts/make_native_view.py [--exl3-checkpoint DIR] [--native DIR] [--dense-pack DIR]
                                [--draft-pack DIR] [--view DIR] [--force]

The view is symlinks only (nothing is copied):

  <view>/                     the EXL3 export's files (checkpoint.json, layers/, dense/, tokenizer, ...)
  <view>/dense-exl3           -> the e-dense q6 EXL3 pack (attention, router, head), if given
  <view>/draft-q6             -> the DSpark draft pack, if given
  <view>/native               -> the release's own weights (NVFP4/MXFP8), streamed by the mixed store
  <view>/model.safetensors.index.json   the export's dense map + the pack's trellis tensors
  <view>/VIEW-E-DENSE.json    what was linked, with digests

The engine reads model.safetensors.index.json for the shard list; the EXL3 experts come from layers/ via
checkpoint.json, the dense tensors from dense/ (bf16) or dense-exl3/ (q6 trellises, auto-detected from the index),
and the native bank from native/ when TF_INKLING2_MIX=1.

Standard library only.
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
LINKS = {  # view name -> path in the EXL3 checkpoint
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
OWN = set(LINKS) | {"model.safetensors.index.json", "VIEW-E-DENSE.json", "README.md",
                    "dense-exl3", "draft-q6", "native"}
PACK_FILES = {"attn": "attn.safetensors", "router": "router.safetensors", "head": "head.safetensors"}
SUFFIXES = ("trellis", "suh", "svh", "mul1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exl3-checkpoint", type=Path, default=home / "models/inkling-2-small-exl3-2.59bpw")
    ap.add_argument("--native", type=Path, default=home / "models/inkling-2-small",
                    help="the release's own weights (the mixed store streams from here); skipped if missing")
    ap.add_argument("--dense-pack", type=Path, default=home / "models/inkling-2-small-dense-exl3-q6",
                    help="the e-dense q6 EXL3 pack; skipped if missing (the engine then uses the export's bf16 dense tensors)")
    ap.add_argument("--draft-pack", type=Path, default=home / "models/inkling-2-small-draft-q6",
                    help="the DSpark draft pack; skipped if missing (the draft falls back to bf16 projections)")
    ap.add_argument("--view", type=Path, default=home / "models/inkling-2-small-exl3-view")
    ap.add_argument("--force", action="store_true", help="replace an existing view's own files")
    a = ap.parse_args()

    ck = a.exl3_checkpoint.resolve()
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

    # the index: the export's dense map, plus the pack's trellis tensors when a pack is given
    wm = dict(receipt["weight_map"])
    bad = [f for f in set(wm.values()) if not f.startswith("dense/") or ".." in f]
    if bad:
        print(f"weight_map points outside dense/: {bad[:3]}", file=sys.stderr)
        return 1
    pack = a.dense_pack.resolve() if a.dense_pack else None
    pack_tensors = 0
    if pack and (pack / "manifest.json").is_file():
        for cls, fname in PACK_FILES.items():
            per_file = pack / f"{cls}.safetensors.json"
            if not per_file.is_file():
                print(f"{pack}: missing {per_file.name}", file=sys.stderr)
                return 1
            for base in json.loads(per_file.read_text()):
                for suffix in SUFFIXES:
                    wm[f"{base}.{suffix}"] = f"dense-exl3/{fname}"
                    pack_tensors += 1
        os.symlink(pack, view / "dense-exl3")
    elif pack:
        print(f"note: no dense pack at {pack} (no manifest.json): the engine will use the export's bf16 dense tensors")

    draft = a.draft_pack.resolve() if a.draft_pack else None
    if draft and (draft / "model.safetensors.index.json").is_file():
        os.symlink(draft, view / "draft-q6")
    elif draft:
        print(f"note: no draft pack at {draft} (no model.safetensors.index.json): the draft falls back to bf16 projections")

    native = a.native.resolve() if a.native else None
    if native and (native / "config.json").is_file():
        os.symlink(native, view / "native")
    elif native:
        print(f"note: no native release at {native}: the mixed store (TF_INKLING2_MIX=1) will not work; set MIX_ENV= to boot the EXL3 store alone")

    index = {"metadata": {"source": str(ck), "format": FORMAT,
                          "note": "dense + vision tensors of the EXL3 export; routed experts are read from layers/ by the native inkling2 family's exl3.zig",
                          **({"e-dense": {"pack": str(pack), "tensors": pack_tensors}} if pack_tensors else {})},
             "weight_map": dict(sorted(wm.items()))}
    (view / "model.safetensors.index.json").write_text(json.dumps(index, indent=1) + "\n")

    meta = {"exl3_checkpoint": str(ck), "checkpoint_json_sha256": sha256(ck / "checkpoint.json"),
            "total_bytes": receipt.get("total_bytes"), "format": FORMAT,
            "dense_tensors": len(receipt["weight_map"]), "dense_exl3_tensors": pack_tensors,
            "dense_pack": str(pack) if pack_tensors else None,
            "draft_pack": str(draft) if (view / "draft-q6").is_symlink() else None,
            "native": str(native) if (view / "native").is_symlink() else None,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "generator": str(Path(__file__).resolve())}
    (view / "VIEW-E-DENSE.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"view {view}: {len(LINKS)} links into {ck}, model.safetensors.index.json ({len(wm)} tensors, {pack_tensors} dense-exl3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
