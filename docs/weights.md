# Weights: what you need and where it comes from

## Public (download from Hugging Face)

### myanvoos/Inkling-2-Small-EXL3-Sparks (~199 GB)

One repo with everything except the native release:

- `exl3-2.59bpw/` — the EXL3 export, mirrored unchanged from
  [`kortexa-ai/Inkling-2-Small-EXL3-2.59bpw`](https://huggingface.co/kortexa-ai/Inkling-2-Small-EXL3-2.59bpw)
  (revision `05c6cd0dff9df5bf20b10419c663a1e2adc7eb64`). Every routed expert as an ExLlamaV3 trellis (mul1 codebook;
  layers 0 and 40–47 at 3 bits, layers 1–39 at 3 bits for the 192 most-routed experts and 2 bits for the rest;
  2.59375 bits mean), plus the release's dense tensors in `dense/` (bf16), the tokenizer, the chat template and
  `mtp.safetensors` (the DSpark draft). Format `kortexa-inkling2-exl3-v1`; `checkpoint.json` is the manifest the
  engine reads. The export's format reference and the original two-node Python runner live at
  [kortexa-ai/inkling2-exl3-spark](https://github.com/kortexa-ai/inkling2-exl3-spark).
- `dense-exl3-q6/` and `draft-q6/` — the two q6 packs (below).

```sh
hf download myanvoos/Inkling-2-Small-EXL3-Sparks --local-dir ~/models/inkling-2-small-exl3
```

### thinkingmachines/Inkling-2-Small (~401 GB)

The native release: NVFP4 experts (layers 0–39), MXFP8 experts (layers 40–47), bf16 dense. The mixed store
(`TF_INKLING2_MIX=1`, the default) streams hot experts from here as a bank alongside the resident EXL3 store.
Only needed for the mixed store; set `MIX_ENV=` to boot the EXL3 store alone.

Pinned revision: `b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b`.

## The q6 packs

Both packs are in the same repo, under `dense-exl3-q6/` and `draft-q6/`.
Checksums, manifests and provenance are in `packs/` in this repository; verify a download with
`sha256sum -c SHA256SUMS` in each pack directory.

### Dense q6 EXL3 pack — `dense-exl3-q6` (4.29 GB)

The non-expert linears (attention q/k/v/r/gate/o, the router, the LM head) as 6-bit EXL3 trellises, so the engine
can run them through its EXL3 path instead of bf16. Built by the e-dense pipeline:

1. Hessian capture: `tf-inkling2-run --hcap` over 262,144 calibration tokens (128 sequences × 2048), on the EXL3
   store at TP2, bf16 operands with fp32 sums, per dense linear's input.
2. Quantization: ExLlamaV3 `quantize_exl3` (a fork of exllamav3, commit `33da09b815a2d39ca58d5c225ca6374eec2aa03d`),
   K=6, mul1, `sigma_reg=0.025`, seed = 2 + layer index (51 for the head), `NO_SWAP=1`, one Hessian per
   (layer, input) shared by the tensors that read it.
3. Calibration mix: 128 rows × 2048 cols; c4 16, code 18, multilingual 8, technical 8, wiki 36, tiny 4, random 8,
   chat 30; token ids sha256 `8b4ffa84691f21c485ad6a51c48b7bd5a94fcc49f865a45f7bafb030e`.

Full provenance (per-tensor errors, Hessian logs, tool paths) is in the pack's `PROVENANCE.json` (in `packs/dense-exl3-q6/`).
The published `manifest.json` is path-scrubbed; its sha256 is in `packs/dense-exl3-q6/SHA256SUMS`.

**Without it**, the engine uses the export's bf16 dense tensors (the default when the view has no `dense-exl3/`).
That works; it costs ~10 GB more memory per node and is slower on the dense path.

### Draft q6 pack — `draft-q6` (0.56 GB)

The DSpark draft head (`mtp.safetensors`) as 6-bit EXL3, for the draft's base logits and projections.
**Without it**, the draft falls back to bf16 projections (`TF_INKLING2_DRAFT_Q6` unset / no `draft-q6/` in the view).

### Rebuilding the packs

The packs were built with the e-dense pipeline, which is not part of this repository (it depends on the
calibration corpus). To rebuild them, the pipeline above is the recipe (a multi-hour, two-GPU job). Otherwise
download them from the Hugging Face repo above, or run without them (see above).

## The view

`scripts/make_native_view.py` builds the directory the engine loads: symlinks into the EXL3 export, plus
`dense-exl3/` → the dense pack, `draft-q6/` → the draft pack, `native/` → the release, and a generated
`model.safetensors.index.json` (the export's dense map plus the pack's trellis tensors). Nothing is copied; the
view is a few KB of symlinks. The engine opens every shard named in the index at load, which is why the load
opens ~2,358 files and needs `NOFILE=500000`.

## `plan-gamma-scores.f32`

`config/plan-gamma-scores.f32` (73 KB) is the prior the mixed store uses to pick which experts to keep resident
(`TF_INKLING2_PRIOR`). It is derived from routing statistics over the calibration corpus. It is included in this
repository.
