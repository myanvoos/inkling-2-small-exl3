# Inkling-2-Small EXL3 q6 packs

Two optional weight packs for the Inkling-2-Small EXL3 stack. They are **not** in this git repository
(the binaries are ~5 GB); they are published on Hugging Face at
[`myanvoos/Inkling-2-Small-EXL3-Sparks`](https://huggingface.co/myanvoos/Inkling-2-Small-EXL3-Sparks),
alongside the EXL3 export itself. This directory carries everything needed to verify and use them: checksums,
manifests, per-tensor metadata and provenance.

| Pack | Size | Contents |
| --- | --- | --- |
| `dense-exl3-q6/` | 4.29 GB | The non-expert linears — attention q/k/v/r/gate/o, the router, the LM head — as 6-bit EXL3 trellises (mul1), so the engine runs them through its EXL3 path instead of bf16. |
| `draft-q6/` | 0.56 GB | The DSpark draft head (`mtp.safetensors`) as 6-bit EXL3, for the draft's base logits and projections. |

Both are derived works of [`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(Apache-2.0). The dense pack was quantized from the release's bf16 dense tensors; the draft pack from the release's
`mtp.safetensors`.

## Download and verify

```sh
hf download myanvoos/Inkling-2-Small-EXL3-Sparks --local-dir ~/models/inkling-2-small-exl3
cd ~/models/inkling-2-small-exl3/dense-exl3-q6 && sha256sum -c SHA256SUMS
cd ../draft-q6 && sha256sum -c SHA256SUMS
```

`SHA256SUMS` covers every file in the pack, binaries included. This git repository carries only the metadata
(the `.safetensors` binaries are excluded by `.gitignore`), so `sha256sum -c` here reports the binaries as missing;
run it after downloading the pack from Hugging Face.

Then point the view builder at them:

```sh
python3 scripts/make_native_view.py \
  --exl3-checkpoint ~/models/inkling-2-small-exl3/exl3-2.59bpw \
  --native ~/models/inkling-2-small \
  --dense-pack ~/models/inkling-2-small-exl3/dense-exl3-q6 \
  --draft-pack ~/models/inkling-2-small-exl3/draft-q6 \
  --view ~/models/inkling-2-small-exl3-view
```

Both packs are optional. Without the dense pack the engine uses the export's bf16 dense tensors (more memory,
slower); without the draft pack the DSpark draft falls back to bf16 projections.

## What is in each pack

- `*.safetensors` — the weights (not in git; see the HF repo).
- `*.safetensors.json` — per-file tensor metadata (shapes, dtypes).
- `model.safetensors.index.json` — the weight map the engine reads.
- `manifest.json` — per-tensor quantization record (bits, errors, timings).
- `SHA256SUMS` — sha256 of every file in the pack, binaries included (`MD5SUMS` is kept for the draft pack for reference).
- `PROVENANCE.json` (dense pack only) — how the pack was built: calibration data, Hessian capture, quantizer
  settings, per-tensor errors. Local paths are scrubbed; the technical record is complete.

## Publishing (maintainer)

The HF repo holds the EXL3 export (mirrored from kortexa-ai) plus this directory's contents plus the
`.safetensors` binaries:

```sh
hf upload myanvoos/Inkling-2-Small-EXL3-Sparks . .
```

`.gitignore` in this repository excludes `packs/**/*.safetensors` so the binaries cannot be committed by accident.
