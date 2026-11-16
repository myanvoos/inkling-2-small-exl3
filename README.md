# Inkling-2-Small on two DGX Sparks — EXL3 2.59bpw

Run [Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (Thinking Machines' 594B-parameter
MoE, ~17B active) as an OpenAI/Anthropic-compatible API on two NVIDIA DGX Sparks (GB10). The engine is the native
TensorFold inkling2 server; the weights are the [kortexa-ai EXL3 2.59bpw export](https://huggingface.co/kortexa-ai/Inkling-2-Small-EXL3-2.59bpw)
with the release's own NVFP4/MXFP8 experts streamed as a hot-expert bank.

This is the stack that serves the model in production on the author's two Sparks, extracted so it runs standalone.

```
client ──HTTP──> inkling gateway (:8080, Anthropic + OpenAI APIs)
                    │
                    ▼
              TensorFold inkling2 server (rank 0, 127.0.0.1:8888)
                    │  TP=2 over RoCE (NCCL)
                    ▼
              TensorFold inkling2 server (rank 1, worker :8889)
```

- **Engine**: TensorFold (`engine/`, Apache-2.0), the native Zig server with the `inkling2` family and the EXL3 lane.
  Vendored from the tree the deployed binary was built from (v1.2.7 + the unreleased inkling2 work).
- **Gateway**: `gateway/`, a stdlib-only Python Anthropic/OpenAI gateway. Parses Inkling's tool-call markup into
  OpenAI `tool_calls` / Anthropic `tool_use`, keeps reasoning always on, and keeps a per-turn reasoning memory so
  clients that drop reasoning still get prompt-cache reuse.
- **Weights**: the EXL3 export (all routed experts resident, 2.59 bits/weight) + the release's native NVFP4/MXFP8
  experts streamed as a hot bank + a q6 EXL3 pack of the dense tensors (attention/router/head).
- **Context**: 500,000 tokens. Reasoning is always on (template C). One stream at a time; concurrent requests queue.

## Requirements

- Two NVIDIA DGX Sparks (GB10, 128 GB unified memory each), or equivalent aarch64 + CUDA 13 GPUs with ~123 GiB each.
- CUDA 13 toolkit (`nvcc`) and Zig **0.17.0** (`engine/.zig-version`) to build the engine.
- Python 3.10+ (gateway and scripts; stdlib only), `rsync`, `ssh` (passwordless, `BatchMode`) between the nodes,
  `systemd --user` on both, Docker on the head node (for the gateway container; the gateway also runs directly with `python3`).
- Disk: ~200 GB per node for the EXL3 export, plus ~400 GB per node for the native release (the mixed store, on by default), plus ~5 GB for the packs.
- Both nodes' clocks roughly in sync; the worker reachable as `WORKER_SSH` from the head.

## Weights

Everything except the native release is in one repo:
[`myanvoos/Inkling-2-Small-EXL3-Sparks`](https://huggingface.co/myanvoos/Inkling-2-Small-EXL3-Sparks) —
the EXL3 export (mirrored from kortexa-ai), the dense q6 pack and the draft q6 pack.

| What | Where | Size |
| --- | --- | --- |
| EXL3 2.59bpw export (experts + bf16 dense) | `myanvoos/Inkling-2-Small-EXL3-Sparks` → `exl3-2.59bpw/` (mirror of [`kortexa-ai/Inkling-2-Small-EXL3-2.59bpw`](https://huggingface.co/kortexa-ai/Inkling-2-Small-EXL3-2.59bpw), revision `05c6cd0dff9df5bf20b10419c663a1e2adc7eb64`) | ~194 GB |
| Dense q6 EXL3 pack (attention/router/head) | `myanvoos/Inkling-2-Small-EXL3-Sparks` → `dense-exl3-q6/` | 4.3 GB |
| Draft q6 pack (DSpark draft head) | `myanvoos/Inkling-2-Small-EXL3-Sparks` → `draft-q6/` | 0.6 GB |
| Native release (NVFP4/MXFP8 experts, bf16 dense) | [`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small), revision `b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b` | ~401 GB |
| `plan-gamma-scores.f32` (mixed-store prior) | `config/plan-gamma-scores.f32` | 73 KB |

Download on **both** nodes (each rank reads its own shards):

```sh
# on each node
hf download myanvoos/Inkling-2-Small-EXL3-Sparks --local-dir ~/models/inkling-2-small-exl3
hf download thinkingmachines/Inkling-2-Small --revision b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b \
  --local-dir ~/models/inkling-2-small
```

That gives you `~/models/inkling-2-small-exl3/{exl3-2.59bpw,dense-exl3-q6,draft-q6}`. Checksums for the packs are in
`packs/` (`sha256sum -c SHA256SUMS` in each pack directory).

The dense and draft packs are optional. Without the dense pack the engine uses the export's bf16 dense tensors
(more memory, slower); without the draft pack the DSpark draft falls back to bf16 projections. The mixed store
(`MIX_ENV`) needs the native release; set `MIX_ENV=` to boot the EXL3 store alone.

## Quickstart

On the head node, from the repo root:

```sh
# 1. build and install the engine on both nodes (needs zig 0.17.0 + nvcc)
scripts/install_engine.sh

# 2. build the model view (symlinks; nothing is copied) — on BOTH nodes: rank 1 reads the same
#    view on the worker, so run the same command there too (or rsync the view over)
python3 scripts/make_native_view.py \
  --exl3-checkpoint ~/models/inkling-2-small-exl3/exl3-2.59bpw \
  --native ~/models/inkling-2-small \
  --dense-pack ~/models/inkling-2-small-exl3/dense-exl3-q6 \
  --draft-pack ~/models/inkling-2-small-exl3/draft-q6 \
  --view ~/models/inkling-2-small-exl3-view
#    ... and on the worker (rank 1 reads the same view there; the script is not copied by serve.sh):
scp scripts/make_native_view.py <worker>:/tmp/ && ssh <worker> \
  "python3 /tmp/make_native_view.py --exl3-checkpoint ~/models/inkling-2-small-exl3/exl3-2.59bpw \
     --native ~/models/inkling-2-small \
     --dense-pack ~/models/inkling-2-small-exl3/dense-exl3-q6 \
     --draft-pack ~/models/inkling-2-small-exl3/draft-q6 \
     --view ~/models/inkling-2-small-exl3-view"

# 3. point the config at your nodes and paths
$EDITOR config/inkling.env     # HEAD_IP, WORKER_SSH, MODEL_VIEW, NCCL_* for your fabric

# 4. serve (preflight, GPU pair lock, both ranks, gateway)
scripts/serve.sh start

# 5. check
scripts/serve.sh status
python3 tests/live/verify_gateway.py
```

The gateway listens on `:8080` and serves the model as `inkling-2-small` (display name `Inkling-2-Small`).
`inkling` is answered but not listed. Point any OpenAI- or Anthropic-compatible client at
`http://<head>:8080/v1`.

## What runs where

| Piece | Where | Notes |
| --- | --- | --- |
| `tensorfold-native` rank 0 | head, `127.0.0.1:8888`, systemd user unit `inkling-serve` | the engine; holds the GPU pair lock |
| `tensorfold-native` rank 1 | worker, `:8889`, systemd user unit `inkling-serve-r1` | TP=2 peer over NCCL/RoCE |
| `inkling-gateway` | head, `:8080`, docker container | Anthropic + OpenAI APIs; `--family inkling2` |
| page-cache droppers | both nodes, `inkling-serve-dropcache[-r1]` | drop file cache every 30 s above 512 MiB |

`scripts/serve.sh` manages all of it: `start | stop | restart | status | logs | gateway | check | dropcache`.
`check` runs the preflight read-only. The serve holds a GPU pair lock (`$GPU_LOCK_DIR`, default
`/var/tmp/inkling-gpu-lock`) for its whole lifetime so no other GPU job starts beside it; on a shared machine,
point `GPU_LOCK_DIR` at the shared lock directory.

## Performance

Measured from the live deployment, 2026-11-17. The engine logs one `[tensorfold] done` line per request with
`ttft`, `decode`, `queued` and `tok/s` (`logs/r0-*.log`); the numbers below are the real traffic of one long
session (prompts 3.5k–286k tokens, ~99.7% of each prompt served from the prefix cache, 293 requests after
excluding two short synthetic probes), not a synthetic benchmark.

- **Prefill: 1300–1600 tok/s** (uncached tokens ÷ TTFT). Healthy reads sit at or above ~1200 tok/s; a linear
  fit over the day gives ~1370 tok/s plus ~0.7 s of fixed per-request overhead. Sustained readings **below
  1000 tok/s are not the norm** — check the page cache and the streamed native bank before assuming a
  regression.
- **Decode: ~47 tok/s median** (mean ~50, p90 ~66, max ~77). The verifier checks ~6.6 draft rows a step and
  keeps ~4.5 (~69% DSpark acceptance), ~98 ms per step (~88 ms verify forward + ~10 ms commit/draft).
- Both fall as the cached prefix grows: at 128k–300k context the session read ~820 tok/s prefill and ~45 tok/s
  decode. The 1300–1600 range is the short-to-mid-context, warm-cache regime.
- Nearly all traffic today was cache-hit continuation; a suffix prefill still attends over the whole cached
  prefix. The two cold full prefills of the day (3.8k and 26.8k tokens, `cached=0`) read 1533 and 1618 tok/s.

## Repository layout

```
engine/     the TensorFold source the binary is built from (Apache-2.0; see engine/README.md)
gateway/    the Anthropic/OpenAI gateway (stdlib-only Python) + tests + Dockerfile
scripts/    serve.sh, install_engine.sh, make_native_view.py, make_view.py, drop_cache.py, dropcache_loop.sh
config/     inkling.env (all knobs), chat_template.engine-C.jinja, plan-gamma-scores.f32
packs/      checksums, manifests and provenance for the two q6 packs (binaries live on Hugging Face)
tests/      live/ (end-to-end checks against a running stack), exl3_ref/ (EXL3 format reference checks), needle/
docs/       deployment.md (operations), weights.md (provenance and the packs), chat-template.md (the template fixes)
```

## Verification

```sh
# gateway unit tests (CPU only)
cd gateway && python3 -m unittest discover -s tests

# end-to-end, against a running stack
python3 tests/live/verify_gateway.py        # 20 checks through the gateway
python3 tests/live/verify_concurrency.py    # two concurrent requests, both served
python3 tests/live/verify_reuse.py          # prompt-cache reuse across turns
```

## Known limits

- **One stream at a time.** The inkling2 backend has a single slot; concurrent requests queue in the engine's lane
  host. A queued stream gets no bytes until it runs.
- **500K context.** Validated in daily use.
- **Reasoning is always on.** `none` maps to `minimal`; the model never stops reasoning.
- **Text only.** Images are refused (the gateway replaces them with a text note).

## Licenses and provenance

Apache-2.0 throughout (see `LICENSE`, `NOTICE`, `engine/THIRD_PARTY_NOTICES.md`). The model weights are Apache-2.0
(the two base models and the packs repo). The EXL3 export is by kortexa-ai; the native release is by Thinking Machines. The dense
and draft packs are derived works of the release, published at
[`myanvoos/Inkling-2-Small-EXL3-Sparks`](https://huggingface.co/myanvoos/Inkling-2-Small-EXL3-Sparks); see `packs/` and `docs/weights.md`.
