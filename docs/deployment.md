# Operating the stack

## Start / stop / status

```sh
scripts/serve.sh start     # preflight, then the serve as the systemd user unit inkling-serve; follows the log
scripts/serve.sh stop      # gateway (if ours), both ranks, the lock, the units
scripts/serve.sh restart
scripts/serve.sh status    # engine, rank 1, gateway, lock, unit, droppers
scripts/serve.sh logs      # tail serve.log and r0.log
scripts/serve.sh check     # preflight only, read-only: what a start would refuse, and why
scripts/serve.sh gateway   # recreate only the gateway (after a gateway/ change)
scripts/serve.sh dropcache start|stop|status
```

`start` refuses at once when: the port is taken by another engine; the gateway container serves something else;
a GPU runs a process outside the pair lock; a node lacks the engine, the template, the view, the dense pack or NCCL;
or a user unit would get fewer open files than the load needs (`NOFILE`). A GPU busy *under* the pair lock is not a
refusal — the serve queues behind it (`LOCK_WAIT_S`).

The serve runs as the systemd user unit `inkling-serve` (rank 0 + the lock holder), so it outlives the shell that
started it. Rank 1 runs as `inkling-serve-r1` on the worker. Ctrl-C during `start` stops following, not the serve.

## The GPU pair lock

`$GPU_LOCK_DIR` (default `/var/tmp/inkling-gpu-lock`) holds `gpu.turnstile` and `gpu.lock`. A serve takes the
turnstile, then the lock, and holds it for its whole lifetime; other GPU jobs that take the same lock queue behind
it. On a shared machine, point `GPU_LOCK_DIR` at the shared lock directory. The serve's markers are
`.inkling-serve-holder` / `.inkling-serve-lock-held` / `.inkling-serve-release`; `stop` releases the lock.

## Page caches

The start drops the page cache on both nodes before the load and once more after READY, using
`/usr/local/sbin/dropcache-helper` via `sudo -n` (root-owned, argument-less: `sync; echo 3 > /proc/sys/vm/drop_caches`).
Where `sudo -n` refuses, `scripts/drop_cache.py` runs instead (fadvise walk, ~25 s). While serving, a dropper unit on
each node (`inkling-serve-dropcache`, `inkling-serve-dropcache-r1`) drops every `DROP_EVERY_S` (30 s) when the
reclaimable file cache exceeds `DROP_MIN_MIB` (512 MiB). Logs: `logs/dropcache.log`, `logs/dropcache-r1.log`.

## Files and units

| What | Where |
| --- | --- |
| Rank 0 | head `127.0.0.1:8888`; unit `inkling-serve`; `logs/r0-<stamp>.log` (`logs/r0.log` symlink) |
| Rank 1 | worker `:8889`; unit `inkling-serve-r1`; `logs/r1-<stamp>.log` on the worker |
| NCCL bootstrap | `$HEAD_IP:29597` (`NCCL_COMM_ID`) |
| Gateway | `inkling-gateway` container on `0.0.0.0:8080` → `127.0.0.1:8888` |
| Gateway memory | `state/gateway/inkling-memory.jsonl`, mounted at `/var/lib/inkling-gateway` |
| Serve log, state | `logs/serve.log`, `logs/state` |

## The gateway

The gateway is a stdlib-only Python program (`gateway/inkling_gateway.py`). The serve runs it as the
`inkling-gateway` container (built from `gateway/Dockerfile` on first use). To run it without Docker:

```sh
python3 gateway/inkling_gateway.py --listen 0.0.0.0:8080 --upstream http://127.0.0.1:8888 \
  --served-model inkling-2-small --display-name Inkling-2-Small --hidden-alias inkling \
  --context 500000 --max-tokens 65536 --root thinkingmachines/Inkling-2-Small-EXL3-Native --family inkling2
```

It speaks OpenAI chat completions, the Responses API, Anthropic Messages (+ count_tokens), and `/v1/models`.
With `--family inkling2` it parses Inkling's tool-call markup into `tool_calls` / `tool_use`, maps every request to
one of Inkling's reasoning levels (never `none`), and keeps a per-turn reasoning memory so clients that drop
reasoning still get prompt-cache reuse. See `gateway/README.md`.

The gateway's memory file survives restarts (`docker stop` sends SIGTERM; the gateway rewrites the file compact).
`scripts/serve.sh gateway` recreates the container and carries the memory file over.

## Troubleshooting

- **`EMFILE` / "cannot open ... .safetensors" on load**: the load opens every shard at once (~2,358 files). A systemd
  user unit's default soft limit is 1024. `NOFILE` in `config/inkling.env` (default 500000) sets `LimitNOFILE` on
  both units; `scripts/serve.sh check` measures what a unit actually gets.
- **Rank 1 exits during load**: check `logs/r1-<stamp>.log` on the worker (`ssh $WORKER_SSH tail -50 ...`). Usually
  memory (`MIN_AVAIL_GIB`) or a missing file (`check` names it).
- **Gateway answers but the engine does not**: `scripts/serve.sh status`; the gateway needs the engine on :8888.
- **A second concurrent request fails**: expected — one stream at a time; the engine queues it. Clients should retry.
- **First request after a restart re-prefills**: the gateway's reasoning memory was lost (it lives in
  `state/gateway/inkling-memory.jsonl`; keep the directory).

## Updating the engine

```sh
scripts/install_engine.sh            # rebuild from engine/ and install on both nodes
# or: scripts/install_engine.sh /path/to/tensorfold-native
```

A new build gets a new md5; set `ENGINE_MD5` in `config/inkling.env` to pin it. The serve never replaces a running
binary in place.


