# Tests

- `live/` — end-to-end checks against a running stack (`scripts/serve.sh start` first). Standard library only.
  - `verify_gateway.py` — 20 checks through the gateway: model card, chat (streamed and not), reasoning levels,
    tool-call round trips on OpenAI and Anthropic APIs, count_tokens.
  - `verify_concurrency.py` — two simultaneous requests; both must succeed (the engine queues the second).
  - `verify_reuse.py` — prompt-cache reuse across turns when the client drops its reasoning.
- `exl3_ref/` — CPU reference checks for the EXL3 format and the engine's TP layout (no GPU needed):
  `check_index.py VIEW` validates a view's checkpoint and shard headers; `check_kernels.py`, `check_semantics.py`,
  `check_tp.py`, `selftest_ref.py` check the trellis decode against a reference. `tf_exl3_format.py` is the format
  reference extracted from TensorFold.
- `needle/` — the 8K needle-in-a-haystack prompt ids used for the export's validation panel
  (`make_kortexa_ids.py` regenerates them; needs `jinja2` and `tokenizers`).
