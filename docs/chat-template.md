# The Inkling chat template (template C) and its fixes

`config/chat_template.engine-C.jinja` is the template the stack serves with. It is the release template with
three fixes, made so the engine's template compiler accepts it and so downstream harnesses parse tool calls
correctly. `config/chat_template.release.jinja` is the unmodified release template, kept for reference and for
diffing; do not serve it.

| File | md5 | What |
| --- | --- | --- |
| `config/chat_template.engine-C.jinja` | `db5391e5c018eec0eaa718aaae664227` | the one in use ("template C") |
| `config/chat_template.release.jinja` | `d6a1867e73ed26b44a2134128891db18` | the release's template, unmodified |
| HF's update (not in this repo) | `a053e53aa4f8721a775b45cea866a2bf` | the release + `\| float`; the engine refuses it at load |

The release template is from
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small) (Apache-2.0).

## Why template C exists

Set on 2026-11-15 to fix tool parsing in downstream harnesses, with the standing rule that the model never stops
reasoning. Reasoning is always on; `none` maps to `minimal` at the gateway. Template D (C plus a generation prompt
that forces a text block when reasoning is off) was built and rejected for the same reason.

## The three fixes

### 1. `enable_thinking` is honored (release: ignored)

```diff
-    {%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
+    {%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else (0.0 if enable_thinking is defined and enable_thinking == false else 0.9) -%}
```

The release template always defaults to effort 0.9. Template C defaults to 0.0 when the caller passes
`enable_thinking: false` (the `chat_template_kwargs` convention), otherwise 0.9. An explicit `reasoning_effort`
still wins over both.

### 2. `| float` dropped (the engine refuses it)

```diff
-        {%- set num = eff | float -%}
+        {%- set num = eff -%}
```

The engine's template compiler does not implement the `float` filter (`engine/zig/src/core/template/filters.zig`:
`float` is in the Jinja filter list but not in the implemented set), so any template using it fails at load. The
release template uses it; HF's update adds another one. Template C uses the value as-is: a string key is looked up
in `effort_map`, a number is used directly.

### 3. Tool-call arguments: top-level keys sorted, nested objects keep their order

```diff
-            {{- (args | tojson(sort_keys=true, separators=(",", ":"))) -}}
-            {{- "}<|end_message|>" -}}
+            {{- "{" -}}
+            {%- for key, value in args | dictsort(case_sensitive=true) -%}
+                {{- (key | tojson) ~ ":" ~ (value | tojson(separators=(",", ":"))) -}}
+                {%- if not loop.last -%},{%- endif -%}
+            {%- endfor -%}
+            {{- "}}<|end_message|>" -}}
```

The release template serializes the whole `args` object with `tojson(sort_keys=true)`, which sorts keys at **every**
level recursively. Template C sorts only the top level (`dictsort(case_sensitive=true)`) and serializes each value
with `tojson(separators=(",", ":"))` without `sort_keys`, so nested objects keep the order the client sent them in.
Top-level key order is unchanged between the two (both sort by raw byte order); only nested objects differ.

The engine's `tojson` and `dictsort` both exist and both sort by raw byte order
(`engine/zig/src/core/template/pyjson.zig`, `filters.zig`), so this is a deliberate choice about *which* keys get
sorted, not a workaround for a missing filter.

## What did not change

Everything else is byte-identical to the release template: the effort map (`none` 0.0 … `max` 0.99), the
`<|message_*|>` role tokens, the tool-declare block, the reasoning-content rendering, the tool-call header
(`<|content_invoke_tool_json|>`), the `<|content_model_end_sampling|>` terminator, and the
image/audio handling.

## Verifying

```sh
md5sum config/chat_template.engine-C.jinja   # db5391e5c018eec0eaa718aaae664227
diff config/chat_template.release.jinja config/chat_template.engine-C.jinja
```

The gateway's test suite renders this template against recorded replies
(`gateway/tests/test_inkling_prefix.py`); with jinja2 installed, `gateway/tests/test_inkling.py` renders converted
Anthropic history through it.
