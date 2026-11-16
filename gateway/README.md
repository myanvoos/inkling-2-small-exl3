# inkling gateway

The Anthropic/OpenAI gateway for the Inkling-2-Small stack. It runs from `inkling_gateway.py`, mounted
at `/gateway` in the `inkling-gateway` container (or run directly with any Python 3.10+; it uses only the
standard library).

## Responses and reasoning compatibility

The OpenAI routes remain passthrough routes, with a small compatibility layer at
this boundary for TensorFold's chat-backed Responses implementation:

- `include: ["reasoning.encrypted_content"]` is accepted and removed before
  forwarding. TensorFold returns the reasoning item's plaintext content, which
  is sufficient for replay; other include values are rejected by name rather
  than receiving the backend's misleading encryption error.
- The effective `reasoning.context` is echoed in Responses responses and
  `response.created`/terminal SSE events. The gateway preserves the existing
  all-turns behavior by default. An explicit `current_turn` filters replayed
  reasoning items from a stateless `input` list; `all_turns` retains them.
- `thinking: {"type": "disabled"}` maps to the engine's `none` effort on both
  Responses and chat completions. `enabled` maps its optional positive
  `budget_tokens` to `thinking_budget`.
- A chat request with tools and an assistant history item lacking non-empty
  `reasoning_content` gets `X-Inkling-Reasoning-Warning` and a `warnings`
  response field. Set `GATEWAY_STRICT_REASONING=1`, or send
  `X-Inkling-Strict-Reasoning: 1`, to turn that warning into a 400. Lenient
  mode is the default for OpenAI-client compatibility.

The compatibility controls are intentionally gateway-local: no model or GPU
container restart is needed when changing this file; restart only
`inkling-gateway` after the source has been updated.

## Inkling 2 (`--family inkling2`)

The serve script starts the gateway with `--family inkling2` (or `GATEWAY_FAMILY=inkling2`) in front of the native
TensorFold inkling2 server. With `--family` unset, none of this section applies: requests and replies pass through
unchanged (`tests/test_inkling.py`, `OtherRecipesHttp`).

### Tool calls

Template C renders a call as `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>`,
one message a call. The native server drops the message markup from replies, so it returns calls unparsed in `content`,
one after another with no separator and after any text, with finish_reason `stop` and `tool_calls` null:

```text
Checking now.get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Wellington"}}get_time<|content_invoke_tool_json|>{"name":"get_time","args":{"city":"Auckland"}}
```

The gateway parses them on all three APIs:

- **OpenAI `/v1/chat/completions`:** `message.tool_calls` (`call_` ids, arguments as a JSON string), `content` the
  remaining text (null when there is none), finish_reason `tool_calls`. Streamed: content deltas without the calls, each
  call as one whole `tool_calls` delta (index 0, 1, ...) when its JSON closes, and the terminal chunk's `stop` becomes
  `tool_calls`. Chunks with no call in them keep their bytes.
- **Anthropic `/v1/messages`:** `tool_use` blocks (`toolu_` ids) and stop_reason `tool_use`; streamed, `tool_use`
  blocks with `input_json_delta`.
- **What counts as a call:** only when the request offers tools (and tool_choice is not `none`). The header must be a
  declared tool's name (case-insensitive; the call takes the declared spelling) right before the marker, and the JSON
  one object whose `name`, when present, names that tool and whose `args` (or `arguments`) is an object. An undeclared
  name and malformed or unclosed JSON stay text exactly as the model wrote them. While streaming, text that could
  still become a header (a declared name's prefix at its end) is held back until it cannot: at most one name.
- **`parallel_tool_calls: false`** (Anthropic `disable_parallel_tool_use`; Codex sends it for a model it does not
  know): the template has no place to tell the model about the limit, so it makes several calls now and then. With
  tools offered, the gateway adds the stop string `}}<|end_message|>` (the end of a call message: its JSON's closing
  braces, then the message end) to the engine request, so the reply ends where the first call message ends
  (`GATEWAY_INKLING_PARALLEL=stop`, the default).
  - The engine checks its stop strings against the decoded tail of the reply after every token, special tokens
    spelled out. It ends the reply at the token that completes the match, and its state keeps the rows up to that
    token and nothing after it. That state holds exactly the one call the client gets, so the next turn renders back
    to it.
  - The engine's text loses the stop string. When the reply's `stop_sequence` is the gateway's own, the gateway puts
    the `}}` back and takes the field out of the client's copy.
  - A thinking or text message that itself ends in `}}` ends the reply there. The log line names it.
  - If the first call's JSON ends any other way, the stop does not fire, and the later calls are dropped.
  - `GATEWAY_INKLING_PARALLEL=all` (once `keep`) hands over every call the model made. `drop` keeps the first call
    without the stop, as the engine does for its own formats. The dropped calls stay in the engine's state, so that
    reply re-runs whole on the next request.
- **History:** Anthropic `tool_use` becomes assistant `tool_calls`, `tool_result` becomes `tool` messages with
  `tool_call_id` (the template finds the call's name by it), and `thinking` becomes `reasoning_content` (the template
  renders it as the model's thinking message). The engine turns argument strings into the objects the template
  requires (`messages.zig`).
- **Responses API (`/v1/responses`):** see below: `function_call` items and their events.

### The Responses API, on the chat route

The engine's own `/v1/responses` runs each request as its chat completion inside the engine, where Inkling's calls stay
text, and its store keeps them as text. So for Inkling the gateway does that translation itself, as a port of the
engine's `responses_translate.zig` and `responses.zig` (the same items, events and fields), with `InklingStream`
between the engine's chat reply and the items:

- **Request:** `input` items become chat messages:
  - `message` → its role's message;
  - `function_call` → an assistant `tool_calls` entry, id = `call_id`;
  - `function_call_output` → a `tool` message with that `tool_call_id`;
  - `reasoning` (its `content`, else its `summary` text) → the next assistant turn's `reasoning_content`;
  - `item_reference` → the stored item it names.

  `instructions` becomes a system message. Function tools, `tool_choice` (including `allowed_tools`), `text.format`,
  `max_output_tokens` and the sampling fields map as the engine maps them, and its refusals are kept. Images become
  the text note.
- **Reply:**
  - Items in order: `reasoning` (`reasoning_text` content), `message` (`output_text`), then one `function_call` per
    call (`call_id` = the chat call's id, `fc_` item ids).
  - Streamed, the engine's events with `sequence_number`: `response.created`, `.in_progress`, then for each item
    `output_item.added` and `content_part.added`, `output_text` / `reasoning_text` deltas, or
    `function_call_arguments.delta` / `.done`, then `output_item.done`, and finally `response.completed` (or
    `.incomplete` / `.failed`) carrying every item and the usage.
  - No raw markup is left in `output_text`, and a malformed call stays text there.
- **Store:** `store` (default true) keeps the finished response in the gateway's memory, as the engine kept its own:
  1,024 responses or 256 MiB. It serves `previous_response_id` (the chain's messages), `item_reference`, and
  `GET` / `DELETE /v1/responses/{id}`. A restart forgets it; a missing id is a 400 naming what to send instead.
- Without `--family inkling2`, `/v1/responses` is relayed to TensorFold unchanged.

### Routes without `/v1`

Some clients use a base URL without `/v1`, such as `http://host:8080`. In this mode the gateway treats these as their
`/v1` routes, so the parsing and the levels apply there too: `/chat/completions`, `/completions`, `/responses` (and
`/responses/{id}`), `/messages`, `/messages/count_tokens` and `/models`.

### Reasoning memory

The engine reuses its prompt state only up to the first token where a request's prompt differs from what it holds:
the previous prompt and the reply it generated. It resumes at that state's end or at a snapshot at the previous
prompt's end. A reply that comes back changed is re-run whole, and a change inside an earlier turn re-runs the prompt
from the start. Template C renders the assistant turn a client sends back, so that turn must render to the model's
own tokens. Clients lose parts of it:

- OpenAI-compatible chat clients and Codex never return `reasoning_content`, and some strip Anthropic thinking
  blocks. The model then also reasons from scratch.
- Text that is only white space never reaches a chat client (its `content` is null) or a client that skips it.
- A client may write a call's arguments again in its own way: keys reordered, `1.0` as `1` (JavaScript's JSON).
- A call's name can differ from the declared spelling (`bash` for a declared `Bash`). The client gets the declared
  name.

So the gateway keeps each reply as the model wrote it: its reasoning and text exactly, and each call's name as
written and arguments in the model's key order. It keeps the reply under what its client sends back: each tool call's
id (`call_`, `toolu_`, a `call_id`), and its text together with the message before it, so two replies that both say
"Done." stay apart. A later request's copy of that turn is put back to the reply where the copy is the same turn:

- missing reasoning, or reasoning that differs only in surrounding white space;
- text that differs only in surrounding white space;
- arguments that parse to the same values;
- a name that differs only in case.

A turn the client really changed keeps the client's version, and OpenRouter's `reasoning` field counts as
`reasoning_content`. Live on 2026-11-15, with a client that drops its reasoning, turn N+1's cached prompt was turn
N's prompt plus its reply, to within one token (the terminal token), on all three APIs.

The memory is bounded (least recently used goes first) and written through to a file, one JSON line a reply.
SIGTERM (`docker stop`, `docker restart`) rewrites the file compact, and a start reads it back. Before this, a gateway
restart dropped every remembered turn at once: the first request after one re-ran its whole prompt (273 s at 299k
tokens on 2026-11-15). The Inkling runner mounts `state/gateway/` at the file's directory and copies a container's
file out before re-creating it.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `GATEWAY_INKLING_MEMORY_MAX` | `8192` | Entries kept. |
| `GATEWAY_INKLING_MEMORY_MB` | `64` | Memory bound for the replies' texts. |
| `GATEWAY_INKLING_MEMORY_FILE` | `/var/lib/inkling-gateway/inkling-memory.jsonl` | The file; empty keeps the memory in RAM only. |
| `GATEWAY_INKLING_PARALLEL` | `stop` | `parallel_tool_calls: false`: `stop` after the first call, `all` calls, or `drop` all but the first. |
| `GATEWAY_INKLING_REPLAY_CHECK` | `1` | `0` turns off the replay check below. |

Some differences cannot be put back, because the template writes a turn in one way: thinking, then one text message,
then the calls, with each call's top-level argument keys sorted and its JSON compact. A reply that differs from that
is re-run once: a second thinking message, text after a call, unsorted keys, or spaced JSON. So are tokens that do
not re-tokenize to themselves. The request log names what the gateway can see:

- `restored: text N, arguments N, names N, trimmed reasoning N` on top of `reasoning restored to N turns`.
- `turn: ...` on the new reply's line when its re-render cannot repeat it, with the reason. For a call's JSON, it
  shows the first differing character and the text around it, as the model wrote it and as the template writes it.
- `prefix: the last reply was not reused (N of its tokens re-ran): <its turn notes>` when this request's cached
  tokens stop at that reply's own prompt. `prefix: cached N is short of the last reply's prompt` means another
  request ran in between (the engine holds one state), or an earlier turn changed.
- **Replay check:** the gateway asks for each reply's token ids (`return_token_ids`). Unless the client asked for
  them too, it takes them out of the client's copy. In the background, it has the engine's `/tokenize` render the
  turn as it will come back, and compares the two. A line `inkling replay: the reply (call ...) re-renders
  differently from token T of N (K before its end): after '...' the reply has '...', its re-render '...'` names the
  first token that differs: a tokenization, or what the turn notes say. A reply that renders back exactly logs
  nothing.

### Reasoning, always on

Template C's `effort_map` is none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99 and max 0.99; unset means
0.9. The engine takes only those names (no numbers), so the gateway gives every request one level, and `none` never
reaches the engine:

| The client sends | Level |
| --- | --- |
| Nothing (any route) | `high` (0.9, the template's own default) |
| A level as OpenAI `reasoning_effort`, Responses `reasoning.effort` or Anthropic `output_config.effort` | That level |
| `none` there | `minimal` |
| A number from 0 to 1 there | The nearest level from `minimal` up (0 is `minimal`; ties go higher) |
| `thinking: {"type": "disabled"}`, `chat_template_kwargs.enable_thinking: false` or `chat_template_kwargs.thinking: false` | `minimal`, over any effort |
| `thinking: {"type": "enabled", "budget_tokens": N}` and no effort | N < 4096 `low`, < 16384 `medium`, < 32768 `high`, else `max` |
| `thinking: {"type": "adaptive"}` | As unset, or its effort |

`budget_tokens` picks a level. It is not forwarded as `thinking_budget`: the engine would cut the thinking message at
that count, which nothing has validated for Inkling. count_tokens renders the same level as generation, and Anthropic's
`budget_tokens` must still be at least 1024.

### Other differences in this mode

- `/v1/models` cards say `input_modalities: ["text"]` and `supports_vision: false`.
- Anthropic image blocks, direct or inside tool results, become the text `[image omitted: this model reads text only]`.
  The engine refuses images, and one refused image would fail every later turn of that conversation. OpenAI image
  parts reach the engine unchanged, and it refuses them.
- No reply pre-fill notifications (`/tokenize` `tf_incoming`): the native server has no reply pre-fill.
- No `<tool_call>` text hoisting, and no reasoning-history warning.
- The engine serves one stream at a time: a second concurrent request fails with "Inkling-2 serves one stream at a
  time: retry once the current request ends" (clients retry). The gateway does not queue them.

## Model discovery

Both `GET /models` and `GET /v1/models` return the gateway's `--served-model`
as `id` and `display_name`. Each
`--alias-model` adds another card with the same limits and `root`; requests
naming an alias ask TensorFold for the served model and reply under the alias.
The `root` field still names the underlying checkpoint. Both routes require a
healthy upstream.

Two more options, which only this stack passes (every other card is as before):

- `--display-name NAME`: the served model's `display_name` (default: its id). Inkling: `Inkling-2-Small`.
- `--hidden-alias ID` (repeatable): an id the gateway answers like an alias, but without a card, for an old name that
  clients still send. This stack: `inkling`, the name clients used before the served id became
  `inkling-2-small`. Requests naming it reach TensorFold as the served model (Messages, count_tokens, Responses), and
  replies name the id the client asked for. On the OpenAI passthrough routes the id is relayed as sent, and the engine
  answers under its own name.

## Image history

Some clients keep only their latest five tool images. Removing an older image
changes the prompt prefix and forces TensorFold to prefill the remaining history.
The gateway remembers image-bearing user messages and restores omitted images:

- An explicit `image omitted to fit request image limits` marker restores the
  original message.
- Silent removal restores only missing image parts within otherwise identical
  tool results. The preceding history, tool call IDs, text, surviving images,
  part order, and other message fields must match.
- Silent deletion of a direct user image is not undone. Edited tool results and
  replaced images pass through unchanged.

Restoration happens before time stabilization and token counting. Request logs
count both direct images and images inside tool results, including restored ones.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `GATEWAY_RESTORE_IMAGES` | `1` | Set to `0` to disable restoration. |
| `GATEWAY_RESTORE_IMAGES_MB` | `64` | Memory bound for original messages; least recently used go first. |
| `GATEWAY_RESTORE_IMAGES_MAX` | `50` | Total images allowed after restoration. Match this to the backend's image limit. |

The image-count cap only limits restoration; it does not strip images supplied
by the client. At the cap, the gateway leaves omitted images absent rather than
pushing the request over the backend's limit. Cache reuse may then fall.

Text-only message content is sent to TensorFold as one string, with parts joined by `""`. Joining
list-content messages with `"\n\n"` instead would change the rendered history when a conversation's first image
arrived and force a full prefill.

History is process-local. A restart loses stored originals and time notes, so an
existing conversation may need one fresh prefill. Images already absent before
the new gateway sees the conversation cannot be recovered.

## Tests

From this directory, run the CPU-only regression tests:

```sh
uv run --no-project python -m unittest discover -s tests -v
```

`tests/test_inkling_responses.py` covers the Responses route, the reasoning memory on every API and Claude-Code-style
tool sets. `tests/test_inkling_prefix.py` takes replies as the model writes them (message markup and all), gives
them to the gateway the way the engine returns them, and sends the turn back as clients do. It then renders what
reaches the engine with the deployed template C, and the render must be the reply, character for character,
wherever the gateway can make it so. It also covers the notes, the replay check, `parallel_tool_calls: false`, the
token ids and the memory file across a restart. For the stop after a first call, it renders the next request with the
template and checks that the engine's state after the stop starts it, for either place a round can put the stop
token. With jinja2, `tests/test_inkling.py` also renders converted Anthropic history through the Inkling
deployment's template C (it is skipped without jinja2):

```sh
uv run --no-project --with jinja2 python -m unittest discover -s tests
```

After tests pass, wait for requests to finish before restarting only the gateway:

```sh
docker restart inkling-gateway
```

The TensorFold containers and their GPU prefix caches do not need a restart.
