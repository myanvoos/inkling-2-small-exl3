#!/usr/bin/env python3
"""The Inkling gateway: Anthropic's Messages API and OpenAI's API in front of the TensorFold inkling2 server.

TensorFold serves OpenAI's API only. This gateway adds Anthropic's Messages API, with local changes (Anthropic's
client tools mapped onto function tools, server tools left out) and Anthropic's fields on /v1/models. Here:

- GET /models and /v1/models: the gateway's model cards, one per id (--served-model and each --alias-model),
  with its context/output limits.
- POST /v1/messages and /v1/messages/count_tokens: Anthropic's Messages API on TensorFold's /v1/chat/completions
  and /tokenize, converted the way profile C's vLLM Anthropic layer converts (a port of its serving.py).
- anything else goes to TensorFold unchanged (OpenAI chat completions, /tokenize, /health, /metrics, ...).
- --family inkling2 (the inkling-2-small-exl3 recipe only): Inkling's tool calls parsed out of content into OpenAI
  tool_calls and Anthropic tool_use, every request given one of its reasoning levels (never none), a text-only card.

Standard library only, so it runs in any image with Python 3.10+.
"""
import argparse, hashlib, json, os, re, sys, threading, time, urllib.error, urllib.parse, urllib.request, uuid
from collections import OrderedDict
from datetime import datetime, timezone
import resume_spool
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
STARTED = int(time.time())
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "accept-encoding", "te",
       "upgrade", "proxy-connection", "trailer"}
BILLING = "x-anthropic-billing-header"   # Claude Code's per-request attribution header: defeats prefix caching

# Anthropic-defined client tools: Anthropic publishes the input schema and the client runs the call, so they map
# onto function tools under the name the request gives them (schemas from Anthropic's tool documentation).
TEXT_EDITOR_PROPERTIES = {
    "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"]},
    "path": {"type": "string", "description": "Absolute path to the file or directory."},
    "view_range": {"type": "array", "items": {"type": "integer"},
                   "description": "view: [start_line, end_line], 1-indexed; -1 reads to the end."},
    "file_text": {"type": "string", "description": "create: the content of the new file."},
    "old_str": {"type": "string", "description": "str_replace: exact text to replace."},
    "new_str": {"type": "string", "description": "str_replace: replacement text."},
    "insert_line": {"type": "integer", "description": "insert: line after which to insert (0 = start)."},
    "insert_text": {"type": "string", "description": "insert: the text to insert."},
}
CLIENT_TOOLS = {
    "bash_": (
        "Run a command in a persistent bash session. The working directory, environment variables and files persist "
        "between calls. Set restart to true to start a fresh session.",
        {"type": "object", "properties": {
            "command": {"type": "string", "description": "The bash command to run."},
            "restart": {"type": "boolean", "description": "Set to true to restart the bash session."},
        }},
    ),
    "text_editor_": (
        "View, create and edit text files. Commands: view (file or directory, optional view_range), create (path, "
        "file_text), str_replace (path, old_str, new_str; old_str must match exactly once), insert (path, "
        "insert_line, insert_text).",
        {"type": "object", "properties": TEXT_EDITOR_PROPERTIES, "required": ["command", "path"]},
    ),
    "memory_": (
        "Store and retrieve information across conversations in the /memories directory. Commands: view, create, "
        "str_replace, insert, delete (path), rename (old_path, new_path).",
        {"type": "object", "properties": {
            **TEXT_EDITOR_PROPERTIES,
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "delete", "rename"]},
            "old_path": {"type": "string", "description": "rename: current path."},
            "new_path": {"type": "string", "description": "rename: new path."},
        }, "required": ["command"]},
    ),
}
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal", "refusal": "refusal"}
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error", 529: "overloaded_error"}


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def client_tool(tool_type):
    for prefix, (description, schema) in CLIENT_TOOLS.items():
        if tool_type.startswith(prefix):
            if prefix == "text_editor_" and tool_type < "text_editor_20250429":   # these also had undo_edit
                schema = json.loads(json.dumps(schema))
                schema["properties"]["command"]["enum"].append("undo_edit")
            return description, schema
    return None


def image_url(source):
    if source.get("type") == "url":
        return source.get("url", "")
    return f"data:{source.get('media_type', 'image/jpeg')};base64,{source.get('data', '')}"


def system_text(content):
    texts = [content] if isinstance(content, str) else [b["text"] for b in content or []
                                                       if b.get("type") == "text" and b.get("text")]
    parts = []
    for text in texts:
        if text.startswith(BILLING):
            text = text.partition("\n")[2]   # remove only attribution, not instructions in the same block
        if text:
            parts.append(text)
    return "".join(parts) if parts else None


def convert_messages(body):
    out, system = [], []
    if top := system_text(body.get("system")):
        system.append(top)
    if system:
        out.append({"role": "system", "content": "".join(system)})
    for msg in body["messages"]:
        role = msg.get("role")
        if role == "system":
            # Inline system messages stay where the client put them (the template renders them in place). Claude
            # Code adds one after every tool result; hoisting them into the top system prompt changed the prompt's
            # start every turn and forced the whole conversation to be prefilled again.
            if text := system_text(msg.get("content")):
                out.append({"role": "system", "content": text})
            continue
        om = {"role": role}
        content = msg.get("content")
        if isinstance(content, str):
            om["content"] = content
        else:
            convert_content(content or [], role, om, out)
        if role == "assistant" and not ({"content", "tool_calls", "reasoning_content"} & om.keys()):
            om["content"] = ""
        if not (role == "user" and "content" not in om):
            out.append(om)
    return out


def convert_content(blocks, role, om, out):
    parts, calls, reasoning = [], [], []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            if block.get("text"):
                parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            if block.get("source") and inkling():
                parts.append({"type": "text", "text": INKLING_IMAGE_NOTE})   # the inkling2 server reads text only
            elif block.get("source"):
                parts.append({"type": "image_url", "image_url": {"url": image_url(block["source"])}})
        elif kind == "thinking":
            if block.get("thinking") is not None:
                reasoning.append(block["thinking"])
        elif kind == "redacted_thinking":
            pass                          # opaque; not model-readable content
        elif kind == "tool_reference":
            raise RequestError("messages: tool_reference must be inside a user tool_result")
        elif kind == "tool_use":
            calls.append({"id": block.get("id") or f"call_{int(time.time())}", "type": "function",
                          "function": {"name": block.get("name") or "", "arguments": json.dumps(block.get("input") or {})}})
        elif kind == "tool_result":
            if role == "user":
                user_tool_result(block, out)
            else:
                parts.append({"type": "text", "text": f"Tool result: {block.get('content') or ''}"})
        else:
            raise RequestError(f"messages: content block type {kind!r} is not supported")
    if reasoning:
        om["reasoning_content"] = "".join(reasoning)
    if calls:
        om["tool_calls"] = calls
    if parts:
        # Text-only content goes as one string, joined the way the engine's text path joins parts (""). Joining
        # list-content messages with "\n\n" instead would render the whole history differently once any image
        # arrived, and the engine would prefill it again.
        if all(p["type"] == "text" for p in parts):
            om["content"] = "".join(p["text"] for p in parts)
        else:
            om["content"] = parts


def user_tool_result(block, out):
    text, images, parts = "", [], []
    content = block.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif item.get("type") == "tool_reference":
                name = item.get("tool_name")
                if not isinstance(name, str) or not name.strip():
                    raise RequestError("tool_result.tool_reference.tool_name must be a non-empty string")
                if texts:
                    parts.append({"type": "text", "text": "\n".join(texts)})
                    texts = []
                parts.append({"type": "tool_reference", "name": name})
            elif item.get("type") == "image" and inkling():
                texts.append(INKLING_IMAGE_NOTE)   # the inkling2 server reads text only: no image message follows
            elif item.get("type") == "image" and (url := image_url(item.get("source", {}))):
                images.append(url)
        text = "\n".join(texts)
        if parts and texts:
            parts.append({"type": "text", "text": text})
    out.append({"role": "tool", "tool_call_id": block.get("tool_use_id") or "", "content": parts or text})
    if images:
        out.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in images]})


# GATEWAY_STABLE_TIME=1 (config/local.sh): a system prompt that carries the local time to the second (a chat client's
# "2026-10-15 19:03:06 +13:00 [Pacific/Auckland]") changes every turn, so TensorFold's kept prompt states stop matching
# there and every turn prefilled the whole conversation after it again (40-50k tokens a turn, 2026-10-15). With the
# flag the system prompt keeps the date and zone only, and the time goes to the end of the user message it arrived
# with, "(local time: ...)", where it stays: each user message's note is remembered, keyed by the conversation up to
# that message, so later turns render the earlier messages exactly as before and resume past them (the reply
# pre-fill's state included). A tool round inside a turn keeps the turn's note. Only a system prompt with exactly one
# such clock is touched. The memory is per process: after a gateway restart a conversation is prefilled in full once.
STABLE_TIME = os.environ.get("GATEWAY_STABLE_TIME") == "1"
CLOCK = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}(?::\d{2})?) ([+-]\d{2}:?\d{2}) \[([A-Za-z_]+(?:/[A-Za-z0-9_+\-]+)+)\]")
TIME_NOTES = OrderedDict()              # sha256 of the conversation through a user message -> that message's note
TIME_NOTES_MAX = 8192
TIME_NOTES_LOCK = threading.Lock()


def with_note(message, note):
    """A user message with ``note`` at the end of its content."""
    m = dict(message)
    content = m.get("content")
    if isinstance(content, list):
        m["content"] = content + [{"type": "text", "text": "\n\n" + note}]
    else:
        m["content"] = (content + "\n\n" if content else "") + note
    return m


def stabilize_time(messages, *, remember=True):
    """Converted messages with the system prompt's clock moved to the user messages (GATEWAY_STABLE_TIME); unchanged
    when the first message is not a system prompt with exactly one clock. ``remember``: store the newest user
    message's note (count_tokens only looks)."""
    if not messages or messages[0].get("role") != "system" or not isinstance(messages[0].get("content"), str):
        return messages
    clocks = CLOCK.findall(messages[0]["content"])
    if len(clocks) != 1:
        return messages
    date, clock, offset, zone = clocks[0]
    system = {**messages[0], "content": CLOCK.sub(lambda _: f"{date} [{zone}]", messages[0]["content"])}
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
    out, h = [system], hashlib.sha256(json.dumps(system, sort_keys=True).encode())
    for i, m in enumerate(messages[1:], 1):
        h.update(json.dumps(m, sort_keys=True).encode())        # the client's message, as it sends it every turn
        if m.get("role") == "user":
            key = h.hexdigest()
            with TIME_NOTES_LOCK:
                note = TIME_NOTES.get(key)
                if note is not None:
                    TIME_NOTES.move_to_end(key)
                elif i == last_user:
                    note = f"(local time: {date} {clock} {offset} [{zone}])"
                    if remember:
                        TIME_NOTES[key] = note
                        while len(TIME_NOTES) > TIME_NOTES_MAX:
                            TIME_NOTES.popitem(last=False)
            if note is not None:
                m = with_note(m, note)
        out.append(m)
    return out


# Pruning old images changes the prompt prefix. Keep originals by the history before each user message, with
# the system clock reduced to its date. Restore marked omissions, or silently removed tool-result images when
# nothing else changed. Hash the restored history so later messages still find their originals.
# This memory is process-local and bounded: images lost before startup or after eviction cannot be restored.
RESTORE_IMAGES = os.environ.get("GATEWAY_RESTORE_IMAGES", "1") != "0"
RESTORE_MARK = "image omitted to fit request image limits"
RESTORE_MAX = int(float(os.environ.get("GATEWAY_RESTORE_IMAGES_MB") or 64) * 2**20)
RESTORE_MAX_IMAGES = int(os.environ.get("GATEWAY_RESTORE_IMAGES_MAX") or 50)
ORIGINALS = OrderedDict()               # conversation-prefix key -> (original user message, its JSON size)
ORIGINALS_BYTES = 0
ORIGINALS_LOCK = threading.Lock()


def _message_texts(message):
    content = message.get("content")
    if isinstance(content, str):
        yield content
        return
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            yield block.get("text") or ""
        elif block.get("type") == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                yield inner
            elif isinstance(inner, list):
                yield from ((i.get("text") or "") for i in inner if isinstance(i, dict) and i.get("type") == "text")


def _image_count(message):
    content = message.get("content") if isinstance(message, dict) else None
    count = 0
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        count += block.get("type") == "image"
        inner = block.get("content") if block.get("type") == "tool_result" else None
        if isinstance(inner, list):
            count += sum(isinstance(i, dict) and i.get("type") == "image" for i in inner)
    return count


def _tool_images_removed(original, message):
    """True only for deleted image parts in otherwise identical tool results, including their call IDs.

    Surviving parts must stay in order. Replaced images, edited text, and direct user-image deletions do not match.
    """
    before, after = original.get("content"), message.get("content")
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after):
        return False
    parts = list(after)
    for i, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        if not (isinstance(old, dict) and isinstance(new, dict)
                and old.get("type") == new.get("type") == "tool_result"):
            return False
        left, right = old.get("content"), new.get("content")
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        j = 0
        for part in left:
            if j < len(right) and part == right[j]:
                j += 1
            elif not (isinstance(part, dict) and part.get("type") == "image"):
                return False
        if j != len(right):
            return False
        parts[i] = {**new, "content": left}
    return parts != after and {**message, "content": parts} == original


def restore_images(body, *, remember=True):
    """(body with omitted images put back, how many messages were restored). ``remember``: store messages that carry
    images (count_tokens only looks)."""
    global ORIGINALS_BYTES
    messages = body.get("messages")
    if not RESTORE_IMAGES or not isinstance(messages, list):
        return body, 0
    system = CLOCK.sub(lambda c: f"{c.group(1)} [{c.group(4)}]", system_text(body.get("system")) or "")
    h = hashlib.sha256(json.dumps(system).encode())
    out, restored = [], 0
    images = sum(_image_count(m) for m in messages)
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            key = h.hexdigest()
            marked = any(RESTORE_MARK in t for t in _message_texts(m))
            with ORIGINALS_LOCK:
                hit = ORIGINALS.get(key)
            extra = _image_count(hit[0]) - _image_count(m) if hit is not None else 0
            if (hit is not None and images + extra <= RESTORE_MAX_IMAGES
                    and (marked or _tool_images_removed(hit[0], m))):
                m = hit[0]
                images += extra
                restored += 1
                with ORIGINALS_LOCK:
                    if key in ORIGINALS:
                        ORIGINALS.move_to_end(key)
            elif remember and not marked and _image_count(m):
                size = len(json.dumps(m))
                with ORIGINALS_LOCK:
                    old = ORIGINALS.pop(key, None)
                    if old is not None:
                        ORIGINALS_BYTES -= old[1]
                    if size <= RESTORE_MAX:
                        ORIGINALS[key] = (m, size)
                        ORIGINALS_BYTES += size
                        while ORIGINALS_BYTES > RESTORE_MAX:
                            ORIGINALS_BYTES -= ORIGINALS.popitem(last=False)[1][1]
        h.update(json.dumps(m, sort_keys=True).encode())
        out.append(m)
    return ({**body, "messages": out} if restored else body), restored


def aliases():
    """The extra ids --alias-model lets clients ask for, each with a card; --served-model itself is never an alias."""
    return tuple(getattr(ARGS, "alias_model", ()) or ())


def hidden_aliases():
    """The ids --hidden-alias lets clients ask for without a card: an old name still in clients' configs."""
    return tuple(getattr(ARGS, "hidden_alias", ()) or ())


def asked_alias(asked):
    """The id a reply names when the request asked for an alias (listed or hidden); None otherwise."""
    return asked if asked in aliases() or asked in hidden_aliases() else None


def canonical_model(asked):
    """The id TensorFold is asked for: an alias maps onto --served-model, any other id passes through unchanged."""
    return ARGS.served_model if asked in aliases() or asked in hidden_aliases() else asked


def to_openai(body, *, count_only=False):
    """An Anthropic Messages (or count_tokens) request as an OpenAI chat completion request, plus the types of the
    tools left out (server tools and toolsets that cannot run here)."""
    if not isinstance(body.get("messages"), list):
        raise RequestError("messages: Field required")
    req = {"model": canonical_model(body.get("model")) or ARGS.served_model, "messages": convert_messages(body)}
    if STABLE_TIME:
        req["messages"] = stabilize_time(req["messages"], remember=not count_only)
    if body.get("chat_template_kwargs"):
        req["chat_template_kwargs"] = body["chat_template_kwargs"]
    if not count_only:
        if not isinstance(body.get("max_tokens"), int):
            raise RequestError("max_tokens: Field required")
        req["max_tokens"] = body["max_tokens"]
        for src, dst in (("stop_sequences", "stop"), ("temperature", "temperature"), ("top_p", "top_p"),
                         ("top_k", "top_k"), ("seed", "seed")):
            if body.get(src) is not None:
                req[dst] = body[src]
        if PIN_TEMPERATURE is not None and "temperature" not in req:
            req["temperature"] = PIN_TEMPERATURE      # a client-sent temperature was copied above
        if body.get("stream"):
            req["stream"] = True
            req["stream_options"] = {"include_usage": True}
        config = body.get("output_config") or {}
        fmt = config.get("format") or {}
        if fmt.get("json_schema"):
            req["response_format"] = {"type": fmt.get("type"), "json_schema": {"name": fmt.get("type"),
                                                                              "schema": fmt["json_schema"]}}
    # Token counting must select the same template as generation (effort line and opened/closed think block).
    config = body.get("output_config") or {}
    if inkling():
        # Inkling always reasons: a level for every request (disabled -> minimal, budget_tokens -> a level, default
        # high), never none, and no thinking_budget (README.md, "Inkling 2")
        req["reasoning_effort"] = inkling_messages_effort(body)
    else:
        if config.get("effort") is not None:
            req["reasoning_effort"] = config["effort"]
        thinking = body.get("thinking") or {}
        if thinking.get("type") == "disabled" or (not thinking and config.get("effort") is None):
            # Anthropic's API thinks only when asked (thinking or effort set); the template would otherwise think at
            # max effort by default. Claude Code's per-action security monitor sends neither and waits for its verdict.
            req["reasoning_effort"] = "none"      # TensorFold: answer without thinking
        elif thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens")
            if not isinstance(budget, int) or budget < 1024:
                raise RequestError("thinking.enabled.budget_tokens: Input should be greater than or equal to 1024")
            if not count_only:
                req["thinking_budget"] = budget      # no generated tokens in count_tokens
    choice = None
    if (tc := body.get("tool_choice")) is not None:
        if not count_only:
            req["parallel_tool_calls"] = not tc.get("disable_parallel_tool_use", False)
        choice = {"auto": "auto", "any": "required", "none": "none"}.get(tc.get("type"))
        if tc.get("type") == "tool":
            choice = {"type": "function", "function": {"name": tc.get("name")}}
    ignored = []
    if body.get("tools") is not None:
        tools = []
        for i, tool in enumerate(body["tools"]):
            kind, description, parameters = tool.get("type"), tool.get("description"), tool.get("input_schema")
            if kind not in (None, "custom"):
                spec = client_tool(kind) if tool.get("name") else None
                if spec is None:          # server tools run on Anthropic's side; computer/browser need screenshots
                    ignored.append(kind)
                    continue
                description, parameters = spec
            elif parameters is None:
                raise RequestError(f"tools.{i}.input_schema: Field required")
            fn = {"name": tool.get("name"), "description": description, "parameters": parameters}
            if tool.get("strict") is not None:
                fn["strict"] = tool["strict"]
            if "defer_loading" in tool:
                if type(tool["defer_loading"]) is not bool:
                    raise RequestError(f"tools.{i}.defer_loading must be a boolean")
                fn["defer_loading"] = tool["defer_loading"]
            tools.append({"type": "function", "function": fn})
        if tools:
            if isinstance(choice, dict) and choice["function"]["name"] not in {t["function"]["name"] for t in tools}:
                choice = "auto"           # the named tool was one that was left out
            req["tools"], req["tool_choice"] = tools, choice or "auto"
    if inkling():
        req["messages"], restored = recall_reasoning(req["messages"])
        if restored:
            req["_inkling_restored"] = restored      # for the request log; taken out before the request is sent
    known = {t["function"]["name"] for t in req.get("tools", [])}
    for message in req["messages"]:
        for part in message.get("content") if isinstance(message.get("content"), list) else ():
            if part.get("type") == "tool_reference" and part["name"] not in known:
                raise RequestError(f"tool_result references unknown tool {part['name']!r}; include its schema in tools")
    return req, ignored


def anthropic_usage(usage):
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0}
    out = {"output_tokens": usage.get("completion_tokens") or 0}
    input_tokens = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None:                # Anthropic: total input = input_tokens + cache reads
        input_tokens -= cached
        out["cache_read_input_tokens"] = cached
    out["input_tokens"] = max(0, input_tokens)
    return out


def stop_fields(finish, matched):
    if finish == "stop" and isinstance(matched, str):
        return "stop_sequence", matched
    return STOP_REASONS.get(finish or "stop", "end_turn"), None


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def text_tool_call(block, tool_names):
    """Parse one complete <tool_call>{json}</tool_call> block into a normalised call."""
    start, end = block.find(TOOL_CALL_OPEN), block.rfind(TOOL_CALL_CLOSE)
    if start < 0 or end < start:
        return None
    try:
        obj = json.loads(block[start + len(TOOL_CALL_OPEN):end].strip())
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else {}
    name = obj.get("name") or fn.get("name")
    args = obj.get("arguments", fn.get("arguments"))
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(args, str):
        try:
            json.loads(args or "{}")
        except ValueError:
            return None
        rendered = args or "{}"
    elif isinstance(args, dict):
        rendered = json.dumps(args)
    else:
        return None
    match = next((t for t in tool_names if isinstance(t, str) and t.lower() == name.strip().lower()), None)
    return {"name": match, "arguments": rendered} if match else None


def hoist_text_tool_calls(text, tool_names, calls):
    """Strip promoted call blocks out of text. Unpromotable blocks stay visible."""
    if not isinstance(text, str) or TOOL_CALL_OPEN not in text:
        return text
    out, pos = [], 0
    while True:
        start = text.find(TOOL_CALL_OPEN, pos)
        if start < 0:
            out.append(text[pos:]); break
        end = text.find(TOOL_CALL_CLOSE, start)
        if end < 0:
            out.append(text[pos:]); break
        end += len(TOOL_CALL_CLOSE)
        call = text_tool_call(text[start:end], tool_names)
        if call:
            out.append(text[pos:start]); calls.append(call)
        else:
            out.append(text[pos:end])
        pos = end
    return "".join(out)


def to_anthropic(resp, tool_names=(), model=None):
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    stop_reason, stop_sequence = stop_fields(choice.get("finish_reason"), choice.get("stop_reason"))
    content = []
    if reasoning := msg.get("reasoning_content") or msg.get("reasoning"):
        content.append({"type": "thinking", "thinking": reasoning, "signature": uuid.uuid4().hex})
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for call in msg.get("tool_calls") or []:
        content.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                        "input": json.loads(call["function"].get("arguments") or "{}")})
    if not any(block["type"] == "tool_use" for block in content):
        loose = []
        for block in content:
            if block["type"] == "thinking":
                block["thinking"] = hoist_text_tool_calls(block["thinking"], tool_names, loose)
            elif block["type"] == "text":
                block["text"] = hoist_text_tool_calls(block["text"], tool_names, loose)
        content = [b for b in content
                   if not (b["type"] in ("text", "thinking") and not b.get(b["type"]))]
        for call in loose:
            content.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex, "name": call["name"],
                            "input": json.loads(call["arguments"])})
    if any(block["type"] == "tool_use" for block in content) and choice.get("finish_reason") in (None, "stop") and stop_sequence is None:
        stop_reason = "tool_use"
    if not content:                       # strict clients assume content[0] exists
        content.append({"type": "text", "text": ""})
    return {"id": resp.get("id"), "type": "message", "role": "assistant", "content": content,
            "model": model or resp.get("model"),
            "stop_reason": stop_reason, "stop_sequence": stop_sequence, "usage": anthropic_usage(resp.get("usage"))}


def event(name, obj):
    return f"event: {name}\ndata: {json.dumps(obj, separators=(',', ':'))}\n\n"


PING = event("ping", {"type": "ping"}).encode()


def keepalive(chunk):
    """TensorFold's keep-alive: a chunk with one choice, an empty delta and no finish."""
    choices = chunk.get("choices") if isinstance(chunk, dict) else None
    return bool(choices) and not (choices[0].get("delta") or {}) and choices[0].get("finish_reason") is None


class StreamConverter:
    """OpenAI chunks to sequential Anthropic blocks, including parallel tool deltas.

    OpenAI tool indexes can interleave and IDs/names can arrive separately. Hold
    those blocks until [DONE], rather than closing a half-written call when the
    next index starts. Ordinary text/thinking before the tools still streams.
    """

    def __init__(self, tool_names=(), model=None):
        self.model = model
        self.index, self.started, self.finish, self.matched, self.usage = 0, False, None, None, None
        self.tools, self.deferred, self.buffered = {}, [], 0
        self.tool_names = tuple(n for n in tool_names if isinstance(n, str))
        self.scan_kind, self.scan_buf, self.text_calls = None, "", []
        self.tail, self.tail_kind = "", None
        self._reset()

    def _reset(self):
        self.kind = self.at = self.signature = None
        self.signed = False

    def _start(self, block):
        out = event("content_block_start", {"type": "content_block_start", "index": self.index, "content_block": block})
        self.kind, self.at = block["type"], self.index
        self.signature = uuid.uuid4().hex if block["type"] == "thinking" else None
        self.signed = block["type"] != "thinking"
        return out

    def _delta(self, delta):
        return event("content_block_delta", {"type": "content_block_delta", "index": self.at, "delta": delta})

    def _stop(self):
        if self.kind is None:
            return []
        out = []
        if self.kind == "thinking" and self.signature and not self.signed:
            out.append(self._delta({"type": "signature_delta", "signature": self.signature}))
        out.append(event("content_block_stop", {"type": "content_block_stop", "index": self.at}))
        self._reset()
        self.index += 1
        return out

    def _message_start(self, chunk):
        self.started = True
        return event("message_start", {"type": "message_start", "message": {
            "id": chunk.get("id") or f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant", "content": [],
            "model": self.model or chunk.get("model") or ARGS.served_model, "stop_reason": None, "stop_sequence": None,
            "usage": anthropic_usage(chunk.get("usage"))}})

    def _reserve(self, text):
        if not isinstance(text, str):
            raise ValueError("Upstream stream fragment is not text")
        self.buffered += len(text.encode("utf-8"))
        if self.buffered > 16 * 1024 * 1024:
            raise ValueError("Upstream pending tool content exceeds size limit")

    def _emit_content(self, kind, text):
        if self.tools:
            self._reserve(text)
            if self.deferred and self.deferred[-1][0] == kind:
                self.deferred[-1][1].append(text)
            else:
                self.deferred.append((kind, [text]))
            return []
        out = []
        if self.kind != kind:
            out += self._stop()
            out.append(self._start({"type": kind, kind: ""}))
        out.append(self._delta({"type": kind + "_delta", kind: text}))
        return out

    def _content(self, kind, text):
        if not self.tool_names:
            return self._emit_content(kind, text)
        out = []
        if self.tail:
            if self.tail_kind == kind:
                text, self.tail = self.tail + text, ""
            else:
                out += self._emit_content(self.tail_kind, self.tail)
                self.tail, self.tail_kind = "", None
        while True:
            if self.scan_kind is not None:
                self.scan_buf += text
                self._reserve(text)
                if len(self.scan_buf) > 1000000:
                    out += self._emit_content(self.scan_kind, self.scan_buf)
                    self.scan_kind, self.scan_buf = None, ""
                    return out
                end = self.scan_buf.find(TOOL_CALL_CLOSE)
                if end < 0:
                    return out
                end += len(TOOL_CALL_CLOSE)
                block, rest = self.scan_buf[:end], self.scan_buf[end:]
                scan_kind = self.scan_kind
                self.scan_kind, self.scan_buf = None, ""
                call = text_tool_call(block, self.tool_names)
                if call:
                    self.text_calls.append(call)
                else:
                    out += self._emit_content(scan_kind, block)
                text = rest
                continue
            start = text.find(TOOL_CALL_OPEN)
            if start < 0:
                hold = 0
                for k in range(min(len(TOOL_CALL_OPEN) - 1, len(text)), 0, -1):
                    if text.endswith(TOOL_CALL_OPEN[:k]):
                        hold = k
                        break
                if hold:
                    if text[:-hold]:
                        out += self._emit_content(kind, text[:-hold])
                    self.tail, self.tail_kind = text[-hold:], kind
                elif text:
                    out += self._emit_content(kind, text)
                return out
            if start:
                out += self._emit_content(kind, text[:start])
            self.scan_kind, self.scan_buf = kind, text[start:]
            text = ""

    def _tool(self, call):
        index = call.get("index")
        if index is None:
            # Some compatible servers omit index for their sole call. Do not
            # guess once several independently named calls exist.
            matches = [i for i, t in self.tools.items() if call.get("id") and t["id"] == call["id"]]
            if matches:
                index = matches[0]
            elif len(self.tools) <= 1 and not call.get("id"):
                index = next(iter(self.tools), 0)
            elif not self.tools:
                index = 0
            else:
                raise ValueError("Parallel tool delta has no index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Invalid upstream tool index")
        if index not in self.tools:
            # Insertion order is first appearance, not argument-arrival order.
            self.tools[index] = {"id": "", "name": "", "arguments": []}
            self.deferred.append(("tool", index))
        tool = self.tools[index]
        fn = call.get("function") or {}
        for name, fragment in (("id", call.get("id")), ("name", fn.get("name"))):
            if fragment is not None:
                self._reserve(fragment)
                # Most servers provide each field once; accept repeated complete
                # values as well as genuine fragments without duplicating them.
                if fragment != tool[name]:
                    tool[name] += fragment
                if len(tool[name]) > 1024:
                    raise ValueError("Upstream tool identifier exceeds size limit")
        if fn.get("arguments") is not None:
            self._reserve(fn["arguments"])
            tool["arguments"].append(fn["arguments"])

    def feed(self, chunk):
        out = [] if self.started else [self._message_start(chunk)]
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        if not chunk.get("choices"):
            return out
        choice = chunk["choices"][0]
        if choice.get("finish_reason") is not None:
            self.finish, self.matched = choice["finish_reason"], choice.get("stop_reason")
        delta = choice.get("delta") or {}
        if reasoning := delta.get("reasoning_content") or delta.get("reasoning"):
            out += self._content("thinking", reasoning)
        if text := delta.get("content"):
            out += self._content("text", text)
        for call in delta.get("tool_calls") or []:
            if not self.tools:
                out += self._stop()
            self._tool(call)
        return out

    def finish_events(self):
        out = [] if self.started else [self._message_start({})]
        if self.tail:
            out += self._emit_content(self.tail_kind, self.tail)
            self.tail, self.tail_kind = "", None
        if self.scan_kind is not None:
            out += self._emit_content(self.scan_kind, self.scan_buf)
            self.scan_kind, self.scan_buf = None, ""
        out += self._stop()
        for kind, fragments in self.deferred:
            if kind == "tool":
                tool = self.tools[fragments]
                name, tool_id = tool["name"], tool["id"]
                arguments = "".join(tool["arguments"]) or "{}"
                if not tool_id or not name:
                    # An incompletely identified call must never execute under a
                    # guessed real tool name. Android turns this inert name into
                    # one correlated is_error result, leaving sibling calls alive.
                    arguments = json.dumps({"error": "Upstream tool call omitted its identifier or name.", "requested_tool": name})
                    tool_id = tool_id or "toolu_invalid_" + uuid.uuid4().hex
                    name = "invalid_tool"
                out.append(self._start({"type": "tool_use", "id": tool_id, "name": name, "input": {}}))
                for at in range(0, len(arguments), 65536):
                    out.append(self._delta({"type": "input_json_delta", "partial_json": arguments[at:at + 65536]}))
            else:
                out.append(self._start({"type": kind, kind: ""}))
                for text in fragments:
                    out.append(self._delta({"type": kind + "_delta", kind: text}))
            out += self._stop()
        if self.text_calls and not self.tools:
            for call in self.text_calls:
                out.append(self._start({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex,
                                        "name": call["name"], "input": {}}))
                for at in range(0, len(call["arguments"]), 65536):
                    out.append(self._delta({"type": "input_json_delta",
                                            "partial_json": call["arguments"][at:at + 65536]}))
                out += self._stop()
        stop_reason, stop_sequence = stop_fields(self.finish, self.matched)
        # A few OpenAI-compatible backends use stop/None even for complete calls.
        # Preserve length/refusal/etc.; only fix the ambiguous ordinary ending.
        if (self.tools or self.text_calls) and self.finish in (None, "stop") and stop_sequence is None:
            stop_reason = "tool_use"
        out.append(event("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": stop_reason, "stop_sequence": stop_sequence}, "usage": anthropic_usage(self.usage)}))
        out.append(event("message_stop", {"type": "message_stop"}))
        return out


# --- Inkling 2 (--family inkling2): on only for the Inkling recipe; every other recipe takes none of this. ---
# The native TensorFold inkling2 server returns a reply's tool calls unparsed in content. Its chat template renders a
# call as <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>, one message a
# call, and the server drops the message markup, so a reply reads "prose" + "NAME<|...|>{json}" + "NAME<|...|>{json}"
# with no separator. The gateway parses those into OpenAI tool_calls / Anthropic tool_use. Inkling always reasons: its
# template's effort_map has none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99, max 0.99, and the engine
# takes only those names, so every request gets one, never none (README.md, "Inkling 2").
INKLING_FAMILY = "inkling2"
INKLING_CALL = "<|content_invoke_tool_json|>"
INKLING_CALL_LIMIT = 16 * 1024 * 1024        # a call's JSON larger than this is shown as text, as an unclosed one is
INKLING_LEVELS = {"minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
INKLING_DEFAULT_EFFORT = "high"              # the template's own default (0.9)
INKLING_OFF_EFFORT = "minimal"               # thinking disabled / effort none: the lowest level, never none
INKLING_BUDGETS = ((4096, "low"), (16384, "medium"), (32768, "high"))   # budget_tokens below each bound; above: max
INKLING_IMAGE_NOTE = "[image omitted: this model reads text only]"
INKLING_EFFORT_ERROR = ("{where} must be none, minimal, low, medium, high, xhigh or max, or a number from 0 to 0.99 "
                        "(Inkling always reasons: none runs as minimal)")


def inkling():
    """The gateway fronts the native inkling2 server (--family inkling2)."""
    return getattr(ARGS, "family", None) == INKLING_FAMILY


def inkling_effort(value, where="reasoning_effort"):
    """A client's effort as a level Inkling's template names: unset -> high, none -> minimal, a number from 0 to 1 ->
    the nearest level from minimal up (ties go higher)."""
    if value is None:
        return INKLING_DEFAULT_EFFORT
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "none":
            return INKLING_OFF_EFFORT
        if key in INKLING_LEVELS:
            return key
        try:
            value = float(key)
        except ValueError:
            raise RequestError(INKLING_EFFORT_ERROR.format(where=where)) from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise RequestError(INKLING_EFFORT_ERROR.format(where=where))
    number = min(max(float(value), INKLING_LEVELS["minimal"]), INKLING_LEVELS["xhigh"])
    levels = [(name, level) for name, level in INKLING_LEVELS.items() if name != "max"]
    return min(levels, key=lambda item: (round(abs(item[1] - number), 9), -item[1]))[0]   # rounded: 0.15 ties high


def inkling_budget_effort(budget):
    """Anthropic's thinking.budget_tokens as a level: under 4096 low, under 16384 medium, under 32768 high, else max."""
    for bound, level in INKLING_BUDGETS:
        if budget < bound:
            return level
    return "max"


def inkling_switches(body):
    """(chat_template_kwargs without the switches that turn Inkling's thinking off, whether one asked for off, the
    reasoning_effort found there). The engine reads kwargs.enable_thinking, kwargs.thinking and kwargs.reasoning_effort;
    Inkling's never turn reasoning off, so they go (enable_thinking stays, set true)."""
    kwargs = body.get("chat_template_kwargs")
    if kwargs is None:
        return None, False, None
    if not isinstance(kwargs, dict):
        raise RequestError("chat_template_kwargs must be a JSON object or null")
    kwargs, off = dict(kwargs), False
    if "enable_thinking" in kwargs:
        off = kwargs["enable_thinking"] is not None and not kwargs["enable_thinking"]
        kwargs["enable_thinking"] = True
    switch = kwargs.pop("thinking", None)
    if switch is False or (isinstance(switch, dict) and switch.get("type") == "disabled"):
        off = True
    return kwargs, off, kwargs.pop("reasoning_effort", None)


def inkling_thinking(thinking):
    """A thinking object as (off, the level its budget_tokens picks or None)."""
    if thinking is None:
        return False, None
    if not isinstance(thinking, dict):
        raise RequestError("thinking must be an object")
    mode = thinking.get("type")
    if mode not in ("disabled", "enabled", "adaptive"):
        raise RequestError("thinking.type must be disabled, enabled or adaptive")
    budget = _thinking_budget(thinking) if mode == "enabled" else None
    return mode == "disabled", (inkling_budget_effort(budget) if budget is not None else None)


def inkling_level(effort, thinking, kwargs_off, where):
    """The level a request runs at: off (thinking disabled, enable_thinking false) -> minimal; else the client's effort;
    else the level enabled thinking's budget_tokens picks; else high."""
    off, budget_level = inkling_thinking(thinking)
    if off or kwargs_off:
        return INKLING_OFF_EFFORT
    if effort is not None:
        return inkling_effort(effort, where)
    return budget_level or INKLING_DEFAULT_EFFORT


def inkling_chat_body(body):
    """An OpenAI chat completion for Inkling: one reasoning_effort level, never none; thinking and the kwargs' off
    switches folded into it; no thinking_budget added (budget_tokens picks a level instead)."""
    out = dict(body)
    kwargs, kwargs_off, kwargs_effort = inkling_switches(out)
    effort = out.get("reasoning_effort")
    if effort is None:
        nested = out.get("reasoning")
        effort = nested.get("effort") if isinstance(nested, dict) else None
    if effort is None:
        effort = kwargs_effort
    out["reasoning_effort"] = inkling_level(effort, out.pop("thinking", None), kwargs_off, "reasoning_effort")
    if kwargs is not None:
        out["chat_template_kwargs"] = kwargs
    out["messages"], INKLING_LOCAL.restored = recall_reasoning(out.get("messages"))
    INKLING_LOCAL.strip_ids = INKLING_REPLAY_CHECK and "return_token_ids" not in out
    if INKLING_LOCAL.strip_ids:
        out["return_token_ids"] = True            # the replay check's; taken out of the client's copy
    INKLING_LOCAL.call_stop = inkling_call_stop(out.get("parallel_tool_calls") is False, inkling_tool_names(out))
    if INKLING_LOCAL.call_stop:
        out["stop"] = inkling_stops(out.get("stop"))   # the reply ends after its first call message
    return out


def inkling_messages_effort(body):
    """An Anthropic request's level: thinking disabled -> minimal; output_config.effort; enabled thinking's
    budget_tokens; else high (Anthropic's API thinks only when asked; Inkling always does)."""
    thinking = body.get("thinking") or None
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1024:
            raise RequestError("thinking.enabled.budget_tokens: Input should be greater than or equal to 1024")
    if isinstance(thinking, dict) and thinking.get("type") not in ("disabled", "enabled", "adaptive"):
        thinking = None                   # another type: thinking stays on, as unset
    effort = (body.get("output_config") or {}).get("effort")
    return inkling_level(effort, thinking, False, "output_config.effort")


def inkling_tool_names(body):
    """The function names an Inkling reply may call: the request's tools unless tool_choice is none."""
    choice = body.get("tool_choice")
    if choice == "none" or (isinstance(choice, dict) and choice.get("type") == "none"):
        return []
    names = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        spec = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if isinstance(spec.get("name"), str) and spec["name"] and spec["name"] not in names:
            names.append(spec["name"])
    return names


# Inkling's own turns, remembered (--family inkling2). The engine reuses its prompt state up to the first token where a
# request's prompt differs from what it holds -- the previous prompt and the reply it generated -- and resumes only at
# that state's end or at a snapshot at the previous prompt's end, so a reply that comes back changed re-runs whole. A
# reply comes back as an assistant turn the template renders again: the turn must render to the model's own tokens.
# Clients lose parts of it: OpenAI-compatible chat clients and Codex never return reasoning, some strip thinking
# blocks, text that is only white space never reaches a client, and a client may write a call's arguments again in its
# own way (keys reordered, 1.0 as 1). Each reply is kept as the model wrote it -- reasoning, text, each call's name as
# written and arguments in the model's key order -- under what its client sends back (each call's id; its text with
# the message before it), and a later request's copy of the turn is put back to it where the copy is the same turn:
# reasoning missing or the same once trimmed, text the same once trimmed, arguments that parse to the same values, a
# name that differs only in case. A turn the client really changed keeps the client's version. The memory is bounded
# (LRU) and written through to GATEWAY_INKLING_MEMORY_FILE (a JSON line a reply; "" keeps it in RAM only), so a gateway
# restart keeps it: before, the first request after a restart lost the reasoning of every remembered turn at once and
# re-ran its whole prompt (4.5 minutes at 300k tokens).
INKLING_MEMORY_MAX = int(os.environ.get("GATEWAY_INKLING_MEMORY_MAX") or 8192)          # entries
INKLING_MEMORY_BYTES = int(float(os.environ.get("GATEWAY_INKLING_MEMORY_MB") or 64) * 2**20)
INKLING_MEMORY_FILE = os.environ.get("GATEWAY_INKLING_MEMORY_FILE", "/var/lib/inkling-gateway/inkling-memory.jsonl")
INKLING_MEMORY = OrderedDict()            # "call:<id>" or "text:<sha256>" -> (turn, size)
INKLING_NAMES = OrderedDict()             # a tool call's id -> its name as the model wrote it (only where it differs)
INKLING_MEMORY_SIZE = 0
INKLING_MEMORY_LOCK = threading.Lock()
INKLING_LOCAL = threading.local()         # per request: what recall put back, the new reply's record (its log line)
INKLING_STORE = {"path": None, "handle": None, "bytes": 0}   # the memory file while it works
# parallel_tool_calls false (the model is never told: its template has no place for it, and it makes several calls
# now and then). "stop" (the default): the engine stops the reply where its first call message ends -- the request
# gets the stop string INKLING_CALL_STOP, the JSON's closing braces and the message end -- so its state holds exactly
# the one call the client gets, and the next turn renders back to it. The engine's stop check reads the decoded tail
# of the reply after every token (its special tokens spelled out) and the reply ends at the token that completes the
# match: its state keeps the rows up to that token and nothing after it (core/lanes/engine.zig cuts the round's kept
# path there; the inkling2 backend commits those rows, its book's live ids with them). Its text loses the stop string,
# so the gateway puts "}}" back when the reply's stop_sequence is its own (and takes that field out of the client's
# copy). A thinking or text message that itself ends in "}}" stops there too: the reply ends after it (logged). If the
# first call's JSON ends any other way, the stop does not fire and later calls are dropped, as under "drop". "all"
# (once called "keep"): every call the model made. "drop": the first call only, no stop (the engine's way for its own
# formats): the dropped calls stay in the engine's state, so that reply re-runs whole on the next request.
INKLING_PARALLEL = (os.environ.get("GATEWAY_INKLING_PARALLEL") or "stop").strip().lower()
INKLING_PARALLEL = "all" if INKLING_PARALLEL == "keep" else INKLING_PARALLEL
INKLING_CALL_TAIL = "}}"                     # a call message's JSON ends {"name":...,"args":{...}}
INKLING_CALL_STOP = INKLING_CALL_TAIL + "<|end_message|>"
# Each reply's tokens (return_token_ids) against its re-render's (the engine's /tokenize of the turn as it will come
# back): a log line where they differ, from the first such token, with the text around it.
INKLING_REPLAY_CHECK = (os.environ.get("GATEWAY_INKLING_REPLAY_CHECK") or "1").strip().lower() not in ("0", "false", "no", "off")
INKLING_CHECKING = threading.Semaphore(1)
INKLING_MARKUP = re.compile(r"<\|[a-z_]+\|>")


def _content_text(content):
    """A message's visible text: the string, or its text parts joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
                       and p.get("type") in ("text", "input_text", "output_text"))
    return ""


def _plain_text(content):
    """No content, a string, or text parts only (the engine joins those into one text message)."""
    return content is None or isinstance(content, str) or (isinstance(content, list) and all(
        isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text") and isinstance(p.get("text"), str)
        for p in content))


def _fingerprint(message):
    """The message before a turn, as a short hash: a reply without calls is kept under its text and this, so two
    replies that say the same ("Done.") after different messages stay apart."""
    if not isinstance(message, dict):
        return ""
    key = [message.get("role"), _content_text(message.get("content")), message.get("tool_call_id")]
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16]


def _said_key(text, before=""):
    return "text:" + hashlib.sha256((before + "\n" + text.strip()).encode()).hexdigest()


def _turn_keys(turn):
    keys = [f"call:{c['id']}" for c in turn.get("calls") or [] if isinstance(c, dict) and isinstance(c.get("id"), str)
            and c["id"]]
    if isinstance(turn.get("text"), str) and turn["text"].strip():
        keys.append(_said_key(turn["text"], turn.get("before") or ""))
    return keys


def _turn_size(turn):
    return 160 + sum(len(turn.get(k) or "") for k in ("reasoning", "text")) + sum(
        96 + len(c.get("args") or "") for c in turn.get("calls") or [] if isinstance(c, dict))


def _memory_put(keys, turn):
    """``turn`` under ``keys`` (INKLING_MEMORY_LOCK held); the oldest entries go past the bounds."""
    global INKLING_MEMORY_SIZE
    size = _turn_size(turn)
    for key in keys:
        old = INKLING_MEMORY.pop(key, None)
        if old is not None:
            INKLING_MEMORY_SIZE -= old[1]
        INKLING_MEMORY[key] = (turn, size)
        INKLING_MEMORY_SIZE += size
    while INKLING_MEMORY and (len(INKLING_MEMORY) > INKLING_MEMORY_MAX or INKLING_MEMORY_SIZE > INKLING_MEMORY_BYTES):
        INKLING_MEMORY_SIZE -= INKLING_MEMORY.popitem(last=False)[1][1]


def remember_turn(turn):
    """Keep a finished reply -- {"reasoning", "text", "calls": [{"id", "name", "written", "args"}], "before", "notes",
    "prompt", "out", "cached"} -- under its calls' ids and its text."""
    keys = _turn_keys(turn)
    if not keys:
        return
    with INKLING_MEMORY_LOCK:
        _memory_put(keys, turn)
        _store_write(keys, turn)


def remember_reply(reasoning, text, call_ids, before=""):
    """remember_turn for a reply known by its reasoning, text and call ids alone."""
    remember_turn({"reasoning": reasoning if isinstance(reasoning, str) else "",
                   "text": text if isinstance(text, str) else "",
                   "calls": [{"id": i} for i in call_ids if isinstance(i, str) and i], "before": before})


def remember_name(call_id, written, declared):
    """A call whose name the model wrote differently from the declared one (only its case can differ)."""
    if not call_id or not isinstance(written, str) or not written or written == declared:
        return
    with INKLING_MEMORY_LOCK:
        INKLING_NAMES[call_id] = written
        INKLING_NAMES.move_to_end(call_id)
        while len(INKLING_NAMES) > INKLING_MEMORY_MAX:
            INKLING_NAMES.popitem(last=False)


def _store_line(keys, turn):
    return json.dumps({"keys": keys, "turn": turn}, ensure_ascii=False, separators=(",", ":")) + "\n"


def _store_fail(what, e):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} inkling memory file {INKLING_STORE['path']}: {what} failed ({e!r}); "
          "the memory stays in RAM only", flush=True)
    handle, INKLING_STORE["handle"], INKLING_STORE["path"] = INKLING_STORE["handle"], None, None
    if handle is not None:
        try:
            handle.close()
        except OSError:
            pass


def _store_write(keys, turn):
    """One remembered reply onto the memory file (INKLING_MEMORY_LOCK held); rewritten compact past twice the bound."""
    handle = INKLING_STORE["handle"]
    if handle is None:
        return
    try:
        line = _store_line(keys, turn)
        handle.write(line)
        handle.flush()
        INKLING_STORE["bytes"] += len(line)
    except (OSError, TypeError, ValueError) as e:
        return _store_fail("a write", e)
    if INKLING_STORE["bytes"] > 2 * INKLING_MEMORY_BYTES + 2**20:
        _store_compact()


def _store_compact():
    """The memory file rewritten from the memory (INKLING_MEMORY_LOCK held): each kept reply once, least recent first."""
    path = INKLING_STORE["path"]
    if path is None:
        return
    groups = OrderedDict()                    # id(turn) -> (turn, its keys), placed by each reply's most recent key
    for key, (turn, _) in INKLING_MEMORY.items():
        entry = groups.pop(id(turn), None) or (turn, [])
        entry[1].append(key)
        groups[id(turn)] = entry
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        if INKLING_STORE["handle"] is not None:
            INKLING_STORE["handle"].close()
            INKLING_STORE["handle"] = None
        size = 0
        with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
            for turn, keys in groups.values():
                line = _store_line(keys, turn)
                f.write(line)
                size += len(line)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        INKLING_STORE["handle"] = open(path, "a", encoding="utf-8")
        INKLING_STORE["bytes"] = size
    except (OSError, TypeError, ValueError) as e:
        _store_fail("a rewrite", e)


def load_inkling_memory(path):
    """At start (--family inkling2): the memory file read back (oldest first, so the bounds keep the newest), then
    rewritten compact and kept open for appends. A line that does not read (a crash's last one) is skipped."""
    if not path:
        return
    INKLING_STORE["path"] = path
    read = bad = 0
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        keys, turn = entry["keys"], entry["turn"]
                        if not (isinstance(keys, list) and isinstance(turn, dict)):
                            raise ValueError("not a remembered reply")
                    except (KeyError, TypeError, ValueError):
                        bad += 1
                        continue
                    with INKLING_MEMORY_LOCK:
                        _memory_put([k for k in keys if isinstance(k, str)], turn)
                    read += 1
        with INKLING_MEMORY_LOCK:
            _store_compact()
    except OSError as e:
        return _store_fail("reading it back", e)
    if INKLING_STORE["path"] is not None:
        print(f"inkling memory: {len(INKLING_MEMORY)} entries from {path} ({read} replies read"
              f"{f', {bad} unreadable lines skipped' if bad else ''})", flush=True)


def close_inkling_memory():
    """The memory file rewritten compact (most recent use last) before the gateway exits."""
    with INKLING_MEMORY_LOCK:
        _store_compact()
        if INKLING_STORE["handle"] is not None:
            try:
                INKLING_STORE["handle"].close()
            except OSError:
                pass
            INKLING_STORE["handle"] = None


def _same_args(given, kept):
    """Whether a call's arguments as a client sent them are the model's: the same values once parsed."""
    try:
        a = json.loads(given) if isinstance(given, str) else given
        b = json.loads(kept) if isinstance(kept, str) else kept
    except ValueError:
        return False
    return isinstance(a, dict) and a == b


def _recall_turn(m, before):
    """The remembered reply an assistant turn is (INKLING_MEMORY_LOCK held): by a call's id, else by its text."""
    keys = [f"call:{c['id']}" for c in m.get("tool_calls") or [] if isinstance(c, dict) and isinstance(c.get("id"), str)]
    said = _content_text(m.get("content"))
    if said.strip():
        keys += [_said_key(said, _fingerprint(before)), _said_key(said)]
    for key in keys:
        hit = INKLING_MEMORY.get(key)
        if hit is not None:
            INKLING_MEMORY.move_to_end(key)
            return hit[0]
    return None


def _restore(m, turn, seen):
    """An assistant turn put back to its remembered reply where the client's copy is the same turn."""
    out = m
    kept, given = turn.get("reasoning") or "", m.get("reasoning_content")
    if isinstance(kept, str) and kept and (not isinstance(given, str) or not given.strip()):
        out, seen["reasoning"] = {**out, "reasoning_content": kept}, seen["reasoning"] + 1
    elif isinstance(kept, str) and kept and given != kept and given.strip() == kept.strip():
        out, seen["trimmed"] = {**out, "reasoning_content": kept}, seen["trimmed"] + 1
    text, content = turn.get("text"), out.get("content")
    if isinstance(text, str) and _plain_text(content):
        said = _content_text(content)
        if said != text and said.strip() == text.strip():
            out, seen["text"] = {**out, "content": text}, seen["text"] + 1
    calls = out.get("tool_calls")
    if isinstance(calls, list) and calls:
        kept_calls = {c["id"]: c for c in turn.get("calls") or [] if isinstance(c, dict) and c.get("id")}
        fixed, changed = [], False
        for c in calls:
            k = kept_calls.get(c.get("id")) if isinstance(c, dict) else None
            fn = c.get("function") if isinstance(c, dict) and isinstance(c.get("function"), dict) else None
            if fn is None:
                fixed.append(c)
                continue
            new = fn
            written = (k or {}).get("written") or INKLING_NAMES.get(c.get("id"))
            if (isinstance(written, str) and written and fn.get("name") != written
                    and str(fn.get("name", "")).lower() == written.lower()):
                new, seen["names"] = {**new, "name": written}, seen["names"] + 1
            args = (k or {}).get("args")
            if isinstance(args, str) and fn.get("arguments") != args and _same_args(fn.get("arguments"), args):
                new, seen["args"] = {**new, "arguments": args}, seen["args"] + 1
            changed |= new is not fn
            fixed.append(c if new is fn else {**c, "function": new})
        if changed:
            out = {**out, "tool_calls": fixed}
    return out


def recall_reasoning(messages):
    """(chat messages with each assistant turn put back as the model wrote it, how many turns got their reasoning back
    from memory). A turn's own reasoning stays unless it is the remembered one trimmed. INKLING_LOCAL.recall: what else
    came back, and the last assistant turn's remembered reply (the request log's prefix note)."""
    seen = {"reasoning": 0, "trimmed": 0, "text": 0, "names": 0, "args": 0, "last": None}
    INKLING_LOCAL.recall = seen
    if not isinstance(messages, list):
        return messages, 0
    out = []
    with INKLING_MEMORY_LOCK:
        for i, m in enumerate(messages):
            if not (isinstance(m, dict) and m.get("role") == "assistant"):
                out.append(m)
                continue
            if (not (isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip())
                    and isinstance(m.get("reasoning"), str) and m["reasoning"].strip()):
                m = {**m, "reasoning_content": m["reasoning"]}       # OpenRouter's spelling
            turn = _recall_turn(m, messages[i - 1] if i else None)
            seen["last"] = turn
            if turn is not None:
                m = _restore(m, turn, seen)
            elif any(isinstance(c, dict) and c.get("id") in INKLING_NAMES for c in m.get("tool_calls") or []):
                m = _restore(m, {}, seen)                            # only a written name is known
            out.append(m)
    return out, seen["reasoning"]


def inkling_call_json(name, args):
    """A call's JSON as the template writes it back: {"name":NAME,"args":{...}}, the args' keys sorted, each value
    compact, non-ASCII as is (its tojson is Python's json.dumps without ensure_ascii; floats as Python spells them)."""
    return ('{"name":' + json.dumps(name, ensure_ascii=False) + ',"args":{' + ",".join(
        json.dumps(k, ensure_ascii=False) + ":" + json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        for k, v in sorted(args.items())) + "}}")


def inkling_text_notes(text):
    """Markup the engine left in a reply's text (a second thinking message, an unparsed call): text to the client,
    and never the model's tokens again once it comes back."""
    return [f"markup in text: {m}" for m in dict.fromkeys(INKLING_MARKUP.findall(text or ""))][:4]


def inkling_rerender(turn):
    """The chat message a remembered reply comes back as once restored: what the engine renders for it next time."""
    m = {"role": "assistant", "content": turn.get("text") or ""}
    if turn.get("reasoning"):
        m["reasoning_content"] = turn["reasoning"]
    calls = [c for c in turn.get("calls") or [] if isinstance(c, dict) and isinstance(c.get("args"), str)]
    if calls:
        m["tool_calls"] = [{"id": c["id"], "type": "function",
                            "function": {"name": c.get("written") or c["name"], "arguments": c["args"]}} for c in calls]
    return m


def inkling_replay_check(turn, ids, finish):
    """The reply's token ids against the engine's tokens for its re-render, in the background: a log line from the
    first token where they differ (one check at a time; the rest are skipped)."""
    if not (INKLING_REPLAY_CHECK and isinstance(ids, list) and ids and all(type(t) is int for t in ids)
            and finish in ("stop", "tool_calls", "length") and INKLING_CHECKING.acquire(blocking=False)):
        return
    threading.Thread(target=_replay_check, args=(turn, list(ids), finish), daemon=True, name="inkling-replay").start()


def _replay_check(turn, ids, finish):
    stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%S")                     # noqa: E731
    try:
        with upstream("/tokenize", {"model": ARGS.served_model, "messages": [inkling_rerender(turn)],
                                    "add_generation_prompt": False}, timeout=30) as r:
            got = json.load(r).get("tokens")
        if not isinstance(got, list) or ids[0] not in got:
            return
        mine = got[got.index(ids[0]):]
        n = min(len(mine), len(ids))
        at = next((i for i in range(n) if mine[i] != ids[i]), None)
        if at is None and len(mine) >= len(ids):
            return                                # the turn renders to the reply's own tokens (a cut one: their start)
        at = n if at is None else at

        def words(seq):
            if not seq:
                return ""
            with upstream("/detokenize", {"tokens": seq}, timeout=30) as r:
                return json.load(r).get("prompt") or ""

        calls = turn.get("calls") or []
        what = f"call {calls[0].get('id')}" if calls else "text reply"
        notes = "; ".join(turn.get("notes") or [])
        print(f"{stamp()} inkling replay: the reply ({what}) re-renders differently from token {at} of {len(ids)} "
              f"({len(ids) - at} before its end): after {words(ids[max(0, at - 8):at])!r} the reply has "
              f"{words(ids[at:at + 12])!r}, its re-render {words(mine[at:at + 12])!r}"
              f"{'; ' + notes if notes else ''}", flush=True)
    except Exception as e:                        # noqa: BLE001 -- a diagnostic: never the request's problem
        print(f"{stamp()} inkling replay check failed: {e!r}", flush=True)
    finally:
        INKLING_CHECKING.release()


def inkling_single(asked):
    """Whether a reply keeps its first call only: parallel_tool_calls false, unless GATEWAY_INKLING_PARALLEL=all."""
    return bool(asked) and INKLING_PARALLEL != "all"


def inkling_call_stop(asked, tool_names):
    """Whether the engine stops the reply after its first call message: parallel_tool_calls false with tools offered,
    under GATEWAY_INKLING_PARALLEL=stop (the default)."""
    return bool(asked) and bool(tool_names) and INKLING_PARALLEL not in ("all", "drop")


def inkling_stops(stop):
    """A request's stop strings with INKLING_CALL_STOP among them (the client's own kept, in order)."""
    stops = [stop] if isinstance(stop, str) else list(stop) if isinstance(stop, list) else []
    return stops + ([INKLING_CALL_STOP] if INKLING_CALL_STOP not in stops else [])


def inkling_notes(usage=None):
    """The request log's Inkling notes: what recall put back besides reasoning, whether the engine reused the last
    assistant turn's reply (by this request's cached tokens), and what the new reply's re-render cannot repeat."""
    seen = getattr(INKLING_LOCAL, "recall", None) or {}
    turn, INKLING_LOCAL.turn = getattr(INKLING_LOCAL, "turn", None), None
    notes = []
    put = [f"{label} {seen[k]}" for k, label in (("text", "text"), ("args", "arguments"), ("names", "names"),
                                                  ("trimmed", "trimmed reasoning")) if seen.get(k)]
    if put:
        notes.append("restored: " + ", ".join(put))
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(usage, dict) else None
    if cached is None and isinstance(turn, dict):
        cached = turn.get("cached")
    last = seen.get("last")
    if (isinstance(last, dict) and isinstance(cached, int) and isinstance(last.get("prompt"), int)
            and isinstance(last.get("out"), int)):
        held = last["prompt"] + last["out"] - 1   # the engine's state right after that reply
        if last["prompt"] <= cached < held:
            why = "; ".join(last.get("notes") or []) or "nothing seen in the reply: a client edit, or its own tokens"
            notes.append(f"prefix: the last reply was not reused ({held - cached} of its tokens re-ran): {why}")
        elif cached < last["prompt"]:
            notes.append(f"prefix: cached {cached} is short of the last reply's prompt {last['prompt']} (another "
                         "request ran in between, or an earlier turn changed)")
    if isinstance(turn, dict) and turn.get("notes"):
        notes.append("turn: " + "; ".join(turn["notes"]))
    return " | ".join(notes)


def _merge_text(events):
    out = []
    for kind, value in events:
        if kind == "text":
            if not value:
                continue
            if out and out[-1][0] == "text":
                out[-1] = ("text", out[-1][1] + value)
                continue
        out.append((kind, value))
    return out


class InklingCalls:
    """Inkling's tool calls in answer text, as ("text", str) and ("call", {"name", "arguments"}) events in order.

    Incremental and split-independent: feeding the text whole or in any pieces gives the same events. A call's header is
    a declared tool name right before the marker (case-insensitive; the call takes the declared spelling), and its JSON
    must be one object whose "name" (when present) names that tool and whose "args" (or "arguments") is an object.
    Anything else -- an undeclared name, malformed or unclosed JSON -- stays text, exactly as the model wrote it. While
    text could still become a header (its tail is a prefix of NAME + marker) it is held back, at most a name's length.
    ``single`` keeps the first call and drops later ones, as the engine does for its formats (parallel_tool_calls false
    under GATEWAY_INKLING_PARALLEL=drop). ``notes`` gathers what the template cannot write back as the model wrote
    it: text after a call, markup in text, a call's JSON that is not the template's (key order, spacing, ...), drops.
    """

    def __init__(self, tool_names=(), single=False):
        self.names = list(dict.fromkeys(n for n in tool_names if isinstance(n, str) and n))
        self.single = single
        self.prefixes = set()
        for name in self.names:
            head = name.lower() + INKLING_CALL
            self.prefixes.update(head[:k] for k in range(1, len(head)))
        self.longest = max((len(n) + len(INKLING_CALL) for n in self.names), default=0)
        self.pending, self.call, self.calls = "", None, 0
        self.notes, self.called, self.last = [], False, None

    def _note(self, note):
        if note not in self.notes and len(self.notes) < 8:
            self.notes.append(note)

    def _watch(self, events):
        """Notes on events the template cannot write back in order: text after a call, markup in text."""
        for kind, value in events:
            if kind == "call":
                self.called, self.last = True, "call"
                continue
            if value:
                self.last = "text"
            if value and self.called:
                self._note("text after a call (the template writes a turn's text before its calls)")
            for note in inkling_text_notes(value):
                self._note(note)
        return events

    def _call_notes(self, raw, written, obj, args):
        """Where a call's JSON as the model wrote it is not the template's rendering of it."""
        mine = inkling_call_json(written, args)
        if raw == mine:
            return
        at = next((i for i, (x, y) in enumerate(zip(raw, mine)) if x != y), min(len(raw), len(mine)))
        if obj.get("name") is not None and obj.get("name") != written:
            why = f"its JSON names {obj.get('name')!r} under the header {written!r}"
        elif list(obj) != ["name", "args"]:
            why = f"its JSON's keys are {list(obj)}"
        elif not isinstance(obj.get("args"), dict):
            why = "its args are a JSON string"
        elif list(args) != sorted(args):
            why = "its args' keys are not sorted"
        else:
            why = "spacing, escapes or number spelling"
        self._note(f"call {written}: {why} (char {at}: {raw[max(0, at - 16):at + 24]!r}, "
                   f"the template's {mine[max(0, at - 16):at + 24]!r})")

    def feed(self, text):
        out = []
        while text:
            if self.call is not None:
                text = self._consume(text, out)   # the text after the call's JSON ("" while it is still open)
            else:
                self.pending += text
                text = self._text(out)            # the text after a header's marker, the call now open
        return self._watch(_merge_text(out))

    def finish(self):
        out = []
        if self.call is not None:                 # an unclosed call: text as written
            out.append(("text", self.call["head"] + INKLING_CALL + "".join(self.call["parts"])))
            self.call = None
        out.append(("text", self.pending))
        self.pending = ""
        return self._watch(_merge_text(out))

    def _hold(self, text):
        tail = text[-self.longest:].lower() if self.longest else ""
        for i in range(len(tail)):
            if tail[i:] in self.prefixes:
                return len(tail) - i
        return 0

    def _text(self, out):
        """Advance the pending text; when a declared tool's header and marker open a call, the text after the marker."""
        while True:
            at = self.pending.find(INKLING_CALL)
            if at < 0:
                keep = self._hold(self.pending)
                cut = len(self.pending) - keep
                out.append(("text", self.pending[:cut]))
                self.pending = self.pending[cut:]
                return ""
            before = self.pending[:at]
            low = before.lower()
            found = [n for n in self.names if low.endswith(n.lower())]
            if not found:                         # no declared tool's header: the marker is text
                out.append(("text", self.pending[:at + len(INKLING_CALL)]))
                self.pending = self.pending[at + len(INKLING_CALL):]
                continue
            span = max(len(n) for n in found)
            out.append(("text", before[:len(before) - span]))
            rest = self.pending[at + len(INKLING_CALL):]
            self.call = {"head": before[len(before) - span:], "names": found, "parts": [], "size": 0,
                         "started": False, "depth": 0, "str": False, "esc": False}
            self.pending = ""
            return rest

    def _consume(self, text, out):
        """Scan one piece of the open call's JSON (each character once); the text after the call when it ends."""
        c = self.call
        depth, in_str, esc, started = c["depth"], c["str"], c["esc"], c["started"]
        for j, ch in enumerate(text):
            if not started:
                if ch in " \t\r\n":
                    continue
                if ch != "{":                     # not a JSON object: the header and marker are text
                    out.append(("text", c["head"] + INKLING_CALL + "".join(c["parts"]) + text[:j]))
                    self.call = None
                    return text[j:]
                started = True
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    self.call = None
                    out.extend(self._resolve(c, "".join(c["parts"]) + text[:j + 1]))
                    return text[j + 1:]
        c["parts"].append(text)
        c["size"] += len(text)
        c["depth"], c["str"], c["esc"], c["started"] = depth, in_str, esc, started
        if c["size"] > INKLING_CALL_LIMIT:
            out.append(("text", c["head"] + INKLING_CALL + "".join(c["parts"])))
            self.call = None
        return ""

    def _resolve(self, c, raw):
        """A closed call's events: its header's prose and the call, or everything as text when it is not one."""
        try:
            obj = json.loads(raw)
        except ValueError:
            obj = None
        name = None
        if isinstance(obj, dict):
            args = obj["args"] if "args" in obj else obj.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = None
            elif args is None:
                args = {}
            asked = obj.get("name")
            if isinstance(args, dict):
                if asked is None:
                    name = max(c["names"], key=len)
                elif isinstance(asked, str):
                    name = next((n for n in c["names"] if n == asked.strip()), None) or next(
                        (n for n in c["names"] if n.lower() == asked.strip().lower()), None)
        if name is None:
            return [("text", c["head"] + INKLING_CALL + raw)]
        written = c["head"][len(c["head"]) - len(name):]          # the header as the model wrote it
        events = [("text", c["head"][:len(c["head"]) - len(name)])]
        if not (self.single and self.calls):
            events.append(("call", {"name": name, "arguments": json.dumps(args, ensure_ascii=False), "written": written}))
            self._call_notes(raw, written, obj, args)
        else:
            self._note("a later call dropped under parallel_tool_calls false (the engine holds it: this reply re-runs)")
        self.calls += 1
        return events


def inkling_split(text, tool_names, single=False):
    """(the text without its calls, the calls) for a whole answer."""
    parser = InklingCalls(tool_names, single)
    events = _merge_text(parser.feed(text) + parser.finish())
    return "".join(v for k, v in events if k == "text"), [v for k, v in events if k == "call"]


class InklingTurn:
    """One Inkling reply as the model wrote it, gathered while the gateway relays it: its reasoning and text exactly as
    the engine sent them, its calls (id, declared name, name as written, arguments in the model's key order), what its
    re-render cannot repeat (notes), its usage and its token ids. ``before`` is the request's last message (a reply
    without calls is kept under its text and it); ``asked_one``: the client sent parallel_tool_calls false; ``strip``:
    the gateway asked for the token ids (return_token_ids) and the client did not, so they leave its copy."""

    def __init__(self, before=None, asked_one=False, strip=False, call_stop=False):
        self.before = _fingerprint(before) if isinstance(before, dict) else ""
        self.asked_one, self.strip, self.call_stop = asked_one, strip, call_stop
        self.reasoning, self.text, self.calls, self.notes, self.parsers = [], [], [], [], []
        self.usage = self.ids = self.finish = None
        self.kept = self.stopped = False
        self.closed = None                        # where the stop's "}}" went back: "answer" or "reasoning"

    def take(self, payload):
        """A chunk's or completion's usage, finish reason and token ids; True when the ids left it (it changed)."""
        if not isinstance(payload, dict):
            return False
        if isinstance(payload.get("usage"), dict):
            self.usage = payload["usage"]
        for choice in payload.get("choices") or []:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        changed = False
        if self.call_stop and payload.get("stop_sequence") == INKLING_CALL_STOP:
            self.stopped = True                   # the gateway's stop: not the client's, so not in its copy
            del payload["stop_sequence"]
            changed = True
        block = payload.get("tensorfold")
        if isinstance(block, dict) and isinstance(block.get("token_ids"), list):
            self.ids = block["token_ids"]
            if self.strip:
                payload["tensorfold"] = {k: v for k, v in block.items() if k != "token_ids"}
                changed = True
        return changed

    def call(self, call_id, event):
        self.calls.append({"id": call_id, "name": event["name"], "written": event.get("written") or event["name"],
                           "args": event["arguments"]})
        remember_name(call_id, event.get("written"), event["name"])

    def record(self):
        notes = list(dict.fromkeys(self.notes + [n for p in self.parsers for n in p.notes]))
        if self.asked_one and len(self.calls) > 1:
            notes.append(f"{len(self.calls)} calls handed over (parallel_tool_calls false; GATEWAY_INKLING_PARALLEL=all)")
        if self.closed == "reasoning":
            notes.append("the stop after a first call fired in the thinking (it ends in '}}'): the reply has no answer")
        elif self.stopped and not any(p.last == "call" for p in self.parsers[-1:]):
            notes.append("the stop after a first call ended a message that is no call (it ends in '}}')")
        usage = self.usage or {}
        return {"reasoning": "".join(self.reasoning), "text": "".join(self.text), "calls": self.calls,
                "before": self.before, "notes": notes, "prompt": usage.get("prompt_tokens"),
                "out": usage.get("completion_tokens"),
                "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}

    def remember(self):
        """Keep the reply (once) and check its re-render against its tokens; its record (None the second time)."""
        if self.kept:
            return None
        self.kept = True
        turn = self.record()
        remember_turn(turn)
        inkling_replay_check(turn, self.ids, self.finish)
        INKLING_LOCAL.turn = turn
        return turn


def inkling_reply(resp, tool_names, single=False, id_prefix="call_", turn=None):
    """(the chat completion with Inkling's calls taken out of message.content, whether it changed, its InklingTurn)."""
    turn = turn if turn is not None else InklingTurn()
    changed = turn.take(resp)
    choices = resp.get("choices") if isinstance(resp, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    msg = choice.get("message") if choice else None
    if turn.stopped and isinstance(msg, dict):    # the engine's text lost the stop string: its "}}" goes back
        if isinstance(msg.get("content"), str) and msg["content"]:
            msg, turn.closed = {**msg, "content": msg["content"] + INKLING_CALL_TAIL}, "answer"
        elif isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"]:
            msg, turn.closed = {**msg, "reasoning_content": msg["reasoning_content"] + INKLING_CALL_TAIL}, "reasoning"
        choice = {**choice, "message": msg}
        resp, changed = {**resp, "choices": [choice, *choices[1:]]}, True
        choices = resp["choices"]
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(msg, dict) and isinstance(msg.get("reasoning_content") or msg.get("reasoning"), str):
        turn.reasoning.append(msg.get("reasoning_content") or msg.get("reasoning"))
    if (not tool_names or not isinstance(msg, dict) or msg.get("tool_calls")
            or not isinstance(content, str) or INKLING_CALL not in content):
        if isinstance(content, str):
            turn.text.append(content)
            turn.notes += inkling_text_notes(content)
        for c in (msg.get("tool_calls") or []) if isinstance(msg, dict) else []:   # the engine's own (none for Inkling)
            fn = c.get("function") if isinstance(c, dict) and isinstance(c.get("function"), dict) else {}
            if isinstance(c.get("id"), str) and isinstance(fn.get("name"), str):
                turn.calls.append({"id": c["id"], "name": fn["name"], "written": fn["name"], "args": fn.get("arguments")})
        return resp, changed, turn
    parser = InklingCalls(tool_names, single)
    turn.parsers.append(parser)
    events = _merge_text(parser.feed(content) + parser.finish())
    text, calls = "".join(v for k, v in events if k == "text"), [v for k, v in events if k == "call"]
    turn.text.append(text)
    if not calls:
        return resp, changed, turn
    tool_calls = []
    for call in calls:
        call_id = id_prefix + uuid.uuid4().hex[:24]
        turn.call(call_id, call)
        tool_calls.append({"id": call_id, "type": "function",
                           "function": {"name": call["name"], "arguments": call["arguments"]}})
    msg = {**msg, "content": text if text.strip() else None, "tool_calls": tool_calls}
    choice = {**choice, "message": msg}
    if choice.get("finish_reason") in (None, "stop"):
        choice["finish_reason"] = "tool_calls"
    return {**resp, "choices": [choice, *choices[1:]]}, True, turn


def inkling_completion(resp, tool_names, single=False, id_prefix="call_"):
    """A chat completion with Inkling's calls taken out of message.content: (completion, whether it changed)."""
    return inkling_reply(resp, tool_names, single, id_prefix)[:2]


def inkling_passthrough_calls(path, data):
    """(tool names, single, the InklingTurn's context) for a relayed chat completion (names empty without tools: the
    reply is still remembered); None for anything else."""
    strip, INKLING_LOCAL.strip_ids = getattr(INKLING_LOCAL, "strip_ids", False), False
    call_stop, INKLING_LOCAL.call_stop = getattr(INKLING_LOCAL, "call_stop", False), False
    if path != CHAT_COMPLETIONS_PATH or not data:
        return None
    try:
        body = json.loads(data)
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    asked = body.get("parallel_tool_calls") is False
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    return inkling_tool_names(body), inkling_single(asked), {
        "before": messages[-1] if messages else None, "asked_one": asked, "strip": strip, "call_stop": call_stop}


def inkling_rewrite_json(payload, tool_names, single=False, context=None):
    """A buffered chat completion body with Inkling's calls parsed and the gateway's token ids taken out; any other
    body unchanged, byte for byte. The reply is remembered."""
    try:
        resp = json.loads(payload)
    except (TypeError, ValueError):
        return payload
    if not isinstance(resp, dict):
        return payload
    resp, changed, turn = inkling_reply(resp, tool_names, single, turn=InklingTurn(**(context or {})))
    turn.remember()
    return json.dumps(resp, ensure_ascii=False).encode() if changed else payload


class InklingStream:
    """An Inkling chat completion stream's chunks with its calls parsed: content deltas lose them, each call comes whole
    as one tool_calls delta (index 0, 1, ...) when its JSON closes, and the terminal chunk's stop becomes tool_calls.
    feed() returns (changed, chunks); unchanged chunks are the input itself, so a relay can keep their bytes."""

    def __init__(self, tool_names=(), single=False, id_prefix="call_", turn=None):
        self.parser = InklingCalls(tool_names, single)
        self.prefix, self.index, self.done, self.native, self.envelope = id_prefix, 0, False, False, {}
        self.turn = turn if turn is not None else InklingTurn()    # what the stream relayed, kept by remember()
        self.turn.parsers.append(self.parser)
        self.ids = []

    def summary(self):
        """(reasoning, text, call ids) of everything relayed so far."""
        return "".join(self.turn.reasoning), "".join(self.turn.text), list(self.ids)

    def remember(self):
        """The relayed reply into the memory (once): its record."""
        return self.turn.remember()

    def _note(self, chunks):
        for c in chunks:
            for choice in (c.get("choices") or []) if isinstance(c, dict) else []:
                d = choice.get("delta") if isinstance(choice, dict) and isinstance(choice.get("delta"), dict) else {}
                thought = d.get("reasoning_content") or d.get("reasoning")
                if isinstance(thought, str):
                    self.turn.reasoning.append(thought)
                if isinstance(d.get("content"), str):
                    self.turn.text.append(d["content"])
                self.ids += [t["id"] for t in d.get("tool_calls") or [] if isinstance(t, dict) and isinstance(t.get("id"), str)]
        return chunks

    def _chunk(self, delta, finish=None):
        return {**self.envelope, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    def _emit(self, events):
        out = []
        for kind, value in events:
            if kind == "text":
                out.append(self._chunk({"content": value}))
                continue
            call_id = self.prefix + uuid.uuid4().hex[:24]
            self.turn.call(call_id, value)
            out.append(self._chunk({"tool_calls": [{"index": self.index, "id": call_id, "type": "function",
                                                    "function": {"name": value["name"], "arguments": value["arguments"]}}]}))
            self.index += 1
        return out

    def feed(self, chunk):
        taken = self.turn.take(chunk)             # usage, finish and the token ids (out of the client's copy)
        changed, chunks = self._feed(chunk)
        return changed or taken, self._note(chunks)

    def _feed(self, chunk):
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if self.done or not choices or not isinstance(choices[0], dict):
            return False, [chunk]
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        self.envelope = {k: chunk[k] for k in ("id", "object", "created", "model") if k in chunk}
        if self.native:
            return False, [chunk]
        if delta.get("tool_calls"):              # the engine parsed calls itself: hand over to it
            self.native = True
            return True, self._emit(self.parser.finish()) + [chunk]
        text, finish = delta.get("content"), choice.get("finish_reason")
        text = text if isinstance(text, str) else ""
        thought = []
        if finish is not None and self.turn.stopped and self.turn.closed is None:
            # the engine's text lost the stop string: "}}" closes the call (or ends the thinking it fired in)
            if self.turn.text or text or self.parser.pending or self.parser.call is not None:
                text, self.turn.closed = text + INKLING_CALL_TAIL, "answer"
            else:
                thought, self.turn.closed = [self._chunk({"reasoning_content": INKLING_CALL_TAIL})], "reasoning"
        if not text and finish is None:
            return False, [chunk]
        events = self.parser.feed(text) if text else []
        if finish is None and events == [("text", text)]:
            return False, [chunk]
        rest = {k: v for k, v in delta.items() if k != "content"}
        if finish is None:
            out = [{**chunk, "choices": [{**choice, "delta": rest}, *choices[1:]]}] if rest else []
            return True, out + self._emit(events)
        self.done = True
        tail = _merge_text(events + self.parser.finish())
        if not self.index and not thought and tail == ([("text", text)] if text else []):
            return False, [chunk]                # nothing held back and no call: the terminal chunk as it came
        out = thought + ([self._chunk(rest)] if rest else [])
        out += self._emit(tail)
        if self.index and finish == "stop":
            finish = "tool_calls"
        return True, out + [{**chunk, "choices": [{**choice, "delta": {}, "finish_reason": finish}, *choices[1:]]}]

    def flush(self):
        """The held text at a stream's end that had no terminal chunk."""
        if self.done or self.native:
            return []
        self.done = True
        return self._note(self._emit(self.parser.finish()))


class InklingSSE:
    """InklingStream over a relayed chat completion SSE body: complete events in, rewritten events out. Events it does
    not change keep their bytes; data: [DONE] (or the body's end) first flushes held text."""

    def __init__(self, tool_names=(), single=False, context=None):
        self.stream = InklingStream(tool_names, single, turn=InklingTurn(**(context or {})))
        self.pending, self.remembered = b"", False

    def _remember(self):
        if not self.remembered:
            self.remembered = True
            self.stream.remember()

    @staticmethod
    def _data(chunks):
        return b"".join(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n" for c in chunks)

    def _event(self, block):
        line = block.strip(b"\r\n")
        if b"\n" in line or not line.startswith(b"data:"):
            return block + b"\n\n"
        raw = line[5:].strip()
        if raw == b"[DONE]":
            out = self._data(self.stream.flush()) + block + b"\n\n"
            self._remember()
            return out
        try:
            chunk = json.loads(raw)
        except ValueError:
            return block + b"\n\n"
        changed, chunks = self.stream.feed(chunk)
        return self._data(chunks) if changed else block + b"\n\n"

    def feed(self, data, *, final=False):
        self.pending += data
        out = []
        while b"\n\n" in self.pending:
            block, self.pending = self.pending.split(b"\n\n", 1)
            out.append(self._event(block))
        if final:
            if self.pending.strip():
                out.append(self._event(self.pending))
            self.pending = b""
            out.append(self._data(self.stream.flush()))
            self._remember()
        return b"".join(out)


class InklingMessagesStream:
    """StreamConverter behind InklingStream: an Inkling stream's calls become Anthropic tool_use blocks."""

    def __init__(self, tool_names=(), model=None, single=False, context=None):
        self.inner = StreamConverter((), model)
        self.calls = InklingStream(tool_names, single, id_prefix="toolu_", turn=InklingTurn(**(context or {})))

    @property
    def finish(self):
        return self.inner.finish

    @property
    def usage(self):
        return self.inner.usage

    def feed(self, chunk):
        out = []
        for part in self.calls.feed(chunk)[1]:
            out += self.inner.feed(part)
        return out

    def finish_events(self):
        out = []
        for part in self.calls.flush():
            out += self.inner.feed(part)
        self.calls.remember()
        return out + self.inner.finish_events()

# --- Inkling 2: the Responses API on the chat route (--family inkling2) ---
# The engine's own /v1/responses runs each request as its chat completion inside the engine, where Inkling's calls stay
# text, and its store keeps them as text. For Inkling the gateway runs that translation itself (a port of the engine's
# responses_translate.zig and responses.zig's Reply: the same items, events and fields), with the chat reply going
# through InklingStream first, so calls become function_call items. Its own store serves previous_response_id,
# item_reference, GET and DELETE. Every other recipe relays /v1/responses to TensorFold unchanged.
RESPONSES_PASSED = ("model", "temperature", "top_p", "top_k", "min_p", "seed", "stream", "parallel_tool_calls", "stop",
                    "draft", "ignore_eos", "priority", "return_token_ids", "chat_template_kwargs", "grammar_budget")
RESPONSES_REFUSED = (("background", "background responses are not supported: send the request and wait for it"),
                     ("conversation", "conversations are not supported: send previous_response_id or the items"),
                     ("prompt", "prompt templates are not supported: send input and instructions"),
                     ("context_management", "context management is not supported"))


def responses_item_id(path):
    """The id in /v1/responses/{id} or /responses/{id}; None for any other path."""
    for prefix in ("/v1/responses/", "/responses/"):
        if path.startswith(prefix) and path[len(prefix):] and "/" not in path[len(prefix):]:
            return path[len(prefix):]
    return None


class ResponsesStore:
    """The newest finished responses, in memory as the engine keeps them (1,024, or 256 MiB): each with the chat
    messages it added and its parent, for previous_response_id; its output items by id, for item_reference."""

    def __init__(self, limit=1024, max_bytes=256 * 2**20):
        self.lock, self.entries, self.bytes, self.limit, self.max_bytes = threading.Lock(), OrderedDict(), 0, limit, max_bytes

    def put(self, response, messages):
        parent = response.get("previous_response_id")
        entry = {"response": json.loads(json.dumps(response)), "messages": json.loads(json.dumps(messages)),
                 "parent": parent if isinstance(parent, str) else None,
                 "size": len(json.dumps(response)) + len(json.dumps(messages))}
        with self.lock:
            old = self.entries.pop(response["id"], None)
            if old is not None:
                self.bytes -= old["size"]
            self.entries[response["id"]] = entry
            self.bytes += entry["size"]
            while self.entries and (len(self.entries) > self.limit or self.bytes > self.max_bytes):
                self.bytes -= self.entries.popitem(last=False)[1]["size"]

    def get(self, rid):
        with self.lock:
            entry = self.entries.get(rid)
            return json.loads(json.dumps(entry["response"])) if entry else None

    def delete(self, rid):
        with self.lock:
            entry = self.entries.pop(rid, None)
            if entry is not None:
                self.bytes -= entry["size"]
            return entry is not None

    def conversation(self, rid):
        """Every message of the chain ending at rid, oldest first."""
        if not isinstance(rid, str):
            raise RequestError("previous_response_id must be a string")
        chain, at = [], rid
        with self.lock:
            while at is not None and len(chain) <= len(self.entries):
                entry = self.entries.get(at)
                if entry is None:
                    raise RequestError(f"previous response {at!r} is not stored here (store: false, deleted, evicted, "
                                       "or the gateway restarted): send the conversation's items instead")
                self.entries.move_to_end(at)
                chain.append(entry["messages"])
                at = entry["parent"]
        return [json.loads(json.dumps(m)) for messages in reversed(chain) for m in messages]

    def item(self, item_id):
        """A stored output item by its id."""
        with self.lock:
            for entry in reversed(self.entries.values()):
                for item in entry["response"].get("output") or []:
                    if isinstance(item, dict) and item.get("id") == item_id:
                        return json.loads(json.dumps(item))
        return None


RESPONSES = ResponsesStore()


def _responses_content(value):
    """A message item's content as chat content (text parts joined; images become Inkling's text note)."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise RequestError("message content must be a string or a list of content parts")
    texts = []
    for part in value:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in ("input_text", "output_text", "text") and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind == "refusal" and isinstance(part.get("refusal"), str):
            texts.append(part["refusal"])
        elif kind in ("input_image", "input_file", "file"):
            texts.append(INKLING_IMAGE_NOTE)       # the inkling2 server reads text only
        else:
            raise RequestError(f"content parts of type {kind!r} are not supported: send input_text or input_image")
    return "".join(texts)


def _responses_output(value):
    """A function_call_output's output as a tool message's content."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(p, dict) and p.get("type") in ("input_text", "input_image", "input_file")
                                       for p in value):
        return _responses_content(value)
    raise RequestError("a function_call_output's output must be a string or input_text and input_image parts")


def responses_messages(items, references=True):
    """Input items (or output items) as chat messages: a turn's reasoning, text and calls are one assistant message."""
    out, thought = [], None
    for item in items:
        if not isinstance(item, dict):
            raise RequestError("each input item must be an object")
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "item_reference" and references:
            found = RESPONSES.item(item.get("id")) if isinstance(item.get("id"), str) else None
            if found is None:
                raise RequestError(f"item_reference {item.get('id')!r} is not stored here (store: false, evicted, or the "
                                   "gateway restarted): send the item itself")
            item, kind = found, found.get("type") or "message"
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise RequestError("a message's role must be user, assistant, system or developer")
            m = {"role": role, "content": _responses_content(item.get("content"))}
            if role == "assistant" and thought is not None:
                m["reasoning_content"], thought = thought, None
            out.append(m)
        elif kind == "function_call":
            call_id, name = item.get("call_id"), item.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise RequestError("a function_call item needs a call_id and a name")
            call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": item.get("arguments") or "{}"}}
            if not out or out[-1].get("role") != "assistant":
                out.append({"role": "assistant", "content": ""})
            if thought is not None:
                out[-1]["reasoning_content"], thought = thought, None
            out[-1]["tool_calls"] = (out[-1].get("tool_calls") or []) + [call]
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str):
                raise RequestError("a function_call_output item needs the call_id of its call")
            out.append({"role": "tool", "tool_call_id": item["call_id"], "content": _responses_output(item.get("output"))})
        elif kind == "reasoning":
            texts = [x["text"] for x in item.get("content") or [] if isinstance(x, dict) and isinstance(x.get("text"), str)]
            texts = texts or [x["text"] for x in item.get("summary") or [] if isinstance(x, dict) and isinstance(x.get("text"), str)]
            thought = "".join(texts) or None       # encrypted-only reasoning cannot be read: the memory may supply it
        else:
            raise RequestError(f"input items of type {kind!r} are not supported: send messages, function_call, "
                               "function_call_output, reasoning and item_reference items")
    return out


def _responses_tools(raw):
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise RequestError("tools must be a list")
    out = []
    for tool in raw:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind != "function":
            raise RequestError(f"tools of type {kind!r} are not supported: this server runs function tools only")
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            raise RequestError("a function tool needs a name")
        out.append({"type": "function", "function": {k: tool[k] for k in ("name", "description", "parameters", "strict")
                                                     if tool.get(k) is not None}})
    return out


def _responses_tool_choice(choice, offered):
    if choice is None or choice in ("none", "auto", "required"):
        return choice, offered
    kind = choice.get("type") if isinstance(choice, dict) else None
    if kind == "function" and isinstance(choice.get("name"), str):
        return {"type": "function", "function": {"name": choice["name"]}}, offered
    mode = choice.get("mode", "auto") if isinstance(choice, dict) else None
    if kind == "allowed_tools" and mode in ("auto", "required"):
        allowed = choice.get("tools") or []
        if not isinstance(allowed, list) or any(not isinstance(t, dict) or t.get("type") != "function" for t in allowed):
            raise RequestError("allowed_tools may name function tools only")
        names = {t.get("name") for t in allowed}
        return mode, [t for t in offered or [] if t["function"]["name"] in names]
    raise RequestError("tool_choice must be none, auto, required, a function or allowed_tools of functions")


def _responses_format(text):
    if text is None:
        return None
    if not isinstance(text, dict):
        raise RequestError("text must be an object")
    fmt = text.get("format")
    if fmt is None or (isinstance(fmt, dict) and fmt.get("type") == "text"):
        return None
    if isinstance(fmt, dict) and fmt.get("type") == "json_object":
        return {"type": "json_object"}
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        return {"type": "json_schema", "json_schema": {k: fmt[k] for k in ("name", "schema", "strict", "description")
                                                       if fmt.get(k) is not None}}
    raise RequestError("text.format must be text, json_object or json_schema")


def responses_to_chat(body, context):
    """A Responses request (after _normalise_responses_body) as the chat completion that runs it, as the engine's
    translate does: (chat request, the response's echoed fields, the messages this turn adds, store, stream)."""
    for key, message in RESPONSES_REFUSED:
        if body.get(key):
            raise RequestError(message)
    if body.get("n") not in (None, 1):
        raise RequestError("n must be 1; multiple choices are not supported")
    if body.get("logprobs") or body.get("top_logprobs"):
        raise RequestError("logprobs are not supported by this model or backend")
    if body.get("truncation") not in (None, "disabled"):
        raise RequestError('truncation must be "disabled": a prompt too long for the context is refused')
    metadata = body.get("metadata") or {}
    if not (isinstance(metadata, dict) and len(metadata) <= 16 and all(
            len(k) <= 64 and isinstance(v, str) and len(v) <= 512 for k, v in metadata.items())):
        raise RequestError("metadata must be at most 16 string pairs (keys up to 64 characters, values up to 512)")
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise RequestError("instructions must be a string")
    given = body.get("input")
    if isinstance(given, str):
        given = [{"role": "user", "content": given}]
    if not isinstance(given, list) or not given:
        raise RequestError("input must be a string or a non-empty list of items")
    added = responses_messages(given)
    parent = body.get("previous_response_id")
    earlier = RESPONSES.conversation(parent) if parent is not None else []
    offered = _responses_tools(body.get("tools"))
    choice, offered = _responses_tool_choice(body.get("tool_choice"), offered)
    chat = {k: body[k] for k in RESPONSES_PASSED if k in body}
    chat["model"] = canonical_model(body.get("model")) or ARGS.served_model
    messages = ([{"role": "system", "content": instructions}] if instructions else []) + earlier + added
    chat["messages"], restored = recall_reasoning(messages)
    if offered is not None:
        chat["tools"] = offered
    if choice is not None:
        chat["tool_choice"] = choice
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = body["max_output_tokens"]
    reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    if reasoning.get("effort") is not None:
        chat["reasoning_effort"] = reasoning["effort"]
    if (fmt := _responses_format(body.get("text"))) is not None:
        chat["response_format"] = fmt
    if chat.get("stream"):
        chat["stream_options"] = {"include_usage": True}
    if PIN_TEMPERATURE is not None and chat.get("temperature") is None:
        chat["temperature"] = PIN_TEMPERATURE
    text = body.get("text") if isinstance(body.get("text"), dict) else {}
    echo = {"instructions": instructions, "max_output_tokens": body.get("max_output_tokens"), "metadata": metadata,
            "parallel_tool_calls": body.get("parallel_tool_calls", True), "previous_response_id": parent,
            "reasoning": {"effort": reasoning.get("effort"), "summary": reasoning.get("summary"), "context": context},
            "store": body.get("store") is not False, "temperature": body.get("temperature"),
            "text": {"format": text.get("format") or {"type": "text"}}, "tool_choice": body.get("tool_choice") or "auto",
            "tools": body.get("tools") or [], "top_p": body.get("top_p"), "truncation": "disabled",
            "user": body.get("user"), "background": False}
    return chat, echo, added, body.get("store") is not False, bool(body.get("stream")), restored


def responses_usage(usage):
    if not usage:
        return None
    prompt, generated = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
    return {"input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0},
            "output_tokens": generated,
            "output_tokens_details": {"reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0},
            "total_tokens": prompt + generated}


class ResponsesReply:
    """A Response built from chat-completion chunks, as the engine's Reply builds it: reasoning, text and calls as items
    in order; with ``emit`` (kind, event), streamed as typed events with sequence numbers. ``emit`` must serialize the
    event at once (items keep changing after)."""

    def __init__(self, base, emit=None):
        self.base, self.emit = base, emit
        self.items, self.current, self.calls, self.buffers, self.seq, self.final = [], None, {}, {}, 0, None
        self.reason = self.usage = self.stats = None

    def event(self, kind, **fields):
        if self.emit is not None:
            self.emit(kind, {"type": kind, "sequence_number": self.seq, **fields})
            self.seq += 1

    def start(self):
        self.event("response.created", response=self.base)
        self.event("response.in_progress", response=self.base)

    def _index(self, item):
        return next((i for i, x in enumerate(self.items) if x is item), 0)

    def open(self, kind, call_id="", name=""):
        self.close("completed")
        part = None
        if kind == "message":
            item = {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
            part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        elif kind == "reasoning":
            item = {"id": "rs_" + uuid.uuid4().hex, "type": "reasoning", "summary": [], "content": [], "status": "in_progress"}
            part = {"type": "reasoning_text", "text": ""}
        else:
            item = {"id": "fc_" + hashlib.sha256(call_id.encode()).hexdigest()[:32], "type": "function_call",
                    "call_id": call_id, "name": name, "arguments": "", "status": "in_progress"}
        self.items.append(item)
        self.buffers[id(item)] = []
        self.current = item
        self.event("response.output_item.added", output_index=len(self.items) - 1, item=item)
        if part is not None:
            item["content"] = [part]
            self.event("response.content_part.added", item_id=item["id"], output_index=self._index(item),
                       content_index=0, part=part)
        return item

    def close(self, status):
        item, self.current = self.current, None
        if item is None:
            return
        item["status"] = status
        where, text = self._index(item), "".join(self.buffers.get(id(item), []))
        if item["type"] == "function_call":
            item["arguments"] = text
            self.event("response.function_call_arguments.done", item_id=item["id"], output_index=where,
                       name=item["name"], arguments=text)
        else:
            part = item["content"][0]
            part["text"] = text
            if item["type"] == "message":
                self.event("response.output_text.done", item_id=item["id"], output_index=where, content_index=0,
                           text=text, logprobs=[])
            else:
                self.event("response.reasoning_text.done", item_id=item["id"], output_index=where, content_index=0, text=text)
            self.event("response.content_part.done", item_id=item["id"], output_index=where, content_index=0, part=part)
        self.event("response.output_item.done", output_index=where, item=item)

    def write(self, kind, text):
        item = self.current if self.current is not None and self.current["type"] == kind else self.open(kind)
        self.buffers[id(item)].append(text)
        if kind == "message":
            self.event("response.output_text.delta", item_id=item["id"], output_index=self._index(item), content_index=0,
                       delta=text, logprobs=[])
        else:
            self.event("response.reasoning_text.delta", item_id=item["id"], output_index=self._index(item),
                       content_index=0, delta=text)

    def delta(self, d):
        if not isinstance(d, dict):
            return
        thought = d.get("reasoning_content") or d.get("reasoning")
        if isinstance(thought, str) and thought:
            self.write("reasoning", thought)
        if isinstance(d.get("content"), str) and d["content"]:
            self.write("message", d["content"])
        for call in d.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            at = call["index"] if isinstance(call.get("index"), int) else 0
            item = self.calls.get(at)
            if item is None:
                call_id = call["id"] if isinstance(call.get("id"), str) and call["id"] else "call_" + uuid.uuid4().hex[:24]
                item = self.calls[at] = self.open("function_call", call_id, fn.get("name") or "")
            if isinstance(fn.get("arguments"), str) and fn["arguments"]:
                self.buffers[id(item)].append(fn["arguments"])
                self.event("response.function_call_arguments.delta", item_id=item["id"], output_index=self._index(item),
                           delta=fn["arguments"])

    def finish(self, reason, usage, stats):
        incomplete = reason in ("length", "content_filter")
        status = "incomplete" if incomplete else "completed"
        if not incomplete and all(i["type"] == "reasoning" for i in self.items):
            self.open("message")                 # an empty answer is still a message
        self.close(status)
        final = {**self.base, "status": status, "output": self.items, "usage": responses_usage(usage),
                 "incomplete_details": ({"reason": "content_filter" if reason == "content_filter" else "max_output_tokens"}
                                        if incomplete else None)}
        if not incomplete:
            final["completed_at"] = int(time.time())
        if stats:
            final["tensorfold"] = stats
        self.final = final
        self.event("response.incomplete" if incomplete else "response.completed", response=final)
        return final

    def fail(self, problem):
        self.close("incomplete")
        message = problem.get("message") if isinstance(problem, dict) else problem
        kind = problem.get("type") if isinstance(problem, dict) else None
        error = {"code": "invalid_prompt" if kind == "invalid_request_error" else "server_error",
                 "message": message if isinstance(message, str) and message else "the reply failed"}
        self.final = {**self.base, "status": "failed", "output": self.items, "error": error}
        self.event("response.failed", response=self.final)
        return self.final

    def chunk(self, payload):
        """One chat-completion stream chunk; the response finishes once the terminal chunk and its usage are in."""
        if self.final is not None or not isinstance(payload, dict):
            return
        if "error" in payload:
            return self.fail(payload["error"])
        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        self.delta(choice.get("delta"))
        if choice.get("finish_reason"):
            self.reason, self.stats = choice["finish_reason"], payload.get("tensorfold") or self.stats
        if payload.get("usage"):
            self.usage = payload["usage"]
        if self.reason is not None and self.usage is not None:
            self.finish(self.reason, self.usage, self.stats)

    def end(self):
        """The chat stream's [DONE] or end."""
        if self.final is None:
            if self.reason is not None:
                self.finish(self.reason, self.usage, self.stats)
            else:
                self.fail("the reply ended early")

    def completion(self, data):
        """A whole chat completion (not streamed)."""
        choice = data["choices"][0]
        message = choice.get("message") or {}
        self.delta({"reasoning_content": message.get("reasoning_content"), "content": message.get("content"),
                    "tool_calls": [{"index": i, **c} for i, c in enumerate(message.get("tool_calls") or []) if isinstance(c, dict)]})
        return self.finish(choice.get("finish_reason"), data.get("usage"), data.get("tensorfold"))

    def remember(self, keep, added, turn=None):
        """The finished response into the store (store: true) and its reply's InklingTurn into Inkling's memory."""
        final = self.final
        if final is None or final.get("status") == "failed":
            return
        if turn is not None:
            turn.remember()
        if keep:
            RESPONSES.put(final, added + responses_messages(final["output"], references=False))


def request_shape(body, nbytes, wire=0):
    """What a Messages request carries, for the request log: sizes, thinking mode, and short fingerprints of the system
    prompt and tools (requests whose fingerprints differ cannot share a cached prompt prefix)."""
    import hashlib
    fp = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]
    msgs = body.get("messages") or []
    blocks = [b for m in msgs if isinstance(m.get("content"), list) for b in m["content"] if isinstance(b, dict)]
    count = lambda kind: sum(b.get("type") == kind for b in blocks)
    thinking = (body.get("thinking") or {}).get("type", "unset")
    effort = (body.get("output_config") or {}).get("effort", "-")
    return (f"msgs {len(msgs)} tools {len(body.get('tools') or [])} images {sum(_image_count(m) for m in msgs)} "
            f"thinking-blocks {count('thinking')} {nbytes // 1024}KB{f' (gzip {wire // 1024}KB)' if wire else ''} think {thinking} effort {effort} "
            f"system {fp(system_text(body.get('system')) or '')} tools {fp(body.get('tools') or [])}")


CAPTURE_DIR = os.environ.get("GATEWAY_CAPTURE_DIR")   # debugging only: keep the last 12 converted requests there
# GATEWAY_CAPTURE_ALL=1: keep every converted request (what TensorFold renders) and nothing else, named so none collide:
# GATEWAY_CAPTURE_ALL=1: keep every converted request (what TensorFold renders), for offline analysis.
CAPTURE_ALL = os.environ.get("GATEWAY_CAPTURE_ALL") == "1"

# GATEWAY_TEMPERATURE (config/local.sh): the sampling temperature a generation request runs at when its client omits
# one. Clients that send a temperature keep it (the greedy probes' temperature 0 stays deterministic); this only pins
# what an omitted temperature means, so a request without the field gets exactly this value instead of inheriting
# whatever TensorFold's current default is. Applied on every route the gateway serves: /v1/messages (converted here,
# to_openai) and, on passthrough, /v1/chat/completions, /v1/completions and /v1/responses. Unset: requests reach
# TensorFold unchanged (engine default today: temperature 1.0, top_p 0.95).
PIN_TEMPERATURE = float(os.environ["GATEWAY_TEMPERATURE"]) if os.environ.get("GATEWAY_TEMPERATURE") else None
# Seconds a relayed SSE stream (the OpenAI routes) may stay quiet before the gateway writes ": keepalive" (0: off).
SSE_KEEPALIVE_S = float(os.environ.get("GATEWAY_SSE_KEEPALIVE_S") or 15)
PIN_TEMPERATURE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
RESPONSES_PATH = "/v1/responses"
# --family inkling2: OpenAI / Anthropic routes some clients send without the /v1 prefix (base URL http://host:8080)
INKLING_UNPREFIXED = {"/chat/completions": "/v1/chat/completions", "/completions": "/v1/completions",
                      "/responses": "/v1/responses", "/messages": "/v1/messages",
                      "/messages/count_tokens": "/v1/messages/count_tokens", "/models": "/v1/models"}
RESPONSES_REASONING_INCLUDE = "reasoning.encrypted_content"
RESPONSES_REASONING_CONTEXTS = frozenset(("current_turn", "all_turns"))
# TensorFold already renders every replayed reasoning item. Keep that behavior when a client omits context;
# an explicit current_turn request can opt out of replay below.
RESPONSES_REASONING_CONTEXT_DEFAULT = os.environ.get("GATEWAY_REASONING_CONTEXT", "all_turns")
if RESPONSES_REASONING_CONTEXT_DEFAULT not in RESPONSES_REASONING_CONTEXTS:
    RESPONSES_REASONING_CONTEXT_DEFAULT = "all_turns"
STRICT_REASONING = os.environ.get("GATEWAY_STRICT_REASONING") == "1"
REASONING_WARNING_HEADER = "X-Inkling-Reasoning-Warning"
REASONING_CONTEXT_HEADER = "X-Inkling-Reasoning-Context"


def _thinking_budget(thinking):
    if "budget_tokens" not in thinking or thinking["budget_tokens"] is None:
        return None
    budget = thinking["budget_tokens"]
    if type(budget) is not int or budget <= 0:
        raise RequestError("thinking.budget_tokens must be a positive integer")
    return budget


def _normalise_responses_body(body):
    """Adapt client-only Responses fields to TensorFold's chat-backed Responses route.

    TensorFold returns plaintext reasoning items, so ``reasoning.encrypted_content`` is a compatibility include and
    is removed before forwarding. The backend has no context switch; explicit ``current_turn`` is implemented for
    stateless item histories by omitting replayed reasoning items, while the existing all-turns behavior remains the
    default and is preserved for chained requests.
    """
    out = dict(body)
    include = out.get("include")
    if include is not None:
        if not isinstance(include, list) or any(not isinstance(value, str) for value in include):
            raise RequestError("include must be an array of strings")
        unknown = list(dict.fromkeys(value for value in include if value != RESPONSES_REASONING_INCLUDE))
        if unknown:
            values = ", ".join(repr(value) for value in unknown)
            raise RequestError(f"include contains unsupported value(s): {values}")
        out.pop("include", None)

    reasoning = out.get("reasoning")
    if reasoning is None:
        reasoning = {}
    elif not isinstance(reasoning, dict):
        raise RequestError("reasoning must be an object")
    else:
        reasoning = dict(reasoning)

    # Some OpenAI-compatible clients put this control at the top level even on Responses requests. The native
    # Responses shape is reasoning.effort, so normalize it before TensorFold translates the request.
    if reasoning.get("effort") is None and out.get("reasoning_effort") is not None:
        reasoning["effort"] = out.pop("reasoning_effort")

    thinking = out.pop("thinking", None)
    if inkling():
        # Inkling always reasons: reasoning.effort is always a level, never none (README.md, "Inkling 2")
        kwargs, kwargs_off, kwargs_effort = inkling_switches(out)
        if kwargs is not None:
            out["chat_template_kwargs"] = kwargs
        effort = reasoning.get("effort") if reasoning.get("effort") is not None else kwargs_effort
        reasoning["effort"] = inkling_level(effort, thinking, kwargs_off, "reasoning.effort")
    elif thinking is not None:
        if not isinstance(thinking, dict):
            raise RequestError("thinking must be an object")
        mode = thinking.get("type")
        if mode not in ("disabled", "enabled"):
            raise RequestError("thinking.type must be disabled or enabled")
        budget = _thinking_budget(thinking)
        if mode == "disabled":
            reasoning["effort"] = "none"
        else:
            if reasoning.get("effort") in (None, "none"):
                reasoning["effort"] = "high"
            if budget is not None:
                out["thinking_budget"] = budget

    context = reasoning.get("context")
    if context is None:
        context = RESPONSES_REASONING_CONTEXT_DEFAULT
    elif context not in RESPONSES_REASONING_CONTEXTS:
        allowed = ", ".join(sorted(RESPONSES_REASONING_CONTEXTS))
        raise RequestError(f"reasoning.context must be one of {allowed}")
    reasoning["context"] = context
    out["reasoning"] = reasoning

    if context == "current_turn" and isinstance(out.get("input"), list):
        out["input"] = [item for item in out["input"]
                        if not (isinstance(item, dict) and item.get("type") == "reasoning")]
    return out, context


def _normalise_chat_body(body):
    """Map the gateway's thinking compatibility field to TensorFold's template controls."""
    thinking = body.get("thinking")
    if thinking is None:
        return dict(body)
    if not isinstance(thinking, dict):
        raise RequestError("thinking must be an object")
    mode = thinking.get("type")
    if mode not in ("disabled", "enabled"):
        raise RequestError("thinking.type must be disabled or enabled")
    out = dict(body)
    out.pop("thinking", None)
    kwargs = out.get("chat_template_kwargs")
    if kwargs is None:
        kwargs = {}
    elif not isinstance(kwargs, dict):
        raise RequestError("chat_template_kwargs must be a JSON object or null")
    else:
        kwargs = dict(kwargs)
    if mode == "disabled":
        out["reasoning_effort"] = "none"
        kwargs["enable_thinking"] = False
    else:
        kwargs["enable_thinking"] = True
        budget = _thinking_budget(thinking)
        if budget is not None:
            out["thinking_budget"] = budget
    out["chat_template_kwargs"] = kwargs
    return out


def missing_chat_reasoning(body):
    """Return a visible warning when tool history has assistant turns without replayable reasoning."""
    if not body.get("tools"):
        return None
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    missing = [index for index, message in enumerate(messages)
               if isinstance(message, dict) and message.get("role") == "assistant"
               and not (isinstance(message.get("reasoning_content"), str)
                        and message["reasoning_content"].strip())]
    if not missing:
        return None
    positions = ", ".join(str(index) for index in missing)
    return (f"tools were supplied but assistant message(s) at index {positions} omit non-empty reasoning_content; "
            "the engine may render a shorter prompt")


def prepare_passthrough(path, data, *, strict_reasoning=False):
    """Validate and adapt JSON on OpenAI generation routes without touching unrelated passthrough traffic."""
    warning = context = None
    if data and path in (CHAT_COMPLETIONS_PATH, RESPONSES_PATH):
        try:
            body = json.loads(data)
        except ValueError:
            body = None
        if isinstance(body, dict):
            if path == RESPONSES_PATH:
                body, context = _normalise_responses_body(body)
            elif inkling():
                body = inkling_chat_body(body)   # the reasoning-history warning does not apply
            else:
                warning = missing_chat_reasoning(body)
                if warning and strict_reasoning:
                    raise RequestError(warning)
                body = _normalise_chat_body(body)
            data = json.dumps(body).encode()
    if data is not None:
        data = apply_temperature_pin(path, data)
    return data, warning, context


def _response_with_context(payload, context):
    """Add the effective reasoning context to a Responses JSON object or SSE event payload."""
    if not isinstance(payload, dict):
        return payload
    target = payload.get("response") if isinstance(payload.get("response"), dict) else payload
    if not isinstance(target, dict) or (target.get("object") != "response" and "reasoning" not in target):
        return payload
    reasoning = target.get("reasoning")
    target["reasoning"] = {**(reasoning if isinstance(reasoning, dict) else {}), "context": context}
    return payload


def _response_with_warning(payload, warning):
    if not isinstance(payload, dict) or "response" in payload:
        return payload
    warnings = payload.get("warnings")
    if warnings is None:
        warnings = []
    elif not isinstance(warnings, list):
        warnings = [str(warnings)]
    if warning not in warnings:
        warnings.append(warning)
    payload["warnings"] = warnings
    return payload


def rewrite_response_json(data, *, context=None, warning=None):
    """Rewrite a buffered JSON response while preserving non-JSON or upstream-error payloads."""
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return data
    if context is not None:
        payload = _response_with_context(payload, context)
    if warning is not None:
        payload = _response_with_warning(payload, warning)
    return json.dumps(payload).encode()


class ResponseSSE:
    """Rewrite only complete Responses SSE events, so response.created exposes the effective context."""

    def __init__(self, context):
        self.context = context
        self.pending = b""

    def _event(self, block):
        lines = block.splitlines(keepends=True)
        out = []
        for line in lines:
            if not line.startswith(b"data: "):
                out.append(line)
                continue
            ending = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
            raw = line[6:-len(ending)] if ending else line[6:]
            if raw == b"[DONE]":
                out.append(line)
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                out.append(line)
                continue
            out.append(b"data: " + json.dumps(_response_with_context(payload, self.context), separators=(",", ":")).encode()
                       + ending)
        return b"".join(out)

    def feed(self, chunk, *, final=False):
        self.pending += chunk
        out = []
        while b"\n\n" in self.pending:
            block, self.pending = self.pending.split(b"\n\n", 1)
            out.append(self._event(block) + b"\n\n")
        if final and self.pending:
            out.append(self._event(self.pending))
            self.pending = b""
        return b"".join(out)


def apply_temperature_pin(path, data):
    """A passthrough generation body with the pinned temperature filled in; unchanged when there is nothing to fill
    (pin off, empty body, not one of the generation routes, not a JSON object, or the client sent a temperature —
    an explicit null counts as omitted, the same rule to_openai applies). A body the pin cannot parse is forwarded
    untouched: pinning never breaks a request."""
    if PIN_TEMPERATURE is None or not data:
        return data
    try:
        body = json.loads(data)
    except ValueError:
        return data
    if isinstance(body, dict) and path in PIN_TEMPERATURE_PATHS and body.get("temperature") is None:
        body["temperature"] = PIN_TEMPERATURE
        return json.dumps(body).encode()
    return data


def capture(req, label):
    if not CAPTURE_DIR:
        return
    from pathlib import Path
    d = Path(CAPTURE_DIR); d.mkdir(parents=True, exist_ok=True)
    if CAPTURE_ALL:
        if label in ("stream", "plain"):
            (d / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.json").write_text(json.dumps(req))
        return
    (d / f"{time.strftime('%H%M%S')}-{label}.json").write_text(json.dumps(req))
    for old in sorted(d.glob("*.json"))[:-12]:
        old.unlink()


def usage_note(usage):
    if not usage:
        return ""
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return f"prompt {usage.get('prompt_tokens')} cached {cached} out {usage.get('completion_tokens')}"


def models_card():
    created_at = datetime.fromtimestamp(STARTED, timezone.utc).isoformat().replace("+00:00", "Z")
    if inkling():                         # the inkling2 server reads text only
        return {"object": "list", "data": [{**card, "input_modalities": ["text"], "supports_vision": False}
                                           for card in _models_card(created_at)]}
    return {"object": "list", "data": _models_card(created_at)}


def _models_card(created_at):
    return [{
        "id": model_id, "object": "model", "created": STARTED, "owned_by": ARGS.owned_by, "root": ARGS.root,
        "parent": None, "max_model_len": ARGS.context,
        "permission": [{"id": f"modelperm-{uuid.uuid4().hex[:16]}", "object": "model_permission", "created": STARTED,
                        "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": True,
                        "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False,
                        "organization": "*", "group": None, "is_blocking": False}],
        "type": "model",
        "display_name": (getattr(ARGS, "display_name", None) or model_id) if model_id == ARGS.served_model else model_id,
        "created_at": created_at,
        "max_input_tokens": ARGS.context, "max_tokens": min(ARGS.max_tokens, ARGS.context),
        "input_modalities": ["text", "image"], "output_modalities": ["text"],
        "supports_vision": True} for model_id in (ARGS.served_model, *aliases())]


def upstream(path, payload=None, method="POST", headers=None, data=None, timeout=3600):
    if payload is not None:
        data, headers = json.dumps(payload).encode(), {"Content-Type": "application/json"}
    req = urllib.request.Request(ARGS.upstream + path, data=data, method=method, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)


GENERATING = ("/v1/messages", "/v1/chat/completions", "/v1/completions", "/v1/responses")


def incoming(key, open_):
    """Tell TensorFold a generation request's body is uploading (``open_``) or handed over: its reply pre-fill
    (TF_GLM_REPLY_PREFILL) never starts under a request about to arrive. Fire and forget: a TensorFold without the
    overlay answers 400, which nothing reads. The inkling2 server has no reply pre-fill: nothing is sent."""
    if inkling():
        return

    def send():
        try:
            upstream("/tokenize", {"tf_incoming": key, "open": int(open_)}, timeout=5).close()
        except Exception:                 # noqa: BLE001
            pass

    threading.Thread(target=send, daemon=True).start()


def upstream_error(e):
    """An upstream HTTPError's status and message."""
    raw = e.read()
    try:
        err = json.loads(raw)
        message = (err.get("error") or {}).get("message") if isinstance(err.get("error"), dict) else err.get("error")
        message = message or err.get("message") or err.get("detail") or raw.decode(errors="replace")
    except ValueError:
        message = raw.decode(errors="replace")
    return e.code, str(message)[:2000]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _log(self, status, note=""):
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {self.command} {'/v1/messages/resume/[private]' if self.path.startswith(resume_spool.PREFIX) else self.path} {status} "
              f"{(time.perf_counter() - self._t0) * 1000:.0f}ms{' ' + note if note else ''}", flush=True)

    def _body(self):
        if self.headers.get("X-Tailnet-Request-ID"):
            length = int(self.headers.get("Content-Length") or 0)
            if self.headers.get("Transfer-Encoding") or not 0 < length <= 32 * 1024 * 1024:
                self.close_connection = True
                raise RequestError("Resumable requests require Content-Length between 1 byte and 32 MiB")
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while (size := int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)):
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            self.rfile.readline()
            data = b"".join(chunks)
            self._nbytes = len(data)
            return data
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length) if length else b""
        self._nbytes = len(data)
        return data

    def _send(self, status, body, ctype="application/json", note="", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)
        self._log(status, note)

    def _json(self, status, obj, note="", headers=None):
        self._send(status, json.dumps(obj).encode(), note=note, headers=headers)

    def _openai_error(self, status, message, param=None):
        self._json(status, {"error": {"message": message, "type": ERROR_TYPES.get(status, "api_error"),
                                      "param": param, "code": None}}, note=message[:200])

    def _anthropic_error(self, status, message):
        self._json(status, {"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message}},
                   note=message[:200])

    def _stream_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        if resume_spool.ENABLED and (self.headers.get("X-Tailnet-Request-ID") or self.path.startswith(resume_spool.PREFIX)):
            self.send_header("X-Tailnet-Resume", "1")
            self.send_header("X-Tailnet-Accept-Encoding", "gzip")   # clients may compress later request bodies
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _route(self):
        self._t0 = time.perf_counter()
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        # Inkling: clients configured with a base URL without /v1 (e.g. http://host:8080) post to /chat/completions;
        # treat those as the /v1 routes so the tool parsing and the reasoning mapping apply (they were relayed raw).
        return INKLING_UNPREFIXED.get(path, path) if inkling() else path

    def _resume_proxy_guard(self, path):
        # Aperture's narrow HTTP connector always injects this routing marker.
        # Never let that connector become an alternative inference endpoint,
        # including after reverse-proxy path normalization/traversal handling.
        if self.headers.get("X-Tailnet-Resume-Proxy") is None:
            return True
        if self.command in ("GET", "DELETE") and path.startswith(resume_spool.PREFIX) and resume_spool.valid_id(path[len(resume_spool.PREFIX):]):
            return True
        self.close_connection = True
        self._anthropic_error(403, "This connector only retrieves or cancels an existing model round.")
        return False

    def do_GET(self):
        path = self._route()
        if not self._resume_proxy_guard(path):
            return
        if path.startswith(resume_spool.PREFIX):
            return resume_spool.attach(self, path)
        if inkling() and (rid := responses_item_id(path)) is not None:
            return self._inkling_stored_response(rid)
        if path in ("/models", "/v1/models"):
            try:                          # the card is served only while TensorFold answers, as profile C's was
                upstream("/v1/models", method="GET", timeout=10).close()
            except Exception as e:
                return self._json(502, {"error": {"message": f"TensorFold is not answering: {e}", "type": "BadGateway",
                                                  "code": 502}})
            return self._json(200, models_card())
        self._passthrough()

    def do_POST(self):
        path = self._route()
        if not self._resume_proxy_guard(path):
            return
        if path in GENERATING:                # its body uploads now: TensorFold hears of it before it arrives
            self._incoming = uuid.uuid4().hex[:16]
            incoming(self._incoming, True)
        try:
            if path == "/v1/messages":
                return self._messages()
            if path == "/v1/messages/count_tokens":
                return self._count_tokens()
            if path == RESPONSES_PATH and inkling():
                return self._inkling_responses()
            self._passthrough()
        finally:
            self._handed()

    def _handed(self):
        """The request reached TensorFold (or failed before): no longer incoming."""
        key, self._incoming = getattr(self, "_incoming", None), None
        if key is not None:
            incoming(key, False)

    def do_PUT(self):
        path = self._route()
        if self._resume_proxy_guard(path):
            self._passthrough()

    do_PATCH = do_PUT

    def do_DELETE(self):
        path = self._route()
        if not self._resume_proxy_guard(path):
            return
        if path.startswith(resume_spool.PREFIX):
            return resume_spool.attach(self, path, cancel=True)
        if inkling() and (rid := responses_item_id(path)) is not None:
            return self._inkling_stored_response(rid, delete=True)
        self._passthrough()

    def _decoded_body(self):
        """Messages/count_tokens bodies may be gzip-compressed (Android tailnet client). Bounded to the same 32 MiB
        as plain bodies; any other Content-Encoding is refused rather than parsed as JSON."""
        data, self._wire_bytes = self._body(), 0
        encoding = self.headers.get("Content-Encoding", "").strip().lower()
        if encoding in ("", "identity"):
            return data
        if encoding != "gzip":
            raise RequestError(f"unsupported Content-Encoding {encoding!r}")
        import zlib
        inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out = inflater.decompress(data, 32 * 1024 * 1024 + 1)
        except zlib.error as e:
            raise RequestError(f"invalid gzip body: {e}") from None
        if len(out) > 32 * 1024 * 1024 or inflater.unconsumed_tail:
            raise RequestError("gzip body expands beyond 32 MiB")
        if not inflater.eof:
            raise RequestError("truncated gzip body")
        self._wire_bytes, self._nbytes = len(data), len(out)
        return out

    def _request_json(self):
        try:
            body = json.loads(self._decoded_body() or b"{}")
        except ValueError as e:
            raise RequestError(f"invalid JSON body: {e}") from None
        if not isinstance(body, dict):
            raise RequestError("the body must be a JSON object")
        return body

    def _ms(self):
        return (time.perf_counter() - self._t0) * 1000

    def _messages(self):
        try:
            body = self._request_json()
            t_read = self._ms()
            body, images_restored = restore_images(body)
            req, ignored = to_openai(body)
            restored = req.pop("_inkling_restored", 0)
            asked = asked_alias(body.get("model"))
            tool_names = [t["name"] for t in (body.get("tools") or [])
                          if isinstance(t, dict) and isinstance(t.get("name"), str)]
            if inkling():                 # Inkling's own call format; no <tool_call> hoisting
                one = req.get("parallel_tool_calls") is False
                calls, single = inkling_tool_names(req), inkling_single(one)
                context = {"before": (req.get("messages") or [None])[-1], "asked_one": one,
                           "call_stop": inkling_call_stop(one, calls)}
                if context["call_stop"]:
                    req["stop"] = inkling_stops(req.get("stop"))    # the reply ends after its first call message
                if INKLING_REPLAY_CHECK:
                    req["return_token_ids"] = True                  # the replay check's (Anthropic replies carry no ids)
                converter = lambda: InklingMessagesStream(calls, asked, single, context)   # noqa: E731
            else:
                converter = lambda: StreamConverter(tool_names, asked)            # noqa: E731
        except RequestError as e:
            return self._anthropic_error(e.status, str(e))
        t_conv = self._ms()
        note = f"left out tools: {', '.join(ignored)}" if ignored else ""
        if restored:
            note = " | ".join(x for x in (note, f"reasoning restored to {restored} turn{'s' * (restored > 1)}") if x)
        if images_restored:
            note = " | ".join(x for x in (note, f"restored {images_restored} image message{'s' * (images_restored > 1)}") if x)
        shape = request_shape(body, getattr(self, "_nbytes", 0), getattr(self, "_wire_bytes", 0))
        capture(req, "stream" if req.get("stream") else "plain")
        capture(body, "raw")
        capture({k: v for k, v in self.headers.items() if k.lower() in ("expect", "content-length", "transfer-encoding",
                 "user-agent", "connection", "content-encoding", "x-stainless-runtime", "x-stainless-package-version")}
                | {"read_ms": round(t_read), "request_version": self.request_version}, "headers")
        if self.headers.get("X-Tailnet-Request-ID"):
            self._resume_note = shape
            return resume_spool.create(self, body, req, upstream, converter, event)
        try:
            r = upstream("/v1/chat/completions", req)
            t_hdr = self._ms()
            self._handed()
        except urllib.error.HTTPError as e:
            return self._anthropic_error(*upstream_error(e))
        except Exception as e:
            return self._anthropic_error(502, f"TensorFold is not answering: {e}")
        timing = f"read {t_read:.0f} conv {t_conv - t_read:.0f} tf-headers {t_hdr - t_conv:.0f}"
        if not req.get("stream"):
            with r:
                resp = json.load(r)
            if inkling():
                resp, _, turn = inkling_reply(resp, calls, single, "toolu_", InklingTurn(**context))
                turn.remember()
            out = to_anthropic(resp, () if inkling() else tool_names, asked)
            extra = inkling_notes(resp.get("usage")) if inkling() else ""
            return self._json(200, out, note=" | ".join(x for x in (note, timing, usage_note(resp.get("usage")), shape,
                                                                    extra) if x))
        self._stream_start()
        conv, done, t_first, last_ping = converter(), False, None, 0.0
        write_lock, heartbeat_stop = threading.Lock(), threading.Event()
        self.connection.settimeout(20)

        def write_stream(data):
            with write_lock:
                self.wfile.write(data)
                self.wfile.flush()

        def heartbeat():
            while not heartbeat_stop.wait(10):
                try:
                    write_stream(PING)
                except OSError:
                    # Wake the upstream reader for this vanished subscriber.
                    try:
                        import socket
                        r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    except (AttributeError, OSError):
                        pass
                    return

        threading.Thread(target=heartbeat, daemon=True, name="messages-heartbeat").start()
        try:
            with r:
                for raw in r:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        write_stream("".join(conv.finish_events()).encode())
                        done = True
                        break
                    chunk = json.loads(data)
                    if out := conv.feed(chunk):
                        if t_first is None and any(e.startswith("event: content_block_start") for e in out):
                            t_first = self._ms()        # first visible block (thinking, text or tool call)
                        write_stream("".join(out).encode())
                    elif keepalive(chunk) and (now := time.monotonic()) - last_ping >= 1.0:
                        # TensorFold's empty delta while it holds a tool call until </tool_call>: an Anthropic ping,
                        # so the client sees the stream is alive (it got zero bytes for the whole call before)
                        last_ping = now
                        write_stream(PING)
            if not done and conv.finish in STOP_REASONS:
                # The final choice is already terminal; losing only [DONE] or a
                # trailing usage chunk must not discard completed tool calls.
                write_stream("".join(conv.finish_events()).encode())
                done = True
            if not done:
                write_stream(event("error", {"type": "error", "error": {
                    "type": "api_error", "message": "TensorFold's stream ended without [DONE]"}}).encode())
            self.wfile.flush()
            first = f" first-block {t_first - t_hdr:.0f}" if t_first is not None else ""
            extra = inkling_notes(conv.usage) if inkling() else ""
            self._log(200, " | ".join(x for x in (note, "stream", timing + first, usage_note(conv.usage), shape, extra)
                                      if x))
        except (BrokenPipeError, ConnectionResetError):
            self._log(499, "client went away")
        except Exception as e:
            try:
                write_stream(event("error", {"type": "error", "error": {"type": "api_error", "message": str(e)}}).encode())
            except OSError:
                pass
            self._log(500, f"stream: {e!r}")
        finally:
            heartbeat_stop.set()

    def _inkling_stored_response(self, rid, delete=False):
        """GET / DELETE /v1/responses/{id} from the gateway's Responses store (Inkling)."""
        if delete and RESPONSES.delete(rid):
            return self._json(200, {"id": rid, "object": "response", "deleted": True})
        found = None if delete else RESPONSES.get(rid)
        if found is not None:
            return self._json(200, found)
        self._json(404, {"error": {"message": f"no stored response has id {rid!r}", "type": "invalid_request_error",
                                   "param": "response_id", "code": None}})

    def _inkling_responses(self):
        """POST /v1/responses for Inkling: the request as its chat completion, the reply (through InklingStream) as the
        Response's items and events."""
        try:
            body, context = _normalise_responses_body(self._request_json())
            chat, echo, added, keep, streaming, restored = responses_to_chat(body, context)
        except RequestError as e:
            return self._openai_error(e.status, str(e))
        one = chat.get("parallel_tool_calls") is False
        names, single = inkling_tool_names(chat), inkling_single(one)
        strip = INKLING_REPLAY_CHECK and "return_token_ids" not in chat
        if strip:
            chat["return_token_ids"] = True       # the replay check's; taken out of the client's copy
        call_stop = inkling_call_stop(one, names)
        if call_stop:
            chat["stop"] = inkling_stops(chat.get("stop"))   # the reply ends after its first call message
        turn = InklingTurn((chat.get("messages") or [None])[-1], one, strip, call_stop)
        model = asked_alias(body.get("model")) or ARGS.served_model
        base = {"id": "resp_" + uuid.uuid4().hex, "object": "response", "created_at": int(time.time()),
                "status": "in_progress", "error": None, "incomplete_details": None, "model": model, "output": [],
                "usage": None, **echo}
        note = f"responses | {len(added)} input items -> {len(chat['messages'])} messages" + (
            f" | reasoning restored to {restored} turn{'s' * (restored > 1)}" if restored else "")
        try:
            r = upstream("/v1/chat/completions", chat)
            self._handed()
        except urllib.error.HTTPError as e:
            status, message = upstream_error(e)
            return self._openai_error(status, message)
        except Exception as e:
            return self._json(502, {"error": {"message": f"TensorFold is not answering: {e}", "type": "BadGateway", "code": 502}})
        headers = {REASONING_CONTEXT_HEADER: context}
        if not streaming:
            with r:
                resp = json.load(r)
            reply = ResponsesReply(base)
            final = reply.completion(inkling_reply(resp, names, single, turn=turn)[0])
            reply.remember(keep, added, turn)
            return self._json(200, final, headers=headers, note=" | ".join(
                x for x in (note, usage_note(resp.get("usage")), inkling_notes(resp.get("usage"))) if x))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header(REASONING_CONTEXT_HEADER, context)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        lock, stop, state = threading.Lock(), threading.Event(), {"last": time.monotonic(), "gone": False}

        def write(data):
            with lock:
                self.wfile.write(data)
                self.wfile.flush()
                state["last"] = time.monotonic()

        def emit(kind, obj):
            write(f"event: {kind}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())

        def heartbeat():                          # a long prefill: an SSE comment every SSE_KEEPALIVE_S of quiet
            while not stop.wait(1):
                if SSE_KEEPALIVE_S <= 0 or time.monotonic() - state["last"] < SSE_KEEPALIVE_S:
                    continue
                try:
                    write(b": keepalive\n\n")
                except OSError:
                    state["gone"] = True
                    try:
                        import socket
                        r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    except (AttributeError, OSError):
                        pass
                    return

        threading.Thread(target=heartbeat, daemon=True, name="responses-heartbeat").start()
        reply, stream = ResponsesReply(base, emit), InklingStream(names, single, turn=turn)
        try:
            reply.start()
            with r:
                for raw in r:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    for part in stream.feed(chunk)[1]:
                        reply.chunk(part)
            for part in stream.flush():
                reply.chunk(part)
            reply.end()
            reply.remember(keep, added, turn)
            if state["gone"]:
                raise BrokenPipeError
            self._log(200, " | ".join(x for x in (note, "stream", usage_note(reply.usage), inkling_notes(reply.usage))
                                      if x))
        except (BrokenPipeError, ConnectionResetError):
            self._log(499, "client went away")
        except Exception as e:                    # the upstream read failed mid-stream: a failed response, if it can go
            if state["gone"]:
                self._log(499, f"client went away ({e.__class__.__name__})")
            else:
                try:
                    reply.fail(f"the reply failed: {e}")
                except OSError:
                    pass
                self._log(500, f"responses stream: {e!r}")
        finally:
            stop.set()

    def _count_tokens(self):
        try:
            req, _ = to_openai(restore_images(self._request_json(), remember=False)[0], count_only=True)
        except RequestError as e:
            return self._anthropic_error(e.status, str(e))
        payload = {k: req[k] for k in ("model", "messages", "tools", "tool_choice", "chat_template_kwargs",
                                     "reasoning_effort") if k in req}
        try:
            with upstream("/tokenize", payload, timeout=120) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            return self._anthropic_error(*upstream_error(e))
        except Exception as e:
            return self._anthropic_error(502, f"TensorFold is not answering: {e}")
        n = d.get("count") or len(d.get("tokens") or [])
        self._json(200, {"input_tokens": n, "context_management": {"original_input_tokens": n}})

    def _passthrough(self):
        path = self._route()
        data = self._body() if self.command in ("POST", "PUT", "PATCH") else None
        warning = context = None
        if inkling():
            INKLING_LOCAL.strip_ids = INKLING_LOCAL.call_stop = False
        try:
            strict = STRICT_REASONING or self.headers.get("X-Inkling-Strict-Reasoning", "").lower() in {
                "1", "true", "yes", "on"}
            data, warning, context = prepare_passthrough(path, data, strict_reasoning=strict)
        except RequestError as e:
            return self._openai_error(e.status, str(e), param="include" if path == RESPONSES_PATH else "messages")
        calls = inkling_passthrough_calls(path, data) if inkling() else None
        restored = getattr(INKLING_LOCAL, "restored", 0) if calls is not None else 0
        INKLING_LOCAL.restored = 0
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            query = urllib.parse.urlsplit(self.path).query
            target = path + ("?" + query if query else "") if inkling() and path != urllib.parse.urlsplit(self.path).path.rstrip("/") else self.path
            r = upstream(target, method=self.command, headers=headers, data=data)
        except urllib.error.HTTPError as e:
            r = e
        except Exception as e:
            return self._json(502, {"error": {"message": f"TensorFold is not answering: {e}", "type": "BadGateway",
                                              "code": 502}})
        finally:
            self._handed()
        ctype = r.headers.get("Content-Type", "application/json")
        status = getattr(r, "status", None) or r.code
        response_headers = {}
        if warning is not None:
            response_headers[REASONING_WARNING_HEADER] = warning
        if context is not None:
            response_headers[REASONING_CONTEXT_HEADER] = context
        with r:
            if not ctype.startswith("text/event-stream"):
                payload = r.read()
                if status == 200 and (context is not None or warning is not None):
                    payload = rewrite_response_json(payload, context=context, warning=warning)
                if status == 200 and calls is not None:
                    payload = inkling_rewrite_json(payload, *calls)
                extra = inkling_notes() if status == 200 and calls is not None else ""
                return self._send(status, payload, ctype, headers=response_headers, note=" | ".join(
                    x for x in (f"reasoning restored to {restored} turns" if restored else "", extra) if x))
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            for name, value in response_headers.items():
                self.send_header(name, value)
            if resume_spool.ENABLED and (self.headers.get("X-Tailnet-Request-ID") or self.path.startswith(resume_spool.PREFIX)):
                self.send_header("X-Tailnet-Resume", "1")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            # A quiet stream (a long prefill, a queue wait) gets an SSE comment every SSE_KEEPALIVE_S, which clients
            # skip: Cloudflare (AI Gateway + the tunnel's edge) drops a stream silent for ~2 min. Comments go in only
            # between events, never inside one a read1() split.
            lock, stop, state = threading.Lock(), threading.Event(), {"last": time.monotonic(), "edge": True, "gone": False}
            event_rewriter = (ResponseSSE(context) if context is not None
                              else InklingSSE(*calls) if calls is not None and status == 200 else None)

            def heartbeat():
                while not stop.wait(1):
                    with lock:
                        if not state["edge"] or time.monotonic() - state["last"] < SSE_KEEPALIVE_S:
                            continue
                        try:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            state["last"] = time.monotonic()
                        except OSError:
                            state["gone"] = True
                    if state["gone"]:
                        try:                    # wake the upstream reader so TensorFold drops the vanished request
                            import socket
                            r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                        except (AttributeError, OSError):
                            pass
                        return

            if SSE_KEEPALIVE_S > 0:
                threading.Thread(target=heartbeat, daemon=True, name="passthrough-heartbeat").start()
            try:
                while chunk := r.read1(65536):
                    if event_rewriter is not None:
                        chunk = event_rewriter.feed(chunk)
                    if not chunk:
                        continue
                    with lock:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        state["last"], state["edge"] = time.monotonic(), chunk.endswith((b"\n\n", b"\r\n\r\n"))
                if event_rewriter is not None and (chunk := event_rewriter.feed(b"", final=True)):
                    with lock:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                if state["gone"]:
                    raise BrokenPipeError
                extra = inkling_notes() if event_rewriter is not None and calls is not None else ""
                self._log(status, "stream" + (f" | reasoning restored to {restored} turns" if restored else "")
                          + (f" | {extra}" if extra else ""))
            except (BrokenPipeError, ConnectionResetError):
                self._log(499, "client went away")
            except Exception as e:              # the shut upstream socket's read error (OSError, IncompleteRead, ...)
                if not state["gone"]:
                    raise
                self._log(499, f"client went away ({e.__class__.__name__})")
            finally:
                stop.set()


def main(argv=None):
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--listen", default="0.0.0.0:8080")
    ap.add_argument("--upstream", default="http://127.0.0.1:8888", help="TensorFold's API")
    ap.add_argument("--served-model", default="inkling-2-small")
    ap.add_argument("--alias-model", action="append", default=[], metavar="ID",
                    help="another model id the gateway answers to and lists beside --served-model (repeatable)")
    ap.add_argument("--context", type=int, default=327680, help="total prompt + reply window to advertise as max_model_len / max_input_tokens")
    ap.add_argument("--max-tokens", type=int, default=65536, help="the output cap to advertise")
    ap.add_argument("--display-name", default=None, metavar="NAME",
                    help="the served model's display_name on its card (default: its id)")
    ap.add_argument("--hidden-alias", action="append", default=[], metavar="ID",
                    help="another model id the gateway answers to without listing it (repeatable): an old name")
    ap.add_argument("--root", default="thinkingmachines/Inkling-2-Small-EXL3-Native")
    ap.add_argument("--owned-by", default="tensorfold")
    ap.add_argument("--family", choices=(INKLING_FAMILY,), default=os.environ.get("GATEWAY_FAMILY") or None,
                    help="the served model's reply dialect: inkling2 = the native inkling2 server (tool-call parsing, "
                         "reasoning levels, text-only card); unset = TensorFold's Python server (every other recipe)")
    ARGS = ap.parse_args(argv)
    if ARGS.family not in (None, INKLING_FAMILY):
        ap.error(f"--family / GATEWAY_FAMILY must be {INKLING_FAMILY} or unset, not {ARGS.family!r}")
    ARGS.alias_model = list(dict.fromkeys(a.strip() for a in ARGS.alias_model
                                          if a.strip() and a.strip() != ARGS.served_model))
    ARGS.hidden_alias = list(dict.fromkeys(a.strip() for a in ARGS.hidden_alias
                                           if a.strip() and a.strip() != ARGS.served_model and a.strip() not in ARGS.alias_model))
    host, port = ARGS.listen.rsplit(":", 1)
    ThreadingHTTPServer.request_queue_size = 64    # the default 5 is tight for clients that open several at once
    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    served = ", ".join((ARGS.served_model, *aliases()))
    if ARGS.display_name:
        served += f" as {ARGS.display_name!r}"
    if hidden_aliases():
        served += f"; also answers {', '.join(hidden_aliases())}"
    family = f"; family {ARGS.family}" if ARGS.family else ""
    print(f"inkling gateway on {ARGS.listen} -> {ARGS.upstream} ({served}{family})", flush=True)
    if inkling():
        import signal
        load_inkling_memory(INKLING_MEMORY_FILE)
        what = {"all": "hands over every call", "drop": "keeps the first call"}.get(INKLING_PARALLEL,
                                                                                   "stops the reply after its first call")
        print(f"inkling: parallel_tool_calls false {what} (GATEWAY_INKLING_PARALLEL={INKLING_PARALLEL}); replay check "
              f"{'on' if INKLING_REPLAY_CHECK else 'off'}", flush=True)

        def stop(signum, frame):                  # docker stop / restart: the memory file compact, then out
            close_inkling_memory()
            os._exit(0)

        signal.signal(signal.SIGTERM, stop)
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
