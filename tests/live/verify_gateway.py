#!/usr/bin/env python3
"""verify_gateway.py [--base URL] [--model ID]: the served model checked end to end through the gateway (default
http://127.0.0.1:8080), after `scripts/serve.sh start`. Standard library only; one small request per check (greedy, short).

  1. /v1/models lists inkling and its alias, text only
  2. OpenAI chat, non-streamed and streamed: reasoning present by default, an answer; the engine ran at "high"
  3. OpenAI reasoning_effort none -> the engine ran at "minimal" (Inkling never stops reasoning)
  4. OpenAI tool-call round trip: tool_calls (finish tool_calls), streamed too, then the tool result -> an answer
  5. Anthropic /v1/messages, non-streamed and streamed: a thinking block by default, text
  6. Anthropic tool-call round trip: tool_use (stop_reason tool_use), streamed too, then tool_result -> an answer
  7. Anthropic count_tokens
Prints one PASS / FAIL line per check and exits 1 if any failed.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--base", default="http://127.0.0.1:8080")
ap.add_argument("--model", default="inkling-2-small")
ap.add_argument("--max-tokens", type=int, default=1024)
args = ap.parse_args()
BASE, MODEL, MAX = args.base.rstrip("/"), args.model, args.max_tokens
MARK = "<|content_invoke_tool_json|>"
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                           "required": ["city"]}}}
A_WEATHER = {"name": "get_weather", "description": "Current weather for a city.",
             "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
ASK_TOOL = "What is the weather in Wellington right now? Use the get_weather tool; do not guess."
RESULT = '{"city": "Wellington", "temperature_c": 14, "conditions": "windy, light rain"}'
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def call(path, body=None, stream=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), time.perf_counter() - t
    if stream:
        return 200, raw.decode(), time.perf_counter() - t
    return 200, json.loads(raw), time.perf_counter() - t


def sse(text):
    """(event name, data) pairs of an SSE body."""
    out = []
    for block in text.split("\n\n"):
        name = data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if data is not None:
            out.append((name, data if data == "[DONE]" else json.loads(data)))
    return out


def openai_stream(body):
    status, text, dt = call("/v1/chat/completions", {**body, "stream": True}, stream=True)
    acc = {"content": "", "reasoning": "", "calls": [], "finish": None, "done": False}
    if status != 200:
        return status, acc, dt
    for _, data in sse(text):
        if data == "[DONE]":
            acc["done"] = True
            continue
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            acc["content"] += delta.get("content") or ""
            acc["reasoning"] += delta.get("reasoning_content") or ""
            acc["calls"] += delta.get("tool_calls") or []
            acc["finish"] = choice.get("finish_reason") or acc["finish"]
    return status, acc, dt


def anthropic_stream(body):
    status, text, dt = call("/v1/messages", {**body, "stream": True}, stream=True)
    if status != 200:
        return status, ([], None), dt
    blocks, acc, stop = {}, {}, None
    for name, data in sse(text):
        if name == "content_block_start":
            blocks[data["index"]] = data["content_block"]
        elif name == "content_block_delta":
            d = data["delta"]
            acc[data["index"]] = acc.get(data["index"], "") + (d.get("text") or d.get("thinking") or d.get("partial_json") or "")
        elif name == "message_delta":
            stop = data["delta"].get("stop_reason")
    return status, ([(blocks[i]["type"], blocks[i].get("name"), acc.get(i, "")) for i in sorted(blocks)], stop), dt


def effort_of(reply):
    return (reply.get("tensorfold") or {}).get("reasoning_effort") if isinstance(reply, dict) else None


# 1. discovery
status, models, _ = call("/v1/models")
ids = [m.get("id") for m in models.get("data", [])] if status == 200 else []
check("/v1/models lists the served id", MODEL in ids, f"ids {ids}")
check("/v1/models: text only", status == 200 and all(m.get("input_modalities") == ["text"] for m in models["data"]))
card = models["data"][0] if status == 200 and models.get("data") else {}
check("/v1/models: one card, display name Inkling-2-Small, root the combined directory",
      len(ids) == 1 and card.get("display_name") == "Inkling-2-Small"
      and card.get("root") == "thinkingmachines/Inkling-2-Small-EXL3-Native",
      f"{[(m.get('id'), m.get('display_name'), m.get('root')) for m in models.get('data', [])]}")
status, legacy, _ = call("/v1/messages", {"model": "inkling", "max_tokens": 64, "messages": [{"role": "user", "content": "Say OK."}]})
check("the afternoon's id `inkling` is still answered (not listed)", status == 200 and legacy.get("type") == "message"
      and "inkling" not in ids, f"status {status}, model {legacy.get('model') if isinstance(legacy, dict) else legacy!r}")

# 2. OpenAI chat, reasoning by default
plain = {"model": MODEL, "max_tokens": MAX, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}
status, reply, dt = call("/v1/chat/completions", plain)
msg = reply["choices"][0]["message"] if status == 200 else {}
check("OpenAI chat: an answer", status == 200 and "391" in (msg.get("content") or ""), f"{dt:.1f} s, content {str(msg.get('content'))[:80]!r}")
check("OpenAI chat: reasoning_content by default", bool(msg.get("reasoning_content")), f"{len(msg.get('reasoning_content') or '')} chars")
check("OpenAI chat: the engine ran at high", effort_of(reply) == "high", f"tensorfold.reasoning_effort {effort_of(reply)!r}")
status, acc, dt = openai_stream(plain)
check("OpenAI chat stream: reasoning deltas, an answer, [DONE]", status == 200 and acc["reasoning"] and "391" in acc["content"] and acc["done"],
      f"{dt:.1f} s, finish {acc['finish']}")

# 3. effort none -> minimal
status, reply, _ = call("/v1/chat/completions", {**plain, "reasoning_effort": "none"})
check("OpenAI reasoning_effort none runs as minimal", effort_of(reply) == "minimal", f"tensorfold.reasoning_effort {effort_of(reply)!r}")

# 4. OpenAI tool round trip
ask = {"model": MODEL, "max_tokens": MAX, "tools": [WEATHER], "messages": [{"role": "user", "content": ASK_TOOL}]}
status, reply, dt = call("/v1/chat/completions", ask)
choice = reply["choices"][0] if status == 200 else {}
calls = (choice.get("message") or {}).get("tool_calls") or []
check("OpenAI tool call: tool_calls, finish tool_calls", bool(calls) and choice.get("finish_reason") == "tool_calls"
      and calls[0]["function"]["name"] == "get_weather", f"{dt:.1f} s, {[(c['function']['name'], c['function']['arguments']) for c in calls]}")
check("OpenAI tool call: no raw markup left in content", MARK not in ((choice.get("message") or {}).get("content") or ""))
status, acc, dt = openai_stream(ask)
check("OpenAI tool call stream: tool_calls deltas, finish tool_calls", status == 200 and acc["calls"] and acc["finish"] == "tool_calls"
      and MARK not in acc["content"], f"{dt:.1f} s, {[(c['function']['name'], c['function']['arguments']) for c in acc['calls']]}")
if calls:
    follow = {**ask, "messages": ask["messages"] + [
        {"role": "assistant", "content": (choice["message"].get("content") or ""), "tool_calls": calls,
         "reasoning_content": choice["message"].get("reasoning_content") or ""},
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": RESULT}]}
    status, reply, dt = call("/v1/chat/completions", follow)
    answer = reply["choices"][0]["message"].get("content") or "" if status == 200 else ""
    check("OpenAI tool round trip: the result is used", status == 200 and "14" in answer and reply["choices"][0]["finish_reason"] == "stop",
          f"{dt:.1f} s, {answer[:100]!r}")

# 5. Anthropic Messages, thinking by default
a_plain = {"model": MODEL, "max_tokens": MAX, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}
status, reply, dt = call("/v1/messages", a_plain)
kinds = [b["type"] for b in reply.get("content", [])] if status == 200 else []
text = "".join(b.get("text", "") for b in reply.get("content", []) if b["type"] == "text") if status == 200 else ""
check("Anthropic messages: thinking block by default, an answer", "thinking" in kinds and "391" in text, f"{dt:.1f} s, blocks {kinds}")
status, (blocks, stop), dt = anthropic_stream(a_plain)
check("Anthropic messages stream: thinking, then text", status == 200 and [b[0] for b in blocks][:1] == ["thinking"]
      and any(b[0] == "text" and "391" in b[2] for b in blocks) and stop == "end_turn", f"{dt:.1f} s, {[b[0] for b in blocks]}, {stop}")

# 6. Anthropic tool round trip
a_ask = {"model": MODEL, "max_tokens": MAX, "tools": [A_WEATHER], "messages": [{"role": "user", "content": ASK_TOOL}]}
status, reply, dt = call("/v1/messages", a_ask)
uses = [b for b in reply.get("content", []) if b["type"] == "tool_use"] if status == 200 else []
check("Anthropic tool call: tool_use, stop_reason tool_use", bool(uses) and reply.get("stop_reason") == "tool_use"
      and uses[0]["name"] == "get_weather", f"{dt:.1f} s, {[(u['name'], u['input']) for u in uses]}")
check("Anthropic tool call: no raw markup left in text", all(MARK not in b.get("text", "") for b in reply.get("content", [])))
status, (blocks, stop), dt = anthropic_stream(a_ask)
check("Anthropic tool call stream: tool_use blocks, stop_reason tool_use", status == 200 and stop == "tool_use"
      and any(b[0] == "tool_use" and b[1] == "get_weather" for b in blocks), f"{dt:.1f} s, {[(b[0], b[1]) for b in blocks]}")
if uses:
    follow = {**a_ask, "messages": a_ask["messages"] + [
        {"role": "assistant", "content": reply["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"], "content": RESULT}]}]}
    status, reply, dt = call("/v1/messages", follow)
    text = "".join(b.get("text", "") for b in reply.get("content", []) if b["type"] == "text") if status == 200 else ""
    check("Anthropic tool round trip: the result is used", status == 200 and "14" in text and reply.get("stop_reason") == "end_turn",
          f"{dt:.1f} s, {text[:100]!r}")

# 7. count_tokens
status, reply, _ = call("/v1/messages/count_tokens", {"model": MODEL, "messages": a_plain["messages"]})
check("Anthropic count_tokens", status == 200 and reply.get("input_tokens", 0) > 0, f"{reply}")

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
sys.exit(1 if failures else 0)
