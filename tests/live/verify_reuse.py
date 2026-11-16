#!/usr/bin/env python3
"""verify_reuse.py [BASE]: the served model live through the gateway (default http://127.0.0.1:8080), after
`scripts/serve.sh start`. Standard library only; run it on the head node (it waits for the engine to be idle on :8888 first).

  1. Responses with a bash tool: a function_call item, the streamed function_call events, and a round trip
     (previous_response_id + function_call_output) whose answer uses the result
  2. on each API (OpenAI chat, Anthropic Messages, Responses) a 3-turn tool conversation whose client DROPS its
     reasoning (no reasoning_content / thinking blocks / reasoning items): the gateway's reasoning memory must put it
     back, so turn N+1's cached prompt is about turn N's prompt + reply (the engine's prompt reuse)
Prints each turn's prompt / reply / cached tokens and one PASS / FAIL line per check; exits 1 if any failed. Another
client's request between two turns evicts the engine's live state: a conversation that misses is run once more."""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MARK = "<|content_invoke_tool_json|>"
BASH_OAI = {"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its output.",
                                             "parameters": {"type": "object", "properties": {
                                                 "command": {"type": "string"}, "description": {"type": "string"}},
                                                 "required": ["command"]}}}
BASH_RESP = {"type": "function", "name": "bash", "description": "Run a shell command and return its output.",
             "parameters": BASH_OAI["function"]["parameters"]}
BASH_ANT = {"name": "Bash", "description": "Run a shell command and return its output.",
            "input_schema": BASH_OAI["function"]["parameters"]}
ASK = ("First run `uname -s` with the bash tool. After you see its output, run `whoami` with the bash tool. "
       "Then tell me both results in one short sentence.")
OUTPUTS = {"uname -s": "Linux"}
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def idle():
    for _ in range(120):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8888/v1/load", timeout=5) as r:
                d = json.load(r)
            if d["running"] == 0 and d["waiting"] == 0:
                return
        except Exception:
            pass
        time.sleep(1)


def post(path, body, stream=False):
    """POST through the gateway; retried while the engine serves someone else's stream."""
    for attempt in range(6):
        idle()
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                raw = r.read().decode()
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            if "one stream at a time" in raw and attempt < 5:
                time.sleep(3)
                continue
            raise RuntimeError(f"HTTP {e.code}: {raw[:300]}")
        if "one stream at a time" in raw and attempt < 5:
            time.sleep(3)
            continue
        return raw if stream else json.loads(raw)
    raise RuntimeError("the engine stayed busy")


def sse(text):
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


def run_tool(command):
    return OUTPUTS.get(command.strip().strip("`"), "ok")


# ---------------------------------------------------------------------------------- Responses, with a bash tool
print("== Responses API (bash tool)")
one = {"model": "inkling-2-small", "input": "List the root directory with the bash tool (`ls /`), then tell me in one "
                                    "sentence which entries it has.",
       "tools": [BASH_RESP], "max_output_tokens": 1024}
r1 = post("/responses", one)
calls = [i for i in r1["output"] if i["type"] == "function_call"]
check("Responses whole: a function_call item for bash", bool(calls) and calls[0]["name"] == "bash",
      f"{[(c['name'], c['arguments']) for c in calls]}")
check("Responses whole: no raw markup, reasoning item present", MARK not in json.dumps(r1)
      and any(i["type"] == "reasoning" for i in r1["output"]), f"items {[i['type'] for i in r1['output']]}")
text = post("/v1/responses", {**one, "stream": True}, stream=True)
ev = sse(text)
names = [n for n, _ in ev]
done = [p["item"] for n, p in ev if n == "response.output_item.done" and p["item"]["type"] == "function_call"]
check("Responses stream: function_call added / arguments delta / done / completed",
      "response.output_item.added" in names and "response.function_call_arguments.delta" in names
      and "response.function_call_arguments.done" in names and names[-1] == "response.completed" and bool(done)
      and MARK not in text, f"{len(ev)} events, calls {[(d['name'], d['arguments']) for d in done]}")
if calls:
    r2 = post("/v1/responses", {"model": "inkling-2-small", "previous_response_id": r1["id"], "tools": [BASH_RESP],
                                "max_output_tokens": 1024, "input": [
                                    {"type": "function_call_output", "call_id": calls[0]["call_id"],
                                     "output": "bin\nboot\ndev\netc\nhome\nusr\nvar"}]})
    answer = "".join(p.get("text", "") for i in r2["output"] if i["type"] == "message" for p in i["content"])
    u1, u2 = r1["usage"], r2["usage"]
    check("Responses round trip (previous_response_id): the result is used", "etc" in answer or "home" in answer
          or "usr" in answer, f"{answer[:120]!r}")
    check("Responses round trip: prompt reuse", u2["input_tokens_details"]["cached_tokens"] >= u1["input_tokens"] + u1["output_tokens"] - 3,
          f"turn 1 prompt {u1['input_tokens']} + reply {u1['output_tokens']} -> turn 2 cached {u2['input_tokens_details']['cached_tokens']} of {u2['input_tokens']}")


# ------------------------------------------------------------- prompt reuse with the reasoning dropped by the client
def report(api, turns):
    print(f"   {api}: " + "; ".join(f"turn {i + 1}: prompt {p} reply {c} cached {k}" for i, (p, c, k) in enumerate(turns)))
    for i in range(1, len(turns)):
        p0, c0, _ = turns[i - 1]
        check(f"{api} turn {i + 1}: cached ~ turn {i}'s prompt + reply (reasoning restored by the gateway)",
              turns[i][2] >= p0 + c0 - 3, f"cached {turns[i][2]} vs {p0 + c0}")


def chat_conversation():
    messages, turns = [{"role": "user", "content": ASK}], []
    for _ in range(3):
        reply = post("/chat/completions", {"model": "inkling-2-small", "messages": messages, "tools": [BASH_OAI], "max_tokens": 1024})
        u = reply["usage"]
        turns.append((u["prompt_tokens"], u["completion_tokens"], (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0))
        msg = reply["choices"][0]["message"]
        sent_back = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}   # reasoning_content dropped
        messages.append(sent_back)
        if not msg.get("tool_calls"):
            break
        for c in msg["tool_calls"]:
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": run_tool(json.loads(c["function"]["arguments"]).get("command", ""))})
    return turns, messages


def anthropic_conversation():
    messages, turns = [{"role": "user", "content": ASK}], []
    for _ in range(3):
        reply = post("/v1/messages", {"model": "inkling-2-small", "max_tokens": 1024, "tools": [BASH_ANT], "messages": messages})
        u = reply["usage"]
        turns.append((u["input_tokens"] + u.get("cache_read_input_tokens", 0), u["output_tokens"], u.get("cache_read_input_tokens", 0)))
        content = [b for b in reply["content"] if b["type"] != "thinking"]           # thinking blocks dropped
        messages.append({"role": "assistant", "content": content})
        uses = [b for b in content if b["type"] == "tool_use"]
        if not uses:
            break
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": b["id"],
                                                      "content": run_tool(b["input"].get("command", ""))} for b in uses]})
    return turns, messages


def responses_conversation():
    items, turns = [{"role": "user", "content": ASK}], []
    for _ in range(3):
        reply = post("/v1/responses", {"model": "inkling-2-small", "input": items, "tools": [BASH_RESP], "store": False,
                                       "max_output_tokens": 1024})
        u = reply["usage"]
        turns.append((u["input_tokens"], u["output_tokens"], u["input_tokens_details"]["cached_tokens"]))
        kept = [i for i in reply["output"] if i["type"] != "reasoning"]                # reasoning items dropped
        items += [{k: v for k, v in i.items() if k != "status"} for i in kept]
        calls = [i for i in kept if i["type"] == "function_call"]
        if not calls:
            break
        items += [{"type": "function_call_output", "call_id": c["call_id"],
                   "output": run_tool(json.loads(c["arguments"]).get("command", ""))} for c in calls]
    return turns, items


print("== prompt reuse, 3 turns each, the client dropping its reasoning")
for api, conversation in (("OpenAI chat", chat_conversation), ("Anthropic Messages", anthropic_conversation),
                          ("Responses", responses_conversation)):
    for attempt in range(2):                    # someone else's request between turns evicts the engine's live state
        turns, history = conversation()
        if all(turns[i][2] >= turns[i - 1][0] + turns[i - 1][1] - 3 for i in range(1, len(turns))) or attempt == 1:
            break
        print(f"   {api}: a turn missed the live state ({turns}); once more")
    report(api, turns)
    tools_used = [c for m in history for c in (m.get("tool_calls") or []) if isinstance(m, dict)] if api == "OpenAI chat" else []
    check(f"{api}: {len(turns)} turns, tool calls made", len(turns) >= 2)

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
sys.exit(1 if failures else 0)
