"""Inkling 2 (--family inkling2): its tool-call format and reasoning levels, and that no other recipe sees either.

The fixtures are the native inkling2 server's replies as captured on 2026-11-15 (out/prefill/p5-bin, template C):
several calls arrive concatenated in content with no separator, after the reply's text message.
"""
import io
import json
import random
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inkling_gateway as gw

MARK = gw.INKLING_CALL
TOOLS = ["get_weather", "get_time"]
# the captured non-streamed content (two calls, no text) and the streamed content pieces (text, then two calls)
WHOLE = ('get_weather' + MARK + '{"name":"get_weather","args":{"city":"Wellington"}}'
         'get_time' + MARK + '{"name":"get_time","args":{"city":"Auckland"}}')
PIECES = ["Checking", " now.get_weather", MARK + '{"name":"get', '_weather","args"', ':{"city":"', 'Wellington"}}get',
          "_time" + MARK + '{"name":', '"get_time","args', '":{"city":"', 'Auckland"}}']
OPENAI_TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}} for n in TOOLS]
ANTHROPIC_TOOLS = [{"name": n, "description": n, "input_schema": {
    "type": "object", "properties": {"city": {"type": "string"}}}} for n in TOOLS]
INKLING_ARGS = SimpleNamespace(family="inkling2", served_model="inkling", alias_model=["inkling-2-small"],
                               owned_by="tensorfold", root="kortexa-ai/Inkling-2-Small-EXL3-2.59bpw",
                               context=500000, max_tokens=65536)
OTHER_ARGS = SimpleNamespace(served_model="test-model", alias_model=[], owned_by="tensorfold", root="test-root",
                           context=524288, max_tokens=65536)


def split(text, names=TOOLS, pieces=None, single=False):
    """(text, calls) from feeding ``pieces`` (default: the whole text) and finishing."""
    parser = gw.InklingCalls(names, single)
    events = []
    for piece in pieces if pieces is not None else [text]:
        events += parser.feed(piece)
    events = gw._merge_text(events + parser.finish())
    return "".join(v for k, v in events if k == "text"), [{k: x for k, x in v.items() if k != "written"}
                                                          for kind, v in events if kind == "call"]


def calls(*pairs):
    return [{"name": n, "arguments": json.dumps(a, ensure_ascii=False)} for n, a in pairs]


class Parser(unittest.TestCase):
    def test_the_captured_two_calls(self):
        self.assertEqual(split(WHOLE), ("", calls(("get_weather", {"city": "Wellington"}),
                                                  ("get_time", {"city": "Auckland"}))))

    def test_text_before_the_calls_is_kept_and_the_header_dropped(self):
        text = "".join(PIECES)
        want = ("Checking now.", calls(("get_weather", {"city": "Wellington"}), ("get_time", {"city": "Auckland"})))
        self.assertEqual(split(text), want)
        self.assertEqual(split(text, pieces=PIECES), want)

    def test_every_split_gives_the_whole_texts_result(self):
        samples = ["".join(PIECES), WHOLE, "Let me checkget_weather" + MARK + '{"name":"get_weather","args":{}}',
                   "a" + MARK + "b", "get_weather" + MARK + '{"name":"get_weather","args":{"city":}',
                   'get_time' + MARK + '{"name":"get_time","args":{"q":"x}y{\\"z"}} and after',
                   "plain text with get_ but no call", "get_weather" + MARK + "not json at all",
                   "get_weather" + MARK + '  {"args":{"a":[1,{"b":2}]}}' + "Done."]
        rng = random.Random(7)
        for text in samples:
            whole = split(text)
            for at in range(len(text) + 1):
                with self.subTest(text=text, at=at):
                    self.assertEqual(split(text, pieces=[text[:at], text[at:]]), whole)
            for _ in range(30):
                cuts = sorted(rng.sample(range(len(text) + 1), min(4, len(text) + 1)))
                parts = [text[i:j] for i, j in zip([0] + cuts, cuts + [len(text)])]
                self.assertEqual(split(text, pieces=parts), whole)
            self.assertEqual(split(text, pieces=list(text)), whole)

    def test_malformed_or_unclosed_json_stays_text(self):
        for text in ("get_weather" + MARK + '{"name":"get_weather","args":{"city":}}',      # invalid JSON
                     "get_weather" + MARK + '{"name":"get_weather","args":{"city":"W"',    # unclosed at the end
                     "get_weather" + MARK + '{"name":"get_weather","args":["W"]}',         # args not an object
                     "get_weather" + MARK + "[1, 2]",                                      # not an object
                     "get_weather" + MARK + '{"name":"get_time","args":{}}',               # names another tool
                     "search" + MARK + '{"name":"search","args":{}}'):                     # undeclared tool
            with self.subTest(text=text):
                self.assertEqual(split(text), (text, []))
                self.assertEqual(split(text, pieces=list(text)), (text, []))

    def test_a_malformed_call_does_not_hide_a_good_one_after_it(self):
        bad = "get_weather" + MARK + '{"name":"get_weather","args":{"city":}}'
        good = "get_time" + MARK + '{"name":"get_time","args":{"city":"Auckland"}}'
        self.assertEqual(split(bad + good), (bad, calls(("get_time", {"city": "Auckland"}))))

    def test_marker_text_inside_arguments_is_an_argument(self):
        args = {"text": "literal " + MARK + " and get_time" + MARK + "{}", "n": 1}
        text = "get_weather" + MARK + json.dumps({"name": "get_weather", "args": args})
        self.assertEqual(split(text), ("", calls(("get_weather", args))))

    def test_case_spelling_json_string_arguments_and_missing_args(self):
        self.assertEqual(split("GET_WEATHER" + MARK + '{"name":"Get_Weather","args":"{\\"city\\": \\"W\\"}"}'),
                         ("", calls(("get_weather", {"city": "W"}))))
        self.assertEqual(split("get_time" + MARK + '{"name":"get_time"}'), ("", calls(("get_time", {}))))
        self.assertEqual(split("get_time" + MARK + '{"args":{"city":"A"}}'), ("", calls(("get_time", {"city": "A"}))))

    def test_longest_declared_name_and_the_jsons_name_pick_the_header(self):
        names = ["weather", "get_weather"]
        self.assertEqual(split("see get_weather" + MARK + '{"name":"get_weather","args":{}}', names),
                         ("see ", calls(("get_weather", {}))))
        self.assertEqual(split("use get_weather" + MARK + '{"name":"weather","args":{}}', names),
                         ("use get_", calls(("weather", {}))))

    def test_single_keeps_the_first_call(self):
        self.assertEqual(split(WHOLE, single=True), ("", calls(("get_weather", {"city": "Wellington"}))))

    def test_no_declared_tools_parses_nothing(self):
        self.assertEqual(split(WHOLE, names=[]), (WHOLE, []))
        resp = {"choices": [{"message": {"role": "assistant", "content": WHOLE}, "finish_reason": "stop"}]}
        self.assertEqual(gw.inkling_completion(resp, []), (resp, False))

    def test_held_text_is_at_most_a_declared_header(self):
        parser = gw.InklingCalls(TOOLS)
        self.assertEqual(parser.feed("Checking now."), [("text", "Checking now.")])
        self.assertEqual(parser.feed(" Then get_ti"), [("text", " Then ")])
        self.assertEqual(parser.feed("me is up"), [("text", "get_time is up")])
        self.assertEqual(parser.finish(), [])


class Completion(unittest.TestCase):
    def test_non_streamed_completion_gets_tool_calls(self):
        resp = {"id": "c", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Sure. " + WHOLE,
                                                                 "reasoning_content": "Need both."},
                                        "finish_reason": "stop"}], "usage": {"prompt_tokens": 3}}
        out, changed = gw.inkling_completion(resp, TOOLS)
        self.assertTrue(changed)
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "Sure. ")
        self.assertEqual(choice["message"]["reasoning_content"], "Need both.")
        self.assertEqual([(c["type"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                          for c in choice["message"]["tool_calls"]],
                         [("function", "get_weather", {"city": "Wellington"}), ("function", "get_time", {"city": "Auckland"})])
        self.assertTrue(all(c["id"].startswith("call_") for c in choice["message"]["tool_calls"]))
        self.assertEqual(resp["choices"][0]["message"]["content"], "Sure. " + WHOLE)     # the input is not mutated

    def test_length_stays_length_and_only_call_text_becomes_null(self):
        resp = {"choices": [{"message": {"content": WHOLE + "\n"}, "finish_reason": "length"}]}
        out, _ = gw.inkling_completion(resp, TOOLS)
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        self.assertIsNone(out["choices"][0]["message"]["content"])

    def test_bodies_without_calls_keep_their_bytes(self):
        payload = b'{"choices": [{"message": {"content": "caf\\u00e9 plain"}, "finish_reason": "stop"}]}'
        self.assertIs(gw.inkling_rewrite_json(payload, TOOLS), payload)
        self.assertEqual(gw.inkling_rewrite_json(b"not json", TOOLS), b"not json")


class Efforts(unittest.TestCase):
    def test_levels(self):
        cases = {None: "high", "none": "minimal", "minimal": "minimal", "low": "low", "medium": "medium",
                 "high": "high", "xhigh": "xhigh", "max": "max", " HIGH ": "high", 0: "minimal", 0.0: "minimal",
                 0.1: "minimal", 0.15: "low", 0.2: "low", 0.5: "medium", 0.7: "medium", 0.8: "high", 0.9: "high",
                 0.95: "xhigh", 0.99: "xhigh", 1: "xhigh", "0.7": "medium"}
        for given, want in cases.items():
            with self.subTest(given=given):
                self.assertEqual(gw.inkling_effort(given), want)
        for bad in ("off", "huge", -0.1, 1.5, True, [], {}):
            with self.subTest(bad=bad), self.assertRaises(gw.RequestError):
                gw.inkling_effort(bad)

    def test_budget_levels(self):
        for budget, want in ((1024, "low"), (4095, "low"), (4096, "medium"), (16383, "medium"), (16384, "high"),
                             (31999, "high"), (32768, "max"), (128000, "max")):
            with self.subTest(budget=budget):
                self.assertEqual(gw.inkling_budget_effort(budget), want)

    def test_anthropic_thinking(self):
        cases = [({}, "high"), ({"thinking": {"type": "disabled"}}, "minimal"),
                 ({"thinking": {"type": "enabled", "budget_tokens": 2048}}, "low"),
                 ({"thinking": {"type": "enabled", "budget_tokens": 10000}}, "medium"),
                 ({"thinking": {"type": "enabled", "budget_tokens": 31999}}, "high"),
                 ({"thinking": {"type": "enabled", "budget_tokens": 64000}}, "max"),
                 ({"thinking": {"type": "adaptive"}}, "high"),
                 ({"output_config": {"effort": "low"}}, "low"),
                 ({"output_config": {"effort": "max"}, "thinking": {"type": "enabled", "budget_tokens": 2048}}, "max"),
                 ({"output_config": {"effort": "high"}, "thinking": {"type": "disabled"}}, "minimal"),
                 ({"output_config": {"effort": "none"}}, "minimal")]
        for body, want in cases:
            with self.subTest(body=body):
                self.assertEqual(gw.inkling_messages_effort(body), want)
        with self.assertRaises(gw.RequestError):
            gw.inkling_messages_effort({"thinking": {"type": "enabled", "budget_tokens": 512}})

    def test_chat_bodies(self):
        def body(**fields):
            return gw.inkling_chat_body({"model": "inkling", "messages": [], **fields})
        self.assertEqual(body()["reasoning_effort"], "high")
        self.assertEqual(body(reasoning_effort="none")["reasoning_effort"], "minimal")
        self.assertEqual(body(reasoning_effort="low")["reasoning_effort"], "low")
        self.assertEqual(body(reasoning={"effort": "medium"})["reasoning_effort"], "medium")
        off = body(chat_template_kwargs={"enable_thinking": False, "other": 1})
        self.assertEqual((off["reasoning_effort"], off["chat_template_kwargs"]),
                         ("minimal", {"enable_thinking": True, "other": 1}))
        switch = body(chat_template_kwargs={"thinking": False})
        self.assertEqual((switch["reasoning_effort"], switch["chat_template_kwargs"]), ("minimal", {}))
        nested = body(chat_template_kwargs={"reasoning_effort": "xhigh"})
        self.assertEqual((nested["reasoning_effort"], nested["chat_template_kwargs"]), ("xhigh", {}))
        disabled = body(thinking={"type": "disabled"}, reasoning_effort="max")
        self.assertEqual(disabled["reasoning_effort"], "minimal")
        self.assertNotIn("thinking", disabled)
        enabled = body(thinking={"type": "enabled", "budget_tokens": 8000})
        self.assertEqual(enabled["reasoning_effort"], "medium")
        self.assertNotIn("thinking_budget", enabled)
        for bad in ({"reasoning_effort": "off"}, {"thinking": {"type": "sometimes"}}, {"chat_template_kwargs": []}):
            with self.subTest(bad=bad), self.assertRaises(gw.RequestError):
                body(**bad)


class Reply(io.BytesIO):
    def __init__(self, body, *, status=200, content_type="application/json"):
        super().__init__(json.dumps(body).encode() if isinstance(body, dict) else body)
        self.status = self.code = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def read1(self, size=-1):
        return self.read(size)


def sse(chunks):
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


def stream_chunks(pieces=PIECES, finish="stop"):
    env = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "inkling"}
    chunks = [{**env, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}]
    chunks += [{**env, "choices": [{"index": 0, "delta": {"reasoning_content": r}, "finish_reason": None}]}
               for r in ("Say it, ", "then call both.")]
    chunks += [{**env, "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]} for p in pieces]
    chunks.append({**env, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "tensorfold": {"x": 1}})
    chunks.append({**env, "choices": [], "usage": {"prompt_tokens": 203, "completion_tokens": 148,
                                                   "prompt_tokens_details": {"cached_tokens": 200}}})
    return chunks


class Http(unittest.TestCase):
    """The gateway over HTTP with a scripted upstream."""
    args = INKLING_ARGS

    def setUp(self):
        self.requests, self.reply = [], None
        for target in (patch.object(gw, "ARGS", self.args),
                       patch.multiple(gw, PIN_TEMPERATURE=None, STRICT_REASONING=False, STABLE_TIME=False,
                                      RESTORE_IMAGES=False, RESPONSES_REASONING_CONTEXT_DEFAULT="all_turns"),
                       patch.object(gw.Handler, "_log")):
            target.start()
            self.addCleanup(target.stop)
        replies = patch.object(gw, "upstream", side_effect=self.upstream_reply)
        replies.start()
        self.addCleanup(replies.stop)
        if self.args is not INKLING_ARGS:     # the reply pre-fill notifications (Inkling sends none: tested there)
            notes = patch.object(gw, "incoming")
            notes.start()
            self.addCleanup(notes.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), gw.Handler)
        self.server.daemon_threads = False
        self.addCleanup(self.server.server_close)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), thread.join(timeout=5)))

    def upstream_reply(self, path, payload=None, **kwargs):
        body = json.loads(kwargs["data"]) if kwargs.get("data") is not None else payload
        self.requests.append((path, body, kwargs.get("data")))
        if path == "/tokenize":
            return Reply({"count": 7})
        if path == "/v1/models":
            return Reply({"object": "list", "data": [{"id": "inkling"}]})
        if self.reply is not None:
            return self.reply(body)
        return Reply({"id": "c", "object": "chat.completion", "model": "inkling", "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "READY", "reasoning_content": "ok"}, "finish_reason": "stop"}]})

    def post(self, path, body, raw=False):
        request = urllib.request.Request(f"http://127.0.0.1:{self.server.server_port}{path}",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                data = response.read()
                return response.status, (data if raw else json.loads(data))
        except urllib.error.HTTPError as response:
            return response.code, json.load(response)

    def last(self, path, raw=False):
        return next((data if raw else body) for p, body, data in reversed(self.requests) if p == path)


def events(data):
    """An SSE body as (event, data) pairs."""
    out = []
    for block in data.decode().split("\n\n"):
        name = payload = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
        if payload is not None:
            out.append((name, payload if payload == "[DONE]" else json.loads(payload)))
    return out


class InklingHttp(Http):
    def test_models_card_is_text_only(self):
        request = urllib.request.Request(f"http://127.0.0.1:{self.server.server_port}/v1/models")
        with urllib.request.urlopen(request, timeout=5) as response:
            card = json.load(response)
        self.assertEqual([m["id"] for m in card["data"]], ["inkling", "inkling-2-small"])
        self.assertTrue(all(m["input_modalities"] == ["text"] and m["supports_vision"] is False for m in card["data"]))
        self.assertEqual(card["data"][0]["max_model_len"], 500000)

    def test_openai_chat_non_stream(self):
        self.reply = lambda body: Reply({"id": "c", "object": "chat.completion", "model": "inkling", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": WHOLE, "reasoning_content": "Both."},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 201, "completion_tokens": 83}})
        status, reply = self.post("/v1/chat/completions", {"model": "inkling", "tools": OPENAI_TOOLS,
                                                           "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual(status, 200)
        choice = reply["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["message"]["reasoning_content"], "Both.")
        self.assertEqual([c["function"]["name"] for c in choice["message"]["tool_calls"]], TOOLS)
        sent = self.last("/v1/chat/completions")
        self.assertEqual(sent["reasoning_effort"], "high")
        self.assertNotIn("thinking_budget", sent)

    def test_openai_chat_without_tools_is_relayed_unchanged(self):
        status, reply = self.post("/v1/chat/completions", {"model": "inkling", "reasoning_effort": "none",
                                                           "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((status, reply["choices"][0]["message"]["content"]), (200, "READY"))
        self.assertEqual(self.last("/v1/chat/completions")["reasoning_effort"], "minimal")

    def test_openai_chat_stream(self):
        self.reply = lambda body: Reply(sse(stream_chunks()), content_type="text/event-stream")
        status, data = self.post("/v1/chat/completions", {"model": "inkling", "stream": True, "tools": OPENAI_TOOLS,
                                                          "messages": [{"role": "user", "content": "go"}]}, raw=True)
        self.assertEqual(status, 200)
        chunks = [c for _, c in events(data)]
        self.assertEqual(chunks[-1], "[DONE]")
        deltas = [c["choices"][0] for c in chunks[:-1] if c["choices"]]
        text = "".join(d["delta"].get("content") or "" for d in deltas)
        reasoning = "".join(d["delta"].get("reasoning_content") or "" for d in deltas)
        tool_calls = [t for d in deltas for t in d["delta"].get("tool_calls") or []]
        self.assertEqual((text, reasoning), ("Checking now.", "Say it, then call both."))
        self.assertEqual([(t["index"], t["function"]["name"], json.loads(t["function"]["arguments"])) for t in tool_calls],
                         [(0, "get_weather", {"city": "Wellington"}), (1, "get_time", {"city": "Auckland"})])
        self.assertEqual([d["finish_reason"] for d in deltas if d["finish_reason"]], ["tool_calls"])
        self.assertEqual(chunks[-2]["usage"]["prompt_tokens"], 203)          # the usage chunk passes unchanged
        self.assertTrue(any("tensorfold" in c for c in chunks[:-1]))         # so do the terminal chunk's extras

    def test_unprefixed_chat_route_is_parsed_like_v1(self):
        # a client whose base URL has no /v1 (the 14:39 downstream failure: the tool call reached it as raw text)
        self.reply = lambda body: Reply({"id": "c", "object": "chat.completion", "model": "inkling", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": WHOLE, "reasoning_content": "Both."},
            "finish_reason": "stop"}]})
        status, reply = self.post("/chat/completions", {"model": "inkling", "tools": OPENAI_TOOLS,
                                                        "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual(status, 200)
        self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual([c["function"]["name"] for c in reply["choices"][0]["message"]["tool_calls"]], TOOLS)
        self.assertEqual(self.last("/v1/chat/completions")["reasoning_effort"], "high")   # forwarded on the /v1 route

    def test_unprefixed_chat_stream_is_parsed_like_v1(self):
        self.reply = lambda body: Reply(sse(stream_chunks()), content_type="text/event-stream")
        status, data = self.post("/chat/completions", {"model": "inkling", "stream": True, "tools": OPENAI_TOOLS,
                                                       "messages": [{"role": "user", "content": "go"}]}, raw=True)
        self.assertEqual(status, 200)
        deltas = [c["choices"][0] for _, c in events(data) if c != "[DONE]" and c["choices"]]
        tool_calls = [t for d in deltas for t in d["delta"].get("tool_calls") or []]
        self.assertEqual([t["function"]["name"] for t in tool_calls], ["get_weather", "get_time"])
        self.assertNotIn("<|content_invoke_tool_json|>", "".join(d["delta"].get("content") or "" for d in deltas))

    def test_unprefixed_messages_route_is_the_messages_api(self):
        status, reply = self.post("/messages", {"model": "inkling", "max_tokens": 64,
                                                "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(reply["type"], "message")

    def test_openai_stream_without_calls_keeps_every_byte(self):
        body = sse(stream_chunks(pieces=["plain ", "answer"]))
        self.reply = lambda _: Reply(body, content_type="text/event-stream")
        status, data = self.post("/v1/chat/completions", {"model": "inkling", "stream": True,
                                                          "messages": [{"role": "user", "content": "hi"}]}, raw=True)
        self.assertEqual((status, data), (200, body))

    def test_anthropic_non_stream_round_trip(self):
        self.reply = lambda body: Reply({"id": "c", "object": "chat.completion", "model": "inkling", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": "Checking now." + WHOLE,
                                    "reasoning_content": "Need both."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        status, reply = self.post("/v1/messages", {"model": "inkling", "max_tokens": 512, "tools": ANTHROPIC_TOOLS,
                                                   "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual(status, 200)
        self.assertEqual(reply["stop_reason"], "tool_use")
        kinds = [(b["type"], b.get("thinking") or b.get("text") or b.get("name")) for b in reply["content"]]
        self.assertEqual(kinds, [("thinking", "Need both."), ("text", "Checking now."), ("tool_use", "get_weather"),
                                 ("tool_use", "get_time")])
        self.assertEqual(reply["content"][2]["input"], {"city": "Wellington"})
        self.assertTrue(all(b["id"].startswith("toolu_") for b in reply["content"][2:]))
        sent = self.last("/v1/chat/completions")
        self.assertEqual(sent["reasoning_effort"], "high")                    # Anthropic sent no thinking: high
        self.assertNotIn("thinking_budget", sent)
        self.assertEqual(sent["model"], "inkling")

    def test_anthropic_stream_round_trip(self):
        self.reply = lambda body: Reply(sse(stream_chunks()), content_type="text/event-stream")
        status, data = self.post("/v1/messages", {"model": "inkling-2-small", "max_tokens": 512, "stream": True,
                                                  "tools": ANTHROPIC_TOOLS, "thinking": {"type": "disabled"},
                                                  "messages": [{"role": "user", "content": "go"}]}, raw=True)
        self.assertEqual(status, 200)
        stream = events(data)
        self.assertEqual(stream[0][1]["message"]["model"], "inkling-2-small")  # the alias answers as asked
        blocks, deltas = {}, {}
        for name, payload in stream:
            if name == "content_block_start":
                blocks[payload["index"]] = payload["content_block"]
            elif name == "content_block_delta":
                d = payload["delta"]
                deltas[payload["index"]] = deltas.get(payload["index"], "") + (
                    d.get("thinking") or d.get("text") or d.get("partial_json") or "")
        self.assertEqual([blocks[i]["type"] for i in sorted(blocks)], ["thinking", "text", "tool_use", "tool_use"])
        self.assertEqual((deltas[0], deltas[1]), ("Say it, then call both.", "Checking now."))
        self.assertEqual([(blocks[i]["name"], json.loads(deltas[i])) for i in (2, 3)],
                         [("get_weather", {"city": "Wellington"}), ("get_time", {"city": "Auckland"})])
        stop = next(p for n, p in stream if n == "message_delta")
        self.assertEqual(stop["delta"]["stop_reason"], "tool_use")
        self.assertEqual(stop["usage"], {"output_tokens": 148, "input_tokens": 3, "cache_read_input_tokens": 200})
        sent = self.last("/v1/chat/completions")
        self.assertEqual((sent["reasoning_effort"], sent["model"]), ("minimal", "inkling"))   # disabled -> minimal

    def test_anthropic_stream_without_terminal_chunk_still_flushes(self):
        chunks = [c for c in stream_chunks(pieces=["Done, get_"]) if not (c["choices"] and c["choices"][0]["finish_reason"])]
        self.reply = lambda body: Reply(sse(chunks), content_type="text/event-stream")
        status, data = self.post("/v1/messages", {"model": "inkling", "max_tokens": 64, "stream": True,
                                                  "tools": ANTHROPIC_TOOLS,
                                                  "messages": [{"role": "user", "content": "go"}]}, raw=True)
        text = "".join(p["delta"].get("text", "") for n, p in events(data) if n == "content_block_delta")
        self.assertEqual((status, text), (200, "Done, get_"))

    def test_anthropic_history_reaches_the_template_fields(self):
        body = {"model": "inkling", "max_tokens": 256, "tools": ANTHROPIC_TOOLS, "messages": [
            {"role": "user", "content": "weather and time?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Need both.", "signature": "x"},
                {"type": "text", "text": "Checking now."},
                {"type": "tool_use", "id": "toolu_a", "name": "get_weather", "input": {"city": "Wellington"}},
                {"type": "tool_use", "id": "toolu_b", "name": "get_time", "input": {"city": "Auckland"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_a", "content": "14C, wind"},
                {"type": "tool_result", "tool_use_id": "toolu_b", "content": [
                    {"type": "text", "text": "09:30"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}]}]}]}
        status, _ = self.post("/v1/messages", body)
        self.assertEqual(status, 200)
        messages = self.last("/v1/chat/completions")["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "tool"])
        assistant = messages[1]
        self.assertEqual((assistant["reasoning_content"], assistant["content"]), ("Need both.", "Checking now."))
        self.assertEqual([(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"]))
                          for c in assistant["tool_calls"]],
                         [("toolu_a", "get_weather", {"city": "Wellington"}), ("toolu_b", "get_time", {"city": "Auckland"})])
        self.assertEqual([(m["tool_call_id"], m["content"]) for m in messages[2:]],
                         [("toolu_a", "14C, wind"), ("toolu_b", "09:30\n" + gw.INKLING_IMAGE_NOTE)])

    def test_count_tokens_renders_the_generation_level(self):
        status, reply = self.post("/v1/messages/count_tokens", {"model": "inkling", "messages": [
            {"role": "user", "content": "hi"}], "thinking": {"type": "enabled", "budget_tokens": 20000}})
        self.assertEqual((status, reply["input_tokens"]), (200, 7))
        self.assertEqual(self.last("/tokenize")["reasoning_effort"], "high")

    def test_responses_levels(self):
        for given, want in (({}, "high"), ({"reasoning": {"effort": "none"}}, "minimal"),
                            ({"reasoning_effort": "low"}, "low"), ({"thinking": {"type": "disabled"}}, "minimal"),
                            ({"thinking": {"type": "enabled", "budget_tokens": 5000}}, "medium"),
                            ({"chat_template_kwargs": {"enable_thinking": False}}, "minimal")):
            with self.subTest(given=given):
                status, reply = self.post("/v1/responses", {"model": "inkling", "input": "hi", **given})
                sent = self.last("/v1/chat/completions")       # Inkling's Responses run on the chat route
                self.assertEqual((status, sent["reasoning_effort"], reply["reasoning"]["effort"]), (200, want, want))
                self.assertNotIn("thinking_budget", sent)
                self.assertNotIn("thinking", sent)
                self.assertTrue(sent.get("chat_template_kwargs", {}).get("enable_thinking", True))

    def test_no_reply_prefill_notifications(self):
        with patch.object(gw, "threading") as threads:
            gw.incoming("key", True)
        threads.Thread.assert_not_called()

    def test_family_flag_accepts_only_inkling2(self):
        with patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
            gw.main(["--family", "other"])


class OtherRecipesHttp(Http):
    """Without --family the gateway is what it was: no Inkling parsing, no level mapping, every reply as relayed."""
    args = OTHER_ARGS

    def test_inkling_markup_is_not_parsed(self):
        resp = {"id": "c", "object": "chat.completion", "model": "test-model", "choices": [{"index": 0, "message": {
            "role": "assistant", "content": WHOLE}, "finish_reason": "stop"}]}
        raw = json.dumps(resp).encode()
        self.reply = lambda body: Reply(raw)
        status, data = self.post("/v1/chat/completions", {"model": "test-model", "tools": OPENAI_TOOLS,
                                                          "messages": [{"role": "user", "content": "go"}]}, raw=True)
        self.assertEqual((status, data), (200, raw))
        status, reply = self.post("/v1/messages", {"model": "test-model", "max_tokens": 8, "tools": ANTHROPIC_TOOLS,
                                                   "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual([b["type"] for b in reply["content"]], ["text"])
        self.assertEqual(reply["stop_reason"], "end_turn")

    def test_requests_are_forwarded_as_before(self):
        body = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "tools": OPENAI_TOOLS}
        self.post("/v1/chat/completions", body)
        self.assertEqual(self.last("/v1/chat/completions", raw=True), json.dumps(body).encode())   # no effort added
        self.post("/v1/messages", {"model": "test-model", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(self.last("/v1/chat/completions")["reasoning_effort"], "none")
        self.post("/v1/messages", {"model": "test-model", "max_tokens": 8, "thinking": {
            "type": "enabled", "budget_tokens": 2048}, "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(self.last("/v1/chat/completions")["thinking_budget"], 2048)
        request = urllib.request.Request(f"http://127.0.0.1:{self.server.server_port}/v1/models")
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(json.load(response)["data"][0]["input_modalities"], ["text", "image"])

    def test_images_stay_images(self):
        self.post("/v1/messages", {"model": "test-model", "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}]}]})
        content = self.last("/v1/chat/completions")["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")


TEMPLATE = Path(__file__).resolve().parents[2] / "config" / "chat_template.engine-C.jinja"
try:
    import jinja2
except ImportError:
    jinja2 = None


@unittest.skipUnless(jinja2 is not None and TEMPLATE.is_file(), "needs jinja2 and the Inkling deployment's template C")
class Template(unittest.TestCase):
    """Converted Anthropic history rendered by template C, after the engine's own normalization (messages.zig: content
    null -> "", call arguments parsed into objects): the reply format the parser reads is the format history renders."""

    def render(self, messages, tools, effort):
        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])

        def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def fail(message):
            raise ValueError(message)

        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = fail
        return env.from_string(TEMPLATE.read_text()).render(messages=messages, tools=tools, reasoning_effort=effort)

    def test_history_renders_inklings_own_call_format(self):
        with patch.object(gw, "ARGS", INKLING_ARGS), patch.object(gw, "STABLE_TIME", False):
            req, _ = gw.to_openai({"model": "inkling", "max_tokens": 64, "tools": ANTHROPIC_TOOLS, "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Need both.", "signature": "x"},
                    {"type": "tool_use", "id": "toolu_a", "name": "get_weather", "input": {"city": "Wellington"}},
                    {"type": "tool_use", "id": "toolu_b", "name": "get_time", "input": {"city": "Auckland"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_a", "content": "14C"},
                                             {"type": "tool_result", "tool_use_id": "toolu_b", "content": "09:30"}]}]})
        messages = []
        for m in req["messages"]:                 # the engine's normalization
            m = {**m, "content": m.get("content") if m.get("content") is not None else ""}
            if m.get("tool_calls"):
                m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                                   for c in m["tool_calls"]]
            messages.append(m)
        prompt = self.render(messages, req["tools"], req["reasoning_effort"])
        self.assertIn("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>", prompt)
        self.assertIn("<|message_model|><|content_thinking|>Need both.<|end_message|>"
                      "<|message_model|>get_weather" + MARK + '{"name":"get_weather","args":{"city":"Wellington"}}'
                      "<|end_message|><|message_model|>get_time" + MARK + '{"name":"get_time","args":{"city":"Auckland"}}'
                      "<|end_message|><|content_model_end_sampling|>", prompt)
        self.assertIn("<|message_tool|>get_weather<|content_text|>14C<|end_message|>"
                      "<|message_tool|>get_time<|content_text|>09:30<|end_message|>", prompt)
        # the reply the server would return for that call message parses back to the same call
        reply = "get_weather" + MARK + '{"name":"get_weather","args":{"city":"Wellington"}}'
        self.assertEqual(split(reply), ("", calls(("get_weather", {"city": "Wellington"}))))
        self.assertIn('"name":"get_time"', prompt.split("tool_declare<|content_xml|>")[1])


if __name__ == "__main__":
    unittest.main()
