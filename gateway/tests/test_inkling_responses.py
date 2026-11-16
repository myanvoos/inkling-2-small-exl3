"""Inkling 2 (--family inkling2): the Responses API on the chat route, the reasoning memory on every API, and
Claude-Code-style tool sets on the Messages API. The upstream is scripted with the native server's reply shapes."""
import json
import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inkling import (ANTHROPIC_TOOLS, MARK, OPENAI_TOOLS, PIECES, TOOLS, WHOLE, Http, Reply, events, gw, sse,
                          stream_chunks)

RESPONSES_TOOLS = [{"type": "function", "name": n, "description": n,
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}} for n in TOOLS]
CHAT_REPLY = {"id": "c", "object": "chat.completion", "model": "inkling", "choices": [{"index": 0, "message": {
    "role": "assistant", "content": "Checking now." + WHOLE, "reasoning_content": "Need both."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 201, "completion_tokens": 83, "prompt_tokens_details": {"cached_tokens": 150},
              "completion_tokens_details": {"reasoning_tokens": 18}}, "tensorfold": {"reasoning_effort": "high"}}
ANSWER = {"id": "c2", "object": "chat.completion", "model": "inkling", "choices": [{"index": 0, "message": {
    "role": "assistant", "content": "It is 14C in Wellington and 09:30 in Auckland.", "reasoning_content": "Both came back."},
    "finish_reason": "stop"}], "usage": {"prompt_tokens": 330, "completion_tokens": 20}}
# Claude Code's tool set, abridged: capitalized names, optional fields
CLAUDE_TOOLS = [{"name": n, "description": f"The {n} tool.", "input_schema": {"type": "object", "properties": props,
                                                                               "required": req}}
                for n, props, req in (
                    ("Bash", {"command": {"type": "string"}, "description": {"type": "string"}, "timeout": {"type": "number"}}, ["command"]),
                    ("Read", {"file_path": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "number"}}, ["file_path"]),
                    ("Edit", {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, ["file_path"]),
                    ("Write", {"file_path": {"type": "string"}, "content": {"type": "string"}}, ["file_path", "content"]),
                    ("Glob", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
                    ("Grep", {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}}, ["pattern"]),
                    ("LS", {"path": {"type": "string"}}, ["path"]),
                    ("TodoWrite", {"todos": {"type": "array"}}, ["todos"]),
                    ("WebFetch", {"url": {"type": "string"}, "prompt": {"type": "string"}}, ["url"]),
                    ("Task", {"description": {"type": "string"}, "prompt": {"type": "string"}}, ["prompt"]))]
BASH_CALL = "bash" + MARK + '{"name":"bash","args":{"command":"ls -la","description":"List files in working directory"}}'


def chat_reply(content, reasoning="Plan it.", usage=None):
    return {"id": "c", "object": "chat.completion", "model": "inkling", "choices": [{"index": 0, "message": {
        "role": "assistant", "content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 30}}


class Fresh(Http):
    """Http with an empty Responses store and reasoning memory for each test."""

    def setUp(self):
        super().setUp()
        fresh = patch.multiple(gw, RESPONSES=gw.ResponsesStore(), INKLING_MEMORY=OrderedDict(),
                               INKLING_NAMES=OrderedDict(), INKLING_MEMORY_SIZE=0)
        fresh.start()
        self.addCleanup(fresh.stop)

    def sent(self):
        return self.last("/v1/chat/completions")


class Responses(Fresh):
    def test_non_streamed_calls_are_function_call_items(self):
        self.reply = lambda body: Reply(CHAT_REPLY)
        status, reply = self.post("/v1/responses", {"model": "inkling", "input": "weather and time?",
                                                    "tools": RESPONSES_TOOLS, "max_output_tokens": 512,
                                                    "instructions": "Be brief."})
        self.assertEqual((status, reply["object"], reply["status"]), (200, "response", "completed"))
        kinds = [(i["type"], i.get("name")) for i in reply["output"]]
        self.assertEqual(kinds, [("reasoning", None), ("message", None), ("function_call", "get_weather"),
                                 ("function_call", "get_time")])
        self.assertEqual(reply["output"][0]["content"][0], {"type": "reasoning_text", "text": "Need both."})
        self.assertEqual(reply["output"][1]["content"][0]["text"], "Checking now.")
        self.assertNotIn(MARK, json.dumps(reply))
        calls = reply["output"][2:]
        self.assertEqual([json.loads(c["arguments"]) for c in calls], [{"city": "Wellington"}, {"city": "Auckland"}])
        self.assertTrue(all(c["call_id"].startswith("call_") and c["id"].startswith("fc_") and c["status"] == "completed"
                            for c in calls))
        self.assertEqual(reply["usage"], {"input_tokens": 201, "input_tokens_details": {"cached_tokens": 150},
                                          "output_tokens": 83, "output_tokens_details": {"reasoning_tokens": 18},
                                          "total_tokens": 284})
        sent = self.sent()
        self.assertEqual(sent["messages"], [{"role": "system", "content": "Be brief."},
                                            {"role": "user", "content": "weather and time?"}])
        self.assertEqual([t["function"]["name"] for t in sent["tools"]], TOOLS)
        self.assertEqual((sent["max_tokens"], sent["reasoning_effort"], sent["model"]), (512, "high", "inkling"))
        self.assertEqual(reply["reasoning"]["effort"], "high")

    def test_streamed_events_carry_the_calls(self):
        self.reply = lambda body: Reply(sse(stream_chunks()), content_type="text/event-stream")
        status, data = self.post("/responses", {"model": "inkling", "input": "go", "stream": True,     # un-prefixed
                                                "tools": RESPONSES_TOOLS}, raw=True)
        self.assertEqual(status, 200)
        stream = events(data)
        kinds = [name for name, _ in stream]
        self.assertEqual(kinds[:2], ["response.created", "response.in_progress"])
        self.assertEqual(kinds[-1], "response.completed")
        self.assertEqual([p["sequence_number"] for _, p in stream], list(range(len(stream))))
        added = [p["item"] for n, p in stream if n == "response.output_item.added"]
        self.assertEqual([(i["type"], i.get("name")) for i in added],
                         [("reasoning", None), ("message", None), ("function_call", "get_weather"), ("function_call", "get_time")])
        text = "".join(p["delta"] for n, p in stream if n == "response.output_text.delta")
        thought = "".join(p["delta"] for n, p in stream if n == "response.reasoning_text.delta")
        self.assertEqual((text, thought), ("Checking now.", "Say it, then call both."))
        args = [(p["output_index"], json.loads(p["arguments"]), p["name"]) for n, p in stream
                if n == "response.function_call_arguments.done"]
        self.assertEqual(args, [(2, {"city": "Wellington"}, "get_weather"), (3, {"city": "Auckland"}, "get_time")])
        self.assertEqual(sum(n == "response.function_call_arguments.delta" for n in kinds), 2)
        done = [p["item"]["type"] for n, p in stream if n == "response.output_item.done"]
        self.assertEqual(done, ["reasoning", "message", "function_call", "function_call"])
        final = stream[-1][1]["response"]
        self.assertEqual([i["type"] for i in final["output"]], ["reasoning", "message", "function_call", "function_call"])
        self.assertEqual(final["usage"]["input_tokens"], 203)        # the usage chunk after the terminal one
        self.assertEqual(final["tensorfold"], {"x": 1})
        self.assertNotIn(MARK, data.decode())
        self.assertEqual(self.sent()["stream_options"], {"include_usage": True})

    def test_previous_response_id_round_trip(self):
        self.reply = lambda body: Reply(ANSWER if any(m["role"] == "tool" for m in body["messages"]) else CHAT_REPLY)
        _, first = self.post("/v1/responses", {"model": "inkling", "input": "weather and time?", "tools": RESPONSES_TOOLS})
        calls = [i for i in first["output"] if i["type"] == "function_call"]
        status, second = self.post("/v1/responses", {
            "model": "inkling", "previous_response_id": first["id"], "tools": RESPONSES_TOOLS,
            "input": [{"type": "function_call_output", "call_id": c["call_id"], "output": out}
                      for c, out in zip(calls, ("14C", [{"type": "input_text", "text": "09:30"}]))]})
        self.assertEqual(status, 200)
        self.assertEqual(second["output"][-1]["content"][0]["text"], "It is 14C in Wellington and 09:30 in Auckland.")
        self.assertEqual(second["previous_response_id"], first["id"])
        messages = self.sent()["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "tool"])
        assistant = messages[1]
        self.assertEqual((assistant["reasoning_content"], assistant["content"]), ("Need both.", "Checking now."))
        self.assertEqual([(c["id"], c["function"]["name"]) for c in assistant["tool_calls"]],
                         [(c["call_id"], c["name"]) for c in calls])
        self.assertEqual([(m["tool_call_id"], m["content"]) for m in messages[2:]],
                         [(calls[0]["call_id"], "14C"), (calls[1]["call_id"], "09:30")])
        status, stored = self.request_json("GET", f"/v1/responses/{first['id']}")
        self.assertEqual((status, stored["id"]), (200, first["id"]))
        self.assertEqual(self.request_json("DELETE", f"/responses/{first['id']}"),
                         (200, {"id": first["id"], "object": "response", "deleted": True}))
        status, error = self.post("/v1/responses", {"model": "inkling", "previous_response_id": first["id"],
                                                    "input": "again"})
        self.assertEqual(status, 400)
        self.assertIn("not stored here", error["error"]["message"])

    def test_stateless_replay_gets_its_reasoning_back_from_memory(self):
        self.reply = lambda body: Reply(ANSWER if any(m["role"] == "tool" for m in body["messages"]) else CHAT_REPLY)
        _, first = self.post("/v1/responses", {"model": "inkling", "input": "weather?", "tools": RESPONSES_TOOLS,
                                               "store": False})
        self.assertIsNone(gw.RESPONSES.get(first["id"]))              # store: false keeps nothing
        replay = [{"role": "user", "content": "weather?"}]
        replay += [{k: i[k] for k in ("type", "call_id", "name", "arguments")} for i in first["output"]
                   if i["type"] == "function_call"]                    # the client dropped the reasoning item
        replay += [{"type": "function_call_output", "call_id": i["call_id"], "output": "ok"} for i in first["output"]
                   if i["type"] == "function_call"]
        status, _ = self.post("/v1/responses", {"model": "inkling", "input": replay, "tools": RESPONSES_TOOLS, "store": False})
        self.assertEqual(status, 200)
        self.assertEqual(self.sent()["messages"][1]["reasoning_content"], "Need both.")

    def test_item_reference_and_reasoning_items(self):
        self.reply = lambda body: Reply(CHAT_REPLY)
        _, first = self.post("/v1/responses", {"model": "inkling", "input": "go", "tools": RESPONSES_TOOLS})
        reasoning = first["output"][0]
        call = first["output"][2]
        self.post("/v1/responses", {"model": "inkling", "tools": RESPONSES_TOOLS, "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "go"}]},
            {"type": "item_reference", "id": reasoning["id"]},
            {"type": "item_reference", "id": call["id"]},
            {"type": "function_call_output", "call_id": call["call_id"], "output": "done"}]})
        assistant = self.sent()["messages"][1]
        self.assertEqual((assistant["reasoning_content"], assistant["tool_calls"][0]["id"]), ("Need both.", call["call_id"]))
        status, error = self.post("/v1/responses", {"model": "inkling", "input": [{"type": "item_reference", "id": "rs_nope"}]})
        self.assertEqual(status, 400)
        self.assertIn("rs_nope", error["error"]["message"])

    def test_summary_only_reasoning_items_are_read(self):
        self.post("/v1/responses", {"model": "inkling", "input": [
            {"role": "user", "content": "x"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Summed."}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Done."}]},
            {"role": "user", "content": "y"}]})
        self.assertEqual(self.sent()["messages"][1], {"role": "assistant", "content": "Done.", "reasoning_content": "Summed."})

    def test_malformed_calls_stay_text(self):
        bad = "get_weather" + MARK + '{"name":"get_weather","args":{"city":}}'
        self.reply = lambda body: Reply(chat_reply(bad))
        _, reply = self.post("/v1/responses", {"model": "inkling", "input": "go", "tools": RESPONSES_TOOLS})
        self.assertEqual([i["type"] for i in reply["output"]], ["reasoning", "message"])
        self.assertEqual(reply["output"][1]["content"][0]["text"], bad)

    def test_refusals_match_the_engine(self):
        for body, words in (({"input": "x", "tools": [{"type": "web_search"}]}, "function tools only"),
                            ({"input": "x", "background": True}, "background"),
                            ({"input": []}, "non-empty"),
                            ({"input": [{"type": "computer_call"}]}, "not supported")):
            with self.subTest(body=body):
                status, error = self.post("/v1/responses", {"model": "inkling", **body})
                self.assertEqual(status, 400)
                self.assertIn(words, error["error"]["message"])

    def test_upstream_errors_keep_their_status(self):
        def refuse(body):
            raise gw.urllib.error.HTTPError("u", 400, "bad", {}, __import__("io").BytesIO(
                b'{"error": {"message": "the rendered prompt is too long"}}'))
        self.reply = refuse
        status, error = self.post("/v1/responses", {"model": "inkling", "input": "x"})
        self.assertEqual((status, error["error"]["message"]), (400, "the rendered prompt is too long"))

    def request_json(self, method, path):
        request = gw.urllib.request.Request(f"http://127.0.0.1:{self.server.server_port}{path}", method=method)
        try:
            with gw.urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except gw.urllib.error.HTTPError as response:
            return response.code, json.load(response)


class Memory(Fresh):
    """A reply's reasoning returns to the next turn on every API, whether or not the client sends it back."""

    def test_openai_chat_turn_without_reasoning_content_gets_it_back(self):
        self.reply = lambda body: Reply(CHAT_REPLY)
        _, reply = self.post("/v1/chat/completions", {"model": "inkling", "tools": OPENAI_TOOLS,
                                                      "messages": [{"role": "user", "content": "go"}]})
        assistant = {k: v for k, v in reply["choices"][0]["message"].items() if k != "reasoning_content"}
        self.post("/v1/chat/completions", {"model": "inkling", "tools": OPENAI_TOOLS, "messages": [
            {"role": "user", "content": "go"}, assistant,
            *[{"role": "tool", "tool_call_id": c["id"], "content": "ok"} for c in assistant["tool_calls"]]]})
        self.assertEqual(self.sent()["messages"][1]["reasoning_content"], "Need both.")

    def test_a_text_reply_is_found_by_its_text_and_sent_reasoning_wins(self):
        self.reply = lambda body: Reply(chat_reply("Paris.", reasoning="Capital of France."))
        self.post("/chat/completions", {"model": "inkling", "messages": [{"role": "user", "content": "capital?"}]})
        follow = [{"role": "user", "content": "capital?"}, {"role": "assistant", "content": "Paris."},
                  {"role": "user", "content": "and Spain?"}]
        self.post("/chat/completions", {"model": "inkling", "messages": follow})
        self.assertEqual(self.sent()["messages"][1]["reasoning_content"], "Capital of France.")
        follow[1] = {**follow[1], "reasoning_content": "Mine."}
        self.post("/chat/completions", {"model": "inkling", "messages": follow})
        self.assertEqual(self.sent()["messages"][1]["reasoning_content"], "Mine.")

    def test_streamed_replies_are_remembered(self):
        self.reply = lambda body: (Reply(sse(stream_chunks()), content_type="text/event-stream") if body.get("stream")
                                   else Reply(ANSWER))
        status, data = self.post("/v1/chat/completions", {"model": "inkling", "stream": True, "tools": OPENAI_TOOLS,
                                                          "messages": [{"role": "user", "content": "go"}]}, raw=True)
        ids = [t["id"] for _, c in events(data) if c != "[DONE]" for ch in c["choices"] for t in ch["delta"].get("tool_calls") or []]
        self.post("/v1/chat/completions", {"model": "inkling", "tools": OPENAI_TOOLS, "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "Checking now.", "tool_calls": [
                {"id": i, "type": "function", "function": {"name": n, "arguments": "{}"}} for i, n in zip(ids, TOOLS)]},
            *[{"role": "tool", "tool_call_id": i, "content": "ok"} for i in ids]]})
        self.assertEqual(self.sent()["messages"][1]["reasoning_content"], "Say it, then call both.")

    def test_anthropic_turn_without_thinking_gets_it_back(self):
        self.reply = lambda body: Reply(CHAT_REPLY)
        _, reply = self.post("/v1/messages", {"model": "inkling", "max_tokens": 256, "tools": ANTHROPIC_TOOLS,
                                              "messages": [{"role": "user", "content": "go"}]})
        stripped = [b for b in reply["content"] if b["type"] != "thinking"]
        body = {"model": "inkling", "max_tokens": 256, "tools": ANTHROPIC_TOOLS, "messages": [
            {"role": "user", "content": "go"}, {"role": "assistant", "content": stripped},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": b["id"], "content": "ok"}
                                         for b in stripped if b["type"] == "tool_use"]}]}
        self.post("/messages", body)                                   # un-prefixed, as some clients send it
        self.assertEqual(self.sent()["messages"][1]["reasoning_content"], "Need both.")
        status, count = self.post("/messages/count_tokens", {k: v for k, v in body.items() if k != "max_tokens"})
        self.assertEqual((status, self.last("/tokenize")["messages"][1]["reasoning_content"]), (200, "Need both."))

    def test_memory_is_bounded(self):
        with patch.multiple(gw, INKLING_MEMORY_MAX=3):
            for i in range(10):
                gw.remember_reply(f"r{i}", f"text {i}", [])
            self.assertEqual(len(gw.INKLING_MEMORY), 3)
            self.assertEqual(gw.recall_reasoning([{"role": "assistant", "content": "text 9"}])[1], 1)
            self.assertEqual(gw.recall_reasoning([{"role": "assistant", "content": "text 0"}])[1], 0)


class ClaudeCodeStyle(Fresh):
    """Messages API with a Claude-Code-like tool set: capitalized names the model writes lowercase."""

    def test_bash_call_becomes_tool_use_and_its_turn_re_renders_as_written(self):
        self.reply = lambda body: Reply(ANSWER if any(m["role"] == "tool" for m in body["messages"])
                                        else chat_reply("I'll list the files." + BASH_CALL, reasoning="List first."))
        status, reply = self.post("/messages", {"model": "inkling", "max_tokens": 1024, "tools": CLAUDE_TOOLS,
                                                "messages": [{"role": "user", "content": "what is here?"}]})
        self.assertEqual((status, reply["stop_reason"]), (200, "tool_use"))
        use = reply["content"][-1]
        self.assertEqual((use["type"], use["name"], use["input"]),
                         ("tool_use", "Bash", {"command": "ls -la", "description": "List files in working directory"}))
        self.assertEqual([b["type"] for b in reply["content"]], ["thinking", "text", "tool_use"])
        self.post("/v1/messages", {"model": "inkling", "max_tokens": 1024, "tools": CLAUDE_TOOLS, "messages": [
            {"role": "user", "content": "what is here?"}, {"role": "assistant", "content": reply["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"], "content": "a.txt\nb.txt"}]}]})
        assistant = self.sent()["messages"][1]
        # the model's own spelling comes back for the engine, so the turn renders as the model wrote it
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(assistant["reasoning_content"], "List first.")
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
                         {"command": "ls -la", "description": "List files in working directory"})

    def test_streamed_several_calls_after_thinking(self):
        pieces = ["Two", " things.", "bash" + MARK + '{"name":"bash","args":{"command":"pwd"}}', "read" + MARK,
                  '{"name":"read","args":{"file_path":"/etc/hostname"}}']
        self.reply = lambda body: Reply(sse(stream_chunks(pieces=pieces)), content_type="text/event-stream")
        status, data = self.post("/v1/messages", {"model": "inkling", "max_tokens": 512, "stream": True,
                                                  "tools": CLAUDE_TOOLS, "messages": [{"role": "user", "content": "go"}]},
                                 raw=True)
        blocks, acc = {}, {}
        for name, payload in events(data):
            if name == "content_block_start":
                blocks[payload["index"]] = payload["content_block"]
            elif name == "content_block_delta":
                d = payload["delta"]
                acc[payload["index"]] = acc.get(payload["index"], "") + (d.get("thinking") or d.get("text") or d.get("partial_json") or "")
        self.assertEqual([(blocks[i]["type"], blocks[i].get("name")) for i in sorted(blocks)],
                         [("thinking", None), ("text", None), ("tool_use", "Bash"), ("tool_use", "Read")])
        self.assertEqual((acc[1], json.loads(acc[2]), json.loads(acc[3])), ("Two things.", {"command": "pwd"},
                                                                            {"file_path": "/etc/hostname"}))
        stop = [p for n, p in events(data) if n == "message_delta"][0]
        self.assertEqual(stop["delta"]["stop_reason"], "tool_use")

    def test_a_malformed_call_is_text_and_count_tokens_is_unprefixed(self):
        bad = "bash" + MARK + '{"name":"bash","args":{"command":'
        self.reply = lambda body: Reply(chat_reply(bad))
        _, reply = self.post("/messages", {"model": "inkling", "max_tokens": 64, "tools": CLAUDE_TOOLS,
                                           "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual([(b["type"], b.get("text")) for b in reply["content"]][-1], ("text", bad))
        self.assertEqual(reply["stop_reason"], "end_turn")
        status, count = self.post("/messages/count_tokens", {"model": "inkling", "tools": CLAUDE_TOOLS,
                                                             "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual((status, count["input_tokens"]), (200, 7))


class OtherRecipesResponses(Http):
    """Without --family the Responses route is still relayed to TensorFold as before."""
    from test_inkling import OTHER_ARGS as args

    def test_responses_are_relayed(self):
        self.reply = lambda body: Reply({"id": "r", "object": "response", "status": "completed", "output": []})
        status, reply = self.post("/v1/responses", {"model": "test-model", "input": "x", "tools": RESPONSES_TOOLS})
        self.assertEqual((status, reply["id"]), (200, "r"))
        self.assertEqual(self.requests[-1][0], "/v1/responses")


if __name__ == "__main__":
    unittest.main()
