"""Inkling 2 (--family inkling2): a reply comes back as the turn the engine holds.

The engine reuses its prompt state only up to the first token where a request's prompt differs from the previous
prompt plus the reply it generated, so the assistant turn a client sends back must re-render to the model's own
tokens. These tests take a reply as the model wrote it (message markup and all), give it to the gateway as the engine
does (reasoning split off, markup dropped), send the turn back as clients do (Codex drops reasoning and whitespace-only
messages; Claude Code re-serializes arguments), and render what reaches the engine with the deployed chat template
(engine-C, through jinja2 with Hugging Face's tojson, as the engine's own template runtime reads it): wherever the
gateway can make it so, the render is the model's reply, character for character. Also: the replay check, the
request log's prefix note, parallel_tool_calls false, and the memory file.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inkling import MARK, Reply, events, gw                        # noqa: E402
from test_inkling_responses import Fresh                               # noqa: E402

try:
    import jinja2
except ImportError:                          # the template tests need it; the gateway itself is stdlib only
    jinja2 = None

TEMPLATE = Path(os.environ.get("INKLING_TEMPLATE") or Path(__file__).resolve().parents[2]
                / "config" / "chat_template.engine-C.jinja")
CAN_RENDER = jinja2 is not None and TEMPLATE.is_file()
HEAD, THINK, TEXT, END, EOS = "<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>", \
    "<|content_model_end_sampling|>"
TOOLS = ["exec_command", "apply_patch", "update_plan"]
RESPONSES_TOOLS = [{"type": "function", "name": n, "description": n, "parameters": {"type": "object"}} for n in TOOLS]
CHAT_TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {"type": "object"}}}
              for n in TOOLS]


def call_message(name, raw_json):
    return HEAD + name + MARK + raw_json + END


def canonical(name, args):
    """A call's JSON as the template writes it (what a model trained on it writes)."""
    return gw.inkling_call_json(name, args)


def model_reply(reasoning, text, calls, *, late_text=None):
    """A reply as the model writes it: its own header, a thinking message, a text message, one message a call."""
    out = HEAD + THINK + reasoning + END
    if text is not None:
        out += HEAD + TEXT + text + END
    for name, raw in calls:
        out += call_message(name, raw)
    if late_text is not None:
        out += HEAD + TEXT + late_text + END
    return out + EOS


def engine_view(raw):
    """(reasoning_content, content) as the native inkling2 server returns a reply: the first thinking message's body,
    and the rest without the message header, the text kind and the message end (reply_text.inklingAnswer), the
    end-of-sampling token being the stop token."""
    assert raw.endswith(EOS)
    body = raw[:-len(EOS)]
    reasoning = ""
    if body.startswith(HEAD + THINK):
        end = body.find(END)
        end = len(body) if end < 0 else end      # a thinking message a stop cut: all reasoning
        reasoning, body = body[len(HEAD + THINK):end], body[end + len(END):]
    for mark in (HEAD, TEXT, END):
        body = body.replace(mark, "")
    return reasoning, body


def engine_messages(messages):
    """The engine's normalization before its template: content parts joined, none as "", arguments parsed (in order)."""
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), list):
            m["content"] = "".join(p.get("text", "") for p in m["content"])
        elif m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                calls.append({**c, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def render(messages):
    """The deployed template's rendering, as the engine runs it (Hugging Face's tojson: json.dumps, ensure_ascii off)."""
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def fail(message):
        raise ValueError(message)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = fail
    return env.from_string(TEMPLATE.read_text()).render(messages=engine_messages(messages))


def turn_of(rendered):
    """The assistant turn of a one-message render: from its first header to the end."""
    return rendered[rendered.index(HEAD):]


SPECIAL = gw.re.compile(r"(<\|[a-z_]+\|>)")
SPECIAL_IDS = {}


def toks(text):
    """Character tokens, a special token one token (its id past Unicode's range)."""
    out = []
    for part in SPECIAL.split(text):
        if SPECIAL.fullmatch(part):
            out.append(SPECIAL_IDS.setdefault(part, 0x200000 + len(SPECIAL_IDS)))
        else:
            out += [ord(ch) for ch in part]
    return out


def detok(ids):
    names = {v: k for k, v in SPECIAL_IDS.items()}
    return "".join(names.get(t) or chr(t) for t in ids)


def completion(raw, usage=None, ids=None):
    reasoning, content = engine_view(raw)
    out = {"id": "c", "object": "chat.completion", "model": "inkling-2-small", "choices": [{"index": 0, "message": {
        "role": "assistant", "content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 120,
                           "prompt_tokens_details": {"cached_tokens": 900}}}
    if ids is not None:
        out["tensorfold"] = {"engine": "inkling2", "token_ids": ids}
    return out


def stream_chunks(raw, pieces=7, usage=None, ids=None):
    """The reply streamed as the engine streams it: reasoning deltas, then the content in pieces, the stop, the usage."""
    reasoning, content = engine_view(raw)
    env = {"id": "c", "object": "chat.completion.chunk", "model": "inkling-2-small"}
    out = [{**env, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": reasoning}}]}]
    step = max(1, len(content) // pieces)
    for i in range(0, len(content), step):
        out.append({**env, "choices": [{"index": 0, "delta": {"content": content[i:i + step]}}]})
    final = {**env, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    if ids is not None:
        final["tensorfold"] = {"engine": "inkling2", "token_ids": ids}
    out.append(final)
    out.append({**env, "choices": [], "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 120,
                                                         "prompt_tokens_details": {"cached_tokens": 900}}})
    return out


def sse(chunks):
    return b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"


ARGS = {"cmd": "rg -n \"TODO\" src | head -20", "max_output_tokens": 4000, "workdir": "/home/u/proj",
        "yield_time_ms": 10000}
PATCH_ARGS = {"input": "*** Begin Patch\n*** Update File: a.py\n@@\n-x = 1\n+x = 1.5  # été — \"ok\"\t\\\n*** End Patch\n"}
PLAN_ARGS = {"explanation": "Two steps", "plan": [{"status": "in_progress", "step": "Read"}, {"step": "Fix", "status": "pending"}]}


@unittest.skipUnless(CAN_RENDER, "needs jinja2 and the deployed chat template")
class Template(unittest.TestCase):
    """inkling_call_json and the restored turn against the deployed template itself."""

    def rerender(self, raw, *, single=False, client=None):
        """The model's reply through the gateway (parsed, remembered), sent back by ``client`` (a function of the chat
        message a client got), restored, rendered."""
        with patch.multiple(gw, INKLING_MEMORY=OrderedDict(), INKLING_NAMES=OrderedDict(), INKLING_MEMORY_SIZE=0):
            resp, _, turn = gw.inkling_reply(completion(raw), TOOLS, single)
            turn.remember()
            msg = resp["choices"][0]["message"]
            sent = client(msg) if client else msg
            restored, _ = gw.recall_reasoning([sent])
            return turn_of(render(restored)), turn.record()

    def test_call_json_is_the_templates(self):
        for args in (ARGS, PATCH_ARGS, PLAN_ARGS, {}, {"b": [1, 2.0, True, None], "a": {"z": 1, "y": {"x": "1e3"}},
                                                     "C": 1e16, "_": -0.0, "num": 12345678901234567890}):
            message = {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {
                "name": "exec_command", "arguments": json.dumps(args)}}]}
            self.assertEqual(turn_of(render([message])),
                             call_message("exec_command", canonical("exec_command", args)) + EOS)

    def test_a_canonical_reply_comes_back_whole(self):
        raw = model_reply("I need the TODOs, then a plan.", "Looking.",
                          [("exec_command", canonical("exec_command", ARGS)),
                           ("update_plan", canonical("update_plan", PLAN_ARGS))])
        codex = lambda m: {k: v for k, v in m.items() if k != "reasoning_content"}          # noqa: E731
        got, record = self.rerender(raw, client=codex)
        self.assertEqual(got, raw)
        self.assertEqual(record["notes"], [])

    def test_white_space_text_and_escapes_come_back(self):
        raw = model_reply("Patch it.", "\n", [("apply_patch", canonical("apply_patch", PATCH_ARGS))])
        drop_blank = lambda m: {"role": "assistant", "content": None, "tool_calls": m["tool_calls"]}  # noqa: E731
        got, _ = self.rerender(raw, client=drop_blank)
        self.assertEqual(got, raw)

    def test_re_serialized_arguments_come_back_in_the_models_order(self):
        nested = {"cmd": "ls", "env": {"B": "2", "A": "1"}, "timeout": 1.0}
        raw = model_reply("List.", None, [("exec_command", canonical("exec_command", nested))])

        def javascript(m):                    # JSON.parse then JSON.stringify: 1.0 is 1, keys as the client sorts them
            calls = [{**c, "function": {**c["function"], "arguments": json.dumps(
                {"timeout": 1, "env": {"A": "1", "B": "2"}, "cmd": "ls"})}} for c in m["tool_calls"]]
            return {"role": "assistant", "content": "", "tool_calls": calls}

        got, _ = self.rerender(raw, client=javascript)
        self.assertEqual(got, raw)

    def test_what_the_template_cannot_write_back_is_named(self):
        unsorted = ('{"name":"exec_command","args":{"cmd":"ls","yield_time_ms":1000,"max_output_tokens":100}}')
        spaced = '{"name": "exec_command", "args": {"cmd": "ls"}}'
        cases = [
            (model_reply("R", None, [("exec_command", unsorted)]), "args' keys are not sorted"),
            (model_reply("R", None, [("exec_command", spaced)]), "spacing, escapes or number spelling"),
            (model_reply("R", None, [("exec_command", canonical("exec_command", {"cmd": "ls"}))], late_text="Done."),
             "text after a call"),
            (model_reply("R", "a" + END + HEAD + THINK + "more", [("exec_command", canonical("exec_command", {"cmd": "ls"}))]),
             "markup in text: <|content_thinking|>"),
        ]
        for raw, note in cases:
            with self.subTest(note=note):
                got, record = self.rerender(raw)
                self.assertNotEqual(got, raw)
                self.assertTrue(any(note in n for n in record["notes"]), record["notes"])

    def test_several_calls_under_parallel_tool_calls_false(self):
        raw = model_reply("Both.", None, [("exec_command", canonical("exec_command", {"cmd": "ls"})),
                                          ("exec_command", canonical("exec_command", {"cmd": "pwd"}))])
        with patch.object(gw, "INKLING_PARALLEL", "all"):       # every call: the turn comes back whole
            got, record = self.rerender(raw, single=gw.inkling_single(True))
        self.assertEqual(got, raw)
        with patch.object(gw, "INKLING_PARALLEL", "drop"):
            got, record = self.rerender(raw, single=gw.inkling_single(True))
        self.assertNotEqual(got, raw)
        self.assertIn("a later call dropped under parallel_tool_calls false (the engine holds it: this reply re-runs)",
                      record["notes"])


class Parser(unittest.TestCase):
    def test_canonical_calls_have_no_notes(self):
        parser = gw.InklingCalls(TOOLS)
        parser.feed("Checking." + "exec_command" + MARK + canonical("exec_command", ARGS))
        parser.finish()
        self.assertEqual(parser.notes, [])

    def test_a_header_and_json_name_that_differ(self):
        parser = gw.InklingCalls(["Bash"])
        parser.feed("bash" + MARK + '{"name":"Bash","args":{"command":"ls"}}')
        self.assertTrue(any("names 'Bash' under the header 'bash'" in n for n in parser.notes), parser.notes)


class Restore(Fresh):
    """recall_reasoning puts the client's copy back to the reply where it is the same turn, and only there."""

    def remember(self, text="Checking now.", reasoning="Need both.", args=None, before=None):
        turn = gw.InklingTurn(before)
        turn.reasoning.append(reasoning)
        turn.text.append(text)
        turn.call("call_1", {"name": "exec_command", "written": "exec_command",
                             "arguments": json.dumps(args or {"cmd": "ls", "workdir": "/w"}, ensure_ascii=False)})
        turn.remember()

    def back(self, **message):
        return gw.recall_reasoning([{"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "exec_command", "arguments": '{"cmd": "ls", "workdir": "/w"}'}}],
                                     **message}])[0][0]

    def test_trimmed_copies_come_back_exact(self):
        self.remember(text=" Checking now.\n", reasoning="Need both.\n")
        got = self.back(content="Checking now.", reasoning_content="Need both.")
        self.assertEqual((got["content"], got["reasoning_content"]), (" Checking now.\n", "Need both.\n"))
        self.assertEqual(gw.INKLING_LOCAL.recall["text"], 1)
        self.assertEqual(gw.INKLING_LOCAL.recall["trimmed"], 1)

    def test_a_changed_turn_keeps_the_clients_version(self):
        self.remember()
        got = self.back(content="Something else.", reasoning_content="My own reasoning.")
        self.assertEqual((got["content"], got["reasoning_content"]), ("Something else.", "My own reasoning."))
        edited = gw.recall_reasoning([{"role": "assistant", "content": "Checking now.", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": '{"cmd": "rm"}'}}]}])
        self.assertEqual(edited[0][0]["tool_calls"][0]["function"]["arguments"], '{"cmd": "rm"}')

    def test_arguments_come_back_in_the_models_key_order(self):
        self.remember(args={"workdir": "/w", "cmd": "ls"})
        got = self.back(content="Checking now.")
        self.assertEqual(got["tool_calls"][0]["function"]["arguments"], '{"workdir": "/w", "cmd": "ls"}')

    def test_the_same_text_after_different_messages_stays_apart(self):
        first, second = {"role": "user", "content": "one"}, {"role": "user", "content": "two"}
        for before, thought in ((first, "First."), (second, "Second.")):
            turn = gw.InklingTurn(before)
            turn.reasoning.append(thought)
            turn.text.append("Done.")
            turn.remember()
        history = [first, {"role": "assistant", "content": "Done."}, second, {"role": "assistant", "content": "Done."}]
        got = gw.recall_reasoning(history)[0]
        self.assertEqual((got[1]["reasoning_content"], got[3]["reasoning_content"]), ("First.", "Second."))


class Diagnostics(Fresh):
    def test_the_prefix_note(self):
        turn = gw.InklingTurn({"role": "user", "content": "go"})
        turn.reasoning.append("R")
        turn.text.append("Hi.")
        turn.notes.append("text after a call (the template writes a turn's text before its calls)")
        turn.usage = {"prompt_tokens": 1000, "completion_tokens": 50}
        turn.remember()
        history = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "Hi."}]
        for cached, expect in ((1049, ""), (1000, "the last reply was not reused (49 of its tokens re-ran): text after"),
                               (12, "cached 12 is short of the last reply's prompt 1000")):
            gw.recall_reasoning(history)
            note = gw.inkling_notes({"prompt_tokens_details": {"cached_tokens": cached}})
            self.assertIn(expect, note) if expect else self.assertNotIn("prefix", note)

    @unittest.skipUnless(CAN_RENDER, "needs jinja2 and the deployed chat template")
    def test_the_replay_check_finds_the_first_token_that_differs(self):
        """Character tokens: /tokenize renders with the deployed template, /detokenize joins characters."""
        raw = model_reply("Both.", None, [("exec_command", canonical("exec_command", {"cmd": "ls"})),
                                          ("exec_command", canonical("exec_command", {"cmd": "pwd"}))])
        ids = toks(raw)

        def engine(path, payload=None, **kwargs):
            if path == "/tokenize":
                return Reply({"tokens": toks(render(payload["messages"]))})
            return Reply({"prompt": detok(payload["tokens"])})

        for policy, differs in (("all", False), ("drop", True)):
            with self.subTest(policy=policy), patch.object(gw, "INKLING_PARALLEL", policy), \
                    patch.object(gw, "upstream", side_effect=engine):
                resp, _, turn = gw.inkling_reply(completion(raw, ids=ids), TOOLS, gw.inkling_single(True))
                record = turn.record()
                out = io.StringIO()
                self.assertTrue(gw.INKLING_CHECKING.acquire(timeout=10))
                with redirect_stdout(out):
                    gw._replay_check(record, ids, "stop")
                line = out.getvalue()
                if differs:
                    at = len(ids) - len(toks(call_message("exec_command", canonical("exec_command", {"cmd": "pwd"})) + EOS))
                    self.assertIn(f"re-renders differently from token {at} of {len(ids)} ({len(ids) - at} before its end)", line)
                    self.assertIn(f"the reply has {detok(ids[at:at + 12])!r}", line)
                    self.assertIn("its re-render '<|content_model_end_sampling|>'", line)
                    self.assertIn("a later call dropped", line)
                else:
                    self.assertEqual(line, "")


class TokenIds(Fresh):
    """The gateway asks the engine for each reply's token ids and keeps them out of the client's copy."""

    def setUp(self):
        super().setUp()
        quiet = patch.object(gw, "inkling_replay_check")       # the check itself: Diagnostics
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_responses_codex_round_trip_and_token_ids(self):
        raw = model_reply("Read both.", "\n", [("exec_command", canonical("exec_command", {"cmd": "cat a"})),
                                               ("exec_command", canonical("exec_command", {"cmd": "cat b"}))])
        self.reply = lambda body: Reply(completion(raw, ids=[1, 2, 3]))
        every = patch.object(gw, "INKLING_PARALLEL", "all")
        every.start()
        self.addCleanup(every.stop)
        status, reply = self.post("/responses", {"model": "inkling", "input": "read a and b", "tools": RESPONSES_TOOLS,
                                                 "parallel_tool_calls": False})
        self.assertEqual(status, 200)
        self.assertIs(self.sent()["return_token_ids"], True)
        self.assertNotIn("token_ids", json.dumps(reply))
        calls = [i for i in reply["output"] if i["type"] == "function_call"]
        self.assertEqual(len(calls), 2)                        # both, though the client asked for one
        # Codex sends the history back without reasoning, and without a message that is only white space
        items = [{"role": "user", "content": "read a and b"}] + [
            {"type": "function_call", "call_id": c["call_id"], "name": c["name"], "arguments": c["arguments"]}
            for c in calls] + [{"type": "function_call_output", "call_id": c["call_id"], "output": "x"} for c in calls]
        self.post("/responses", {"model": "inkling", "input": items, "tools": RESPONSES_TOOLS,
                                 "parallel_tool_calls": False})
        turn = self.sent()["messages"][1]
        self.assertEqual((turn["reasoning_content"], turn["content"]), ("Read both.", "\n"))
        if CAN_RENDER:
            self.assertEqual(turn_of(render([turn])), raw)

    def test_streamed_chat_keeps_token_ids_out_unless_asked(self):
        raw = model_reply("R", "Hi.", [])
        self.reply = lambda body: Reply(sse(stream_chunks(raw, ids=[5, 6])), content_type="text/event-stream")
        for asked in (False, True):
            body = {"model": "inkling", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
            if asked:
                body["return_token_ids"] = True
            _, data = self.post("/v1/chat/completions", body, raw=True)
            self.assertIs(self.sent()["return_token_ids"], True)
            seen = [c.get("tensorfold", {}).get("token_ids") for _, c in events(data) if c != "[DONE]"]
            self.assertEqual([s for s in seen if s], [[5, 6]] if asked else [])

    def test_chat_json_and_messages(self):
        raw = model_reply("R", "Hi.", [])
        self.reply = lambda body: Reply(completion(raw, ids=[5, 6]))
        _, reply = self.post("/v1/chat/completions", {"model": "inkling", "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("token_ids", json.dumps(reply))
        self.assertEqual(reply["tensorfold"], {"engine": "inkling2"})
        _, reply = self.post("/v1/messages", {"model": "inkling", "max_tokens": 64,
                                              "messages": [{"role": "user", "content": "hi"}]})
        self.assertIs(self.sent()["return_token_ids"], True)
        self.assertNotIn("token_ids", json.dumps(reply))

    def test_no_replay_check_no_token_ids(self):
        self.reply = lambda body: Reply(completion(model_reply("R", "Hi.", [])))
        with patch.object(gw, "INKLING_REPLAY_CHECK", False):
            self.post("/v1/chat/completions", {"model": "inkling", "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("return_token_ids", self.sent())


def engine_stopped(raw, client_stop=None):
    """A reply the engine's stop ended (core/lanes: the reply ends at the token that completes the first stop string
    in its decoded text; the content loses the stop string): (reasoning_content, content, the reply's tokens, the
    matched stop)."""
    stops = ([client_stop] if client_stop else []) + [gw.INKLING_CALL_STOP]
    hits = [(raw.index(s), s) for s in stops if s in raw]
    at, which = min(hits)
    reasoning, content = engine_view(raw[:at] + EOS)
    return reasoning, content, toks(raw[:at + len(which)]), which


def stopped_completion(raw, **kw):
    reasoning, content, ids, which = engine_stopped(raw, **kw)
    return {"id": "c", "object": "chat.completion", "model": "inkling-2-small", "choices": [{"index": 0, "message": {
        "role": "assistant", "content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": len(ids), "prompt_tokens_details": {"cached_tokens": 900}},
        "stop_sequence": which, "tensorfold": {"engine": "inkling2", "token_ids": ids}}


def stopped_chunks(raw, pieces=6):
    reasoning, content, ids, which = engine_stopped(raw)
    env = {"id": "c", "object": "chat.completion.chunk", "model": "inkling-2-small"}
    out = [{**env, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": reasoning}}]}] if reasoning else []
    step = max(1, len(content) // pieces)
    out += [{**env, "choices": [{"index": 0, "delta": {"content": content[i:i + step]}}]} for i in range(0, len(content), step)]
    out.append({**env, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "stop_sequence": which,
                "tensorfold": {"engine": "inkling2", "token_ids": ids}})
    out.append({**env, "choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": len(ids),
                                                "prompt_tokens_details": {"cached_tokens": 900}}})
    return out


TWO_CALLS = model_reply("Read both.", "Reading.", [("exec_command", canonical("exec_command", {"cmd": "cat a"})),
                                                   ("exec_command", canonical("exec_command", {"cmd": "cat b"}))])


class StopAfterFirstCall(Fresh):
    """parallel_tool_calls false (GATEWAY_INKLING_PARALLEL=stop, the default): the engine ends the reply where its first
    call message ends, so its state holds the one call the client gets and the next turn renders back to it."""

    def setUp(self):
        super().setUp()
        quiet = patch.object(gw, "inkling_replay_check")
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_the_engine_gets_the_stop_only_for_one_call_with_tools(self):
        self.reply = lambda body: Reply(completion(model_reply("R", "Hi.", [])))
        user = [{"role": "user", "content": "hi"}]
        cases = [("/v1/chat/completions", {"messages": user, "tools": CHAT_TOOLS, "parallel_tool_calls": False,
                                           "stop": "END"}, ["END", gw.INKLING_CALL_STOP]),
                 ("/responses", {"input": "hi", "tools": RESPONSES_TOOLS, "parallel_tool_calls": False},
                  [gw.INKLING_CALL_STOP]),
                 ("/v1/messages", {"max_tokens": 64, "messages": user, "tools": [{"name": "exec_command",
                  "input_schema": {"type": "object"}}], "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}},
                  [gw.INKLING_CALL_STOP]),
                 ("/v1/chat/completions", {"messages": user, "tools": CHAT_TOOLS}, None),          # parallel allowed
                 ("/v1/chat/completions", {"messages": user, "parallel_tool_calls": False}, None)]  # no tools
        for path, body, stop in cases:
            with self.subTest(path=path, stop=stop):
                self.post(path, {"model": "inkling", **body})
                self.assertEqual(self.sent().get("stop"), stop if stop else body.get("stop"))
        for policy in ("all", "drop"):
            with self.subTest(policy=policy), patch.object(gw, "INKLING_PARALLEL", policy):
                self.post("/responses", {"model": "inkling", "input": "hi", "tools": RESPONSES_TOOLS,
                                         "parallel_tool_calls": False})
                self.assertNotIn("stop", self.sent())

    @unittest.skipUnless(CAN_RENDER, "needs jinja2 and the deployed chat template")
    def test_the_next_turn_starts_with_the_engines_state(self):
        """The engine state after the stop (the prompt, then the reply's tokens with or without the one that completed
        the stop, as the round fell) starts the next turn's prompt, rendered by the deployed template."""
        user = {"role": "user", "content": "read a and b"}
        for path in ("/v1/chat/completions", "/responses"):
            with self.subTest(path=path):
                self.reply = lambda body: Reply(stopped_completion(TWO_CALLS))
                if path == "/responses":
                    _, reply = self.post(path, {"model": "inkling", "input": [user], "tools": RESPONSES_TOOLS,
                                                "parallel_tool_calls": False})
                    calls = [i for i in reply["output"] if i["type"] == "function_call"]
                    said = [{"role": "assistant", "content": i["content"][0]["text"]} for i in reply["output"]
                            if i["type"] == "message"]               # Codex: the message back, its reasoning not
                    back = [user] + said + [{"type": "function_call", "call_id": c["call_id"], "name": c["name"],
                                             "arguments": c["arguments"]} for c in calls] + [
                        {"type": "function_call_output", "call_id": c["call_id"], "output": "A"} for c in calls]
                    self.assertNotIn("stop_sequence", json.dumps(reply))
                    self.post(path, {"model": "inkling", "input": back, "tools": RESPONSES_TOOLS,
                                     "parallel_tool_calls": False})
                else:
                    _, reply = self.post(path, {"model": "inkling", "messages": [user], "tools": CHAT_TOOLS,
                                                "parallel_tool_calls": False})
                    self.assertNotIn("stop_sequence", reply)
                    msg = reply["choices"][0]["message"]
                    self.assertEqual(reply["choices"][0]["finish_reason"], "tool_calls")
                    calls = msg["tool_calls"]
                    back = [user, {"role": "assistant", "content": msg["content"], "tool_calls": calls}] + [
                        {"role": "tool", "tool_call_id": c["id"], "content": "A"} for c in calls]
                    self.post(path, {"model": "inkling", "messages": back, "tools": CHAT_TOOLS,
                                     "parallel_tool_calls": False})
                self.assertEqual(len(calls), 1)
                ids = engine_stopped(TWO_CALLS)[2]
                prompt = toks(render([user]))
                nxt = toks(render([m for m in self.sent()["messages"]]))
                for state in (prompt + ids, prompt + ids[:-1]):
                    self.assertEqual(nxt[:len(state)], state)

    @unittest.skipUnless(CAN_RENDER, "needs jinja2 and the deployed chat template")
    def test_a_streamed_reply_the_stop_ended(self):
        ids = engine_stopped(TWO_CALLS)[2]
        self.reply = lambda body: Reply(sse(stopped_chunks(TWO_CALLS)), content_type="text/event-stream")
        user = {"role": "user", "content": "read a and b"}
        _, data = self.post("/v1/chat/completions", {"model": "inkling", "stream": True, "messages": [user],
                                                     "tools": CHAT_TOOLS, "parallel_tool_calls": False}, raw=True)
        chunks = [c for _, c in events(data) if c != "[DONE]"]
        self.assertFalse(any("stop_sequence" in c for c in chunks))
        calls = [t for c in chunks for ch in c["choices"] for t in ch["delta"].get("tool_calls") or []]
        self.assertEqual([json.loads(c["function"]["arguments"]) for c in calls], [{"cmd": "cat a"}])
        self.assertIn("tool_calls", [ch.get("finish_reason") for c in chunks for ch in c["choices"]])
        record = gw.INKLING_MEMORY[f"call:{calls[0]['id']}"][0]
        self.assertEqual(toks(turn_of(render([gw.inkling_rerender(record)])))[:len(ids)], ids)
        _, data = self.post("/responses", {"model": "inkling", "stream": True, "input": [user],
                                           "tools": RESPONSES_TOOLS, "parallel_tool_calls": False}, raw=True)
        done = [e for name, e in events(data) if name == "response.completed"][0]["response"]
        self.assertEqual([i["arguments"] for i in done["output"] if i["type"] == "function_call"], ['{"cmd": "cat a"}'])
        self.assertNotIn("stop_sequence", json.dumps(done))

    def test_a_message_that_ends_in_braces_ends_the_reply(self):
        thinking = model_reply('Use {"a":{"b":1}}', None, [("exec_command", canonical("exec_command", {"cmd": "ls"}))])
        text = model_reply("R", 'Config: {"a":{"b":1}}', [("exec_command", canonical("exec_command", {"cmd": "ls"}))])
        kept, remember = [], gw.remember_turn
        spy = patch.object(gw, "remember_turn", side_effect=lambda turn: (kept.append(turn), remember(turn)))
        spy.start()
        self.addCleanup(spy.stop)
        for raw, part, note in ((thinking, "reasoning_content", "fired in the thinking"),
                                (text, "content", "ended a message that is no call")):
            with self.subTest(part=part):
                self.reply = lambda body: Reply(stopped_completion(raw))
                _, reply = self.post("/v1/chat/completions", {"model": "inkling", "tools": CHAT_TOOLS,
                                                              "parallel_tool_calls": False,
                                                              "messages": [{"role": "user", "content": part}]})
                msg = reply["choices"][0]["message"]
                self.assertTrue(msg[part].endswith('{"a":{"b":1}}'), msg)
                self.assertNotIn("tool_calls", msg)
                self.assertTrue(any(note in n for n in kept[-1]["notes"]), kept[-1]["notes"])

    def test_the_clients_own_stop_stays_its_own(self):
        raw = model_reply("R", "One. END two.", [])
        self.reply = lambda body: Reply(stopped_completion(raw, client_stop="END"))
        _, reply = self.post("/v1/chat/completions", {"model": "inkling", "tools": CHAT_TOOLS, "stop": ["END"],
                                                      "parallel_tool_calls": False,
                                                      "messages": [{"role": "user", "content": "go"}]})
        self.assertEqual((reply["stop_sequence"], reply["choices"][0]["message"]["content"]), ("END", "One. "))


class MemoryFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "state", "inkling-memory.jsonl")
        fresh = patch.multiple(gw, INKLING_MEMORY=OrderedDict(), INKLING_NAMES=OrderedDict(), INKLING_MEMORY_SIZE=0,
                               INKLING_STORE={"path": None, "handle": None, "bytes": 0})
        fresh.start()
        self.addCleanup(fresh.stop)
        self.addCleanup(lambda: gw.INKLING_STORE["handle"] and gw.INKLING_STORE["handle"].close())

    def reply(self, i):
        turn = gw.InklingTurn({"role": "user", "content": f"q{i}"})
        turn.reasoning.append(f"thought {i}")
        turn.text.append(f"text {i}")
        turn.call(f"call_{i}", {"name": "exec_command", "written": "exec_command", "arguments": '{"cmd": "ls"}'})
        return turn.remember()

    def restart(self, crash=False):
        """What the gateway keeps across a restart (or a crash: no rewrite on the way out): the file, read back into a
        fresh memory."""
        out = io.StringIO()
        with redirect_stdout(out):
            if crash:
                gw.INKLING_STORE["handle"].close()
            else:
                gw.close_inkling_memory()
        gw.INKLING_MEMORY.clear()
        gw.INKLING_STORE.update(path=None, handle=None, bytes=0)
        with redirect_stdout(out):
            gw.load_inkling_memory(self.path)
        return out.getvalue()

    def test_a_restart_keeps_the_memory(self):
        with redirect_stdout(io.StringIO()):
            gw.load_inkling_memory(self.path)
        for i in range(3):
            self.reply(i)
        self.assertEqual(oct(os.stat(self.path).st_mode & 0o777), "0o600")
        said = self.restart()
        self.assertIn("inkling memory: 6 entries", said)
        back = gw.recall_reasoning([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "text 1"}])[0]
        self.assertEqual(back[1]["reasoning_content"], "thought 1")
        call = gw.recall_reasoning([{"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_2", "type": "function", "function": {"name": "exec_command", "arguments": "{}"}}]}])[0]
        self.assertEqual(call[0]["reasoning_content"], "thought 2")
        with open(self.path) as f:                             # compact: each reply once, under both its keys
            self.assertEqual([len(json.loads(line)["keys"]) for line in f], [2, 2, 2])

    def test_a_torn_last_line_and_the_bounds(self):
        with redirect_stdout(io.StringIO()):
            gw.load_inkling_memory(self.path)
        for i in range(5):
            self.reply(i)
        gw.INKLING_STORE["handle"].write('{"keys": ["call:x"], "turn": {"reaso')     # a crash mid-write
        gw.INKLING_STORE["handle"].flush()
        with patch.object(gw, "INKLING_MEMORY_MAX", 4):
            said = self.restart(crash=True)
        self.assertIn("5 replies read, 1 unreadable lines skipped", said)
        self.assertEqual(len(gw.INKLING_MEMORY), 4)            # the newest kept
        self.assertEqual({k for k in gw.INKLING_MEMORY if k.startswith("call:")}, {"call:call_3", "call:call_4"})

    def test_a_file_that_cannot_be_written_leaves_ram_only(self):
        blocker = os.path.join(self.dir.name, "file")
        Path(blocker).write_text("x")
        out = io.StringIO()
        with redirect_stdout(out):
            gw.load_inkling_memory(os.path.join(blocker, "inkling-memory.jsonl"))   # a directory under a file
            self.reply(0)
        self.assertIn("the memory stays in RAM only", out.getvalue())
        self.assertIsNone(gw.INKLING_STORE["path"])
        self.assertIn("call:call_0", gw.INKLING_MEMORY)


if __name__ == "__main__":
    unittest.main()
