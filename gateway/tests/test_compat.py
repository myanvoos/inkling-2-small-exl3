"""OpenAI Responses/chat compatibility belongs at the gateway boundary."""
import io
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inkling_gateway as gw


class Reply(io.BytesIO):
    def __init__(self, body, *, status=200, content_type="application/json"):
        super().__init__(json.dumps(body).encode() if isinstance(body, dict) else body)
        self.status = status
        self.code = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def read1(self, size=-1):
        return self.read(size)


class Compatibility(unittest.TestCase):
    def setUp(self):
        self.last_request = None
        self.upstream = patch.object(gw, "upstream", side_effect=self.upstream_reply)
        self.upstream.start()
        self.addCleanup(self.upstream.stop)
        self.flags = patch.multiple(gw, PIN_TEMPERATURE=None, STRICT_REASONING=False,
                                    RESPONSES_REASONING_CONTEXT_DEFAULT="all_turns")
        self.flags.start()
        self.addCleanup(self.flags.stop)
        self.incoming = patch.object(gw, "incoming")
        self.incoming.start()
        self.addCleanup(self.incoming.stop)
        self.logs = patch.object(gw.Handler, "_log")
        self.logs.start()
        self.addCleanup(self.logs.stop)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), gw.Handler)
        self.server.daemon_threads = False
        self.addCleanup(self.server.server_close)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=5)

    def upstream_reply(self, path, payload=None, **kwargs):
        if kwargs.get("data") is not None:
            self.last_request = json.loads(kwargs["data"])
        if path == "/v1/responses":
            body = {
                "id": "resp-test", "object": "response", "status": "completed",
                "reasoning": {"effort": None, "summary": None},
                "output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "step"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        else:
            body = {
                "id": "chat-test", "object": "chat.completion", "model": "aperture",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "READY"},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        return Reply(body)

    def request(self, path, body, headers=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.headers, json.load(response)
        except urllib.error.HTTPError as response:
            return response.code, response.headers, json.load(response)

    def test_encrypted_reasoning_include_is_accepted_and_context_echoed(self):
        status, headers, response = self.request("/v1/responses", {
            "model": "aperture", "input": "Say OK", "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"context": "all_turns"},
        })
        self.assertEqual(status, 200)
        self.assertNotIn("include", self.last_request)
        self.assertEqual(response["reasoning"]["context"], "all_turns")
        self.assertEqual(headers[gw.REASONING_CONTEXT_HEADER], "all_turns")

    def test_unknown_include_values_are_named_without_forwarding(self):
        status, _, response = self.request("/v1/responses", {
            "model": "aperture", "input": "Say OK",
            "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
        })
        self.assertEqual(status, 400)
        self.assertIn("message.output_text.logprobs", response["error"]["message"])
        self.assertIsNone(self.last_request)

    def test_current_turn_filters_replayed_reasoning_but_all_turns_keeps_it(self):
        body = {
            "model": "aperture", "input": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "old"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "old answer"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "next"}]},
            ], "reasoning": {"context": "current_turn"},
        }
        status, _, _ = self.request("/v1/responses", body)
        self.assertEqual(status, 200)
        self.assertFalse(any(item.get("type") == "reasoning" for item in self.last_request["input"]))
        body["reasoning"] = {"context": "all_turns"}
        status, _, _ = self.request("/v1/responses", body)
        self.assertEqual(status, 200)
        self.assertTrue(any(item.get("type") == "reasoning" for item in self.last_request["input"]))

    def test_thinking_disabled_maps_to_zero_reasoning_for_both_routes(self):
        status, _, _ = self.request("/v1/responses", {
            "model": "aperture", "input": "Say OK", "thinking": {"type": "disabled"},
        })
        self.assertEqual(status, 200)
        self.assertEqual(self.last_request["reasoning"]["effort"], "none")
        status, _, _ = self.request("/v1/chat/completions", {
            "model": "aperture", "messages": [{"role": "user", "content": "Say OK"}],
            "thinking": {"type": "disabled"},
        })
        self.assertEqual(status, 200)
        self.assertEqual(self.last_request["reasoning_effort"], "none")
        self.assertFalse(self.last_request["chat_template_kwargs"]["enable_thinking"])

    def test_missing_chat_reasoning_is_visible_and_strict_mode_is_opt_in(self):
        request = {
            "model": "aperture", "messages": [
                {"role": "user", "content": "Call the tool."},
                {"role": "assistant", "content": None, "tool_calls": []},
            ], "tools": [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        }
        status, headers, response = self.request("/v1/chat/completions", request)
        self.assertEqual(status, 200)
        self.assertIn("reasoning_content", headers[gw.REASONING_WARNING_HEADER])
        self.assertIn("warnings", response)
        with patch.object(gw, "STRICT_REASONING", True):
            status, _, response = self.request("/v1/chat/completions", request)
        self.assertEqual(status, 400)
        self.assertIn("reasoning_content", response["error"]["message"])

    def test_response_sse_rewrites_context_only_on_response_events(self):
        rewriter = gw.ResponseSSE("all_turns")
        first = b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"reasoning\":{}}}\n\n"
        second = b"data: [DONE]\n\n"
        output = rewriter.feed(first[:31]) + rewriter.feed(first[31:]) + rewriter.feed(second, final=True)
        self.assertIn(b'"context":"all_turns"', output)
        self.assertIn(b"data: [DONE]", output)


if __name__ == "__main__":
    unittest.main()
