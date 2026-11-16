import io
import json
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


class Models(unittest.TestCase):
    def setUp(self):
        flags = patch.object(gw, "ARGS", SimpleNamespace(
            served_model="aperture", alias_model=["test-model"], owned_by="tensorfold",
            root="upstream-checkpoint", context=1048576, max_tokens=65536,
        ))
        flags.start()
        self.addCleanup(flags.stop)
        upstream = patch.object(gw, "upstream", side_effect=self.upstream_reply)
        self.upstream = upstream.start()
        self.addCleanup(upstream.stop)
        logs = patch.object(gw.Handler, "_log")
        logs.start()
        self.addCleanup(logs.stop)
        incoming = patch.object(gw, "incoming")
        incoming.start()
        self.addCleanup(incoming.stop)
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
            body = {"id": "chatcmpl-test", "object": "chat.completion", "model": "aperture",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "READY"},
                                 "finish_reason": "stop"}]}
        elif path == "/v1/chat/completions":
            self.last_request = payload
            body = {"id": "chatcmpl-test", "object": "chat.completion", "model": "aperture",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "READY"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        else:
            body = {"object": "list", "data": [{"id": "upstream-checkpoint"}]}
        reply = io.BytesIO(json.dumps(body).encode())
        reply.headers = {"Content-Type": "application/json"}
        reply.status = 200
        return reply

    def get(self, path, headers=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}{path}", headers=headers or {},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.load(response)

    def post(self, path, body, headers=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.load(response)

    def test_both_discovery_routes_return_the_gateway_cards(self):
        for path in ("/models", "/models/", "/models?limit=10", "/v1/models", "/v1/models/"):
            with self.subTest(path=path):
                self.upstream.reset_mock()
                card = self.get(path)
                self.assertEqual(card["object"], "list")
                self.assertEqual([model["id"] for model in card["data"]], ["aperture", "test-model"])
                self.assertEqual([model["display_name"] for model in card["data"]], ["aperture", "test-model"])
                model = card["data"][0]
                self.assertEqual(model["root"], "upstream-checkpoint")
                self.assertEqual(model["max_model_len"], 1048576)
                self.assertEqual(model["max_tokens"], 65536)
                self.upstream.assert_called_once_with("/v1/models", method="GET", timeout=10)

    def test_alias_card_matches_the_served_model(self):
        served, alias = self.get("/v1/models")["data"]
        for field in ("root", "max_model_len", "max_input_tokens", "max_tokens", "owned_by",
                      "input_modalities", "output_modalities", "supports_vision"):
            with self.subTest(field=field):
                self.assertEqual(alias[field], served[field])

    def test_without_aliases_only_the_served_model_is_listed(self):
        with patch.object(gw, "ARGS", SimpleNamespace(
            served_model="aperture", alias_model=[], owned_by="tensorfold",
            root="upstream-checkpoint", context=1048576, max_tokens=65536,
        )):
            self.assertEqual([model["id"] for model in gw.models_card()["data"]], ["aperture"])

    def test_anthropic_alias_asks_for_the_served_model_and_echoes_the_alias(self):
        reply = self.post("/v1/messages", {
            "model": "test-model", "max_tokens": 8,
            "messages": [{"role": "user", "content": "Say READY."}],
        })
        self.assertEqual(self.last_request["model"], "aperture")
        self.assertEqual(reply["model"], "test-model")
        self.assertEqual(reply["type"], "message")

    def test_passthrough_keeps_the_alias_unchanged(self):
        self.post("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Say READY."}],
        })
        self.assertEqual(self.last_request["model"], "test-model")

    def test_discovery_fails_when_the_engine_is_down(self):
        self.upstream.side_effect = urllib.error.URLError("engine stopped")
        for path in ("/models", "/v1/models"):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as raised:
                self.get(path)
            with raised.exception as response:
                self.assertEqual(response.code, 502)
                self.assertIn("engine stopped", json.load(response)["error"]["message"])

    def test_resume_connector_cannot_discover_models(self):
        for path in ("/models", "/v1/models"):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as raised:
                self.get(path, {"X-Tailnet-Resume-Proxy": "1"})
            with raised.exception as response:
                self.assertEqual(response.code, 403)
        self.upstream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
