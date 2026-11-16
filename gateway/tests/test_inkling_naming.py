"""The Inkling recipe's names: one card (id inkling-2-small, display name Inkling-2-Small, root the combined model
directory), and the afternoon's id `inkling` still answered without a card (--hidden-alias)."""
import json
import sys
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_inkling import OTHER_ARGS, Http, Reply, gw

NAMING_ARGS = SimpleNamespace(family="inkling2", served_model="inkling-2-small", alias_model=[],
                              hidden_alias=["inkling"], display_name="Inkling-2-Small", owned_by="tensorfold",
                              root="thinkingmachines/Inkling-2-Small-EXL3-Native", context=500000, max_tokens=65536)


class Naming(Http):
    args = NAMING_ARGS

    def setUp(self):
        super().setUp()
        # the engine as started now (--name inkling-2-small): its replies name its own id
        self.reply = lambda body: Reply({"id": "c", "object": "chat.completion", "model": "inkling-2-small", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "READY", "reasoning_content": "ok"},
             "finish_reason": "stop"}]})

    def card(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.server.server_port}{path}", timeout=5) as response:
            return json.load(response)

    def test_one_card_with_the_display_name_and_root(self):
        for path in ("/v1/models", "/models"):
            with self.subTest(path=path):
                data = self.card(path)["data"]
                self.assertEqual([(m["id"], m["display_name"], m["root"]) for m in data],
                                 [("inkling-2-small", "Inkling-2-Small", "thinkingmachines/Inkling-2-Small-EXL3-Native")])
                self.assertEqual((data[0]["input_modalities"], data[0]["max_model_len"]), (["text"], 500000))

    def test_the_hidden_alias_is_answered_on_messages_and_count_tokens(self):
        status, reply = self.post("/v1/messages", {"model": "inkling", "max_tokens": 16,
                                                   "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((status, reply["model"]), (200, "inkling"))              # the client's own name comes back
        self.assertEqual(self.last("/v1/chat/completions")["model"], "inkling-2-small")
        status, reply = self.post("/v1/messages", {"model": "inkling-2-small", "max_tokens": 16,
                                                   "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((status, reply["model"]), (200, "inkling-2-small"))
        status, _ = self.post("/messages/count_tokens", {"model": "inkling", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((status, self.last("/tokenize")["model"]), (200, "inkling-2-small"))

    def test_the_hidden_alias_is_answered_on_responses_and_chat(self):
        status, reply = self.post("/v1/responses", {"model": "inkling", "input": "hi"})
        self.assertEqual((status, reply["model"]), (200, "inkling"))
        self.assertEqual(self.last("/v1/chat/completions")["model"], "inkling-2-small")
        status, reply = self.post("/chat/completions", {"model": "inkling", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((status, reply["choices"][0]["message"]["content"]), (200, "READY"))

    def test_an_unknown_id_is_relayed_as_before(self):
        self.post("/v1/messages", {"model": "something-else", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(self.last("/v1/chat/completions")["model"], "something-else")


class OtherRecipesNaming(Http):
    """Without the new options a card's display_name stays its id and nothing is hidden."""
    args = OTHER_ARGS

    def test_display_name_is_the_id(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.server.server_port}/v1/models", timeout=5) as response:
            data = json.load(response)["data"]
        self.assertEqual([(m["id"], m["display_name"]) for m in data], [("test-model", "test-model")])
        self.assertEqual(gw.hidden_aliases(), ())


if __name__ == "__main__":
    unittest.main()
