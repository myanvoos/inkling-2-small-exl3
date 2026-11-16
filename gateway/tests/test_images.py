"""Image history must stay stable without replacing edited tool results."""
import copy
import io
import json
import sys
import threading
import unittest
import urllib.request
from collections import OrderedDict
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inkling_gateway as gw


def image(name):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": name}}


def body(count=1):
    messages = [{"role": "user", "content": "Inspect these images."}]
    for i in range(count):
        messages.extend([
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"read-{i}", "name": "read",
                                                "input": {"path": f"{i}.png"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"read-{i}", "content": [
                {"type": "text", "text": f"Read image file {i}.png"}, image(f"pixels-{i}")]}]},
        ])
    return {"system": "Image test", "max_tokens": 2, "messages": messages}


def result(request, i=0):
    return request["messages"][2 + 2 * i]["content"][0]


def prune(request, *indices):
    request = copy.deepcopy(request)
    for i in indices:
        block = result(request, i)
        block["content"] = [p for p in block["content"] if p["type"] != "image"]
    return request


class Images(unittest.TestCase):
    def setUp(self):
        flags = patch.multiple(gw, RESTORE_IMAGES=True, RESTORE_MAX=64 * 2**20,
                               RESTORE_MAX_IMAGES=50, ORIGINALS=OrderedDict(), ORIGINALS_BYTES=0,
                               STABLE_TIME=False, TIME_NOTES=OrderedDict(),
                               ARGS=SimpleNamespace(served_model="test", max_tokens=16))
        flags.start()
        self.addCleanup(flags.stop)

    def test_silent_tool_image_removal(self):
        original = body()
        gw.restore_images(original)
        incoming = prune(original, 0)
        before = copy.deepcopy(incoming)
        restored, count = gw.restore_images(incoming)
        self.assertEqual((restored, count), (original, 1))
        self.assertEqual(incoming, before)

    def test_rolling_five_image_window_keeps_converted_prefix(self):
        previous = None
        for n in range(5, 10):
            original = body(n)
            incoming = prune(original, *range(n - 5))
            restored, count = gw.restore_images(incoming)
            self.assertEqual((restored, count), (original, n - 5))
            converted = gw.to_openai(restored)[0]["messages"]
            if previous is not None:
                self.assertEqual(converted[:len(previous)], previous)
            previous = converted

    def test_partial_removal_keeps_remaining_images_in_order(self):
        original = body()
        result(original)["content"].extend([{"type": "text", "text": "Second view"}, image("second")])
        gw.restore_images(original)
        incoming = copy.deepcopy(original)
        del result(incoming)["content"][1]
        self.assertEqual(gw.restore_images(incoming), (original, 1))

    def test_multiple_tool_results_in_one_message(self):
        original = body()
        extra = copy.deepcopy(result(original))
        extra["tool_use_id"] = "other-call"
        original["messages"][-1]["content"].extend([extra, {"type": "text", "text": "Compare both."}])
        gw.restore_images(original)
        incoming = copy.deepcopy(original)
        for block in incoming["messages"][-1]["content"][:2]:
            block["content"].pop()
        self.assertEqual(gw.restore_images(incoming), (original, 1))

    def test_edited_text_id_flags_and_user_content_are_not_restored(self):
        original = body()
        gw.restore_images(original)
        variants = []
        changed = prune(original, 0)
        result(changed)["content"][0]["text"] = "Edited result"
        variants.append(changed)
        changed = prune(original, 0)
        result(changed)["tool_use_id"] = "different-call"
        variants.append(changed)
        changed = prune(original, 0)
        result(changed)["is_error"] = True
        variants.append(changed)
        changed = prune(original, 0)
        changed["messages"][-1]["content"].append({"type": "text", "text": "New instruction"})
        variants.append(changed)
        for incoming in variants:
            with self.subTest(incoming=incoming):
                self.assertEqual(gw.restore_images(incoming), (incoming, 0))

    def test_new_replaced_and_reordered_images_are_not_overwritten(self):
        original = body()
        result(original)["content"].append(image("second"))
        variants = []
        changed = copy.deepcopy(original)
        result(changed)["content"][1] = image("replacement")
        variants.append(changed)
        changed = copy.deepcopy(original)
        result(changed)["content"].append(image("extra"))
        variants.append(changed)
        changed = copy.deepcopy(original)
        result(changed)["content"][1:] = list(reversed(result(changed)["content"][1:]))
        variants.append(changed)
        for incoming in variants:
            with self.subTest(incoming=incoming):
                gw.ORIGINALS.clear()
                gw.ORIGINALS_BYTES = 0
                gw.restore_images(original)
                self.assertEqual(gw.restore_images(incoming), (incoming, 0))

    def test_changed_images_replace_stale_memory(self):
        original = body()
        gw.restore_images(original)
        updated = copy.deepcopy(original)
        result(updated)["content"][1] = image("updated-pixels")
        self.assertEqual(gw.restore_images(updated), (updated, 0))
        self.assertEqual(gw.restore_images(prune(updated, 0)), (updated, 1))
        self.assertEqual(gw.ORIGINALS_BYTES, sum(size for _, size in gw.ORIGINALS.values()))

    def test_silent_removal_of_direct_user_image_is_not_undone(self):
        original = {"system": "test", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Look"}, image("direct")]}]}
        gw.restore_images(original)
        incoming = copy.deepcopy(original)
        incoming["messages"][0]["content"].pop()
        self.assertEqual(gw.restore_images(incoming), (incoming, 0))

    def test_different_system_or_history_does_not_borrow_images(self):
        original = body()
        gw.restore_images(original)
        incoming = prune(original, 0)
        incoming["system"] = "Another conversation"
        self.assertEqual(gw.restore_images(incoming), (incoming, 0))
        incoming = prune(original, 0)
        incoming["messages"][0]["content"] = "A different branch"
        self.assertEqual(gw.restore_images(incoming), (incoming, 0))

    def test_marked_omission_still_restores(self):
        original = body()
        gw.restore_images(original)
        incoming = prune(original, 0)
        result(incoming)["content"].append({"type": "text", "text": f"[{gw.RESTORE_MARK}; old image]"})
        self.assertEqual(gw.restore_images(incoming), (original, 1))

    def test_direct_user_image_with_marker_still_restores(self):
        original = {"messages": [{"role": "user", "content": [image("direct")]}]}
        gw.restore_images(original)
        incoming = {"messages": [{"role": "user", "content": [{"type": "text", "text": gw.RESTORE_MARK}]}]}
        self.assertEqual(gw.restore_images(incoming), (original, 1))

    def test_count_tokens_reads_without_learning_images(self):
        original = body()
        self.assertEqual(gw.restore_images(original, remember=False), (original, 0))
        self.assertFalse(gw.ORIGINALS)
        gw.restore_images(original)
        before = copy.deepcopy(gw.ORIGINALS)
        size = gw.ORIGINALS_BYTES
        restored, count = gw.restore_images(prune(body(2), 0), remember=False)
        self.assertEqual((restored, count), (body(2), 1))
        self.assertEqual(gw.ORIGINALS, before)
        self.assertEqual(gw.ORIGINALS_BYTES, size)
        counted = gw.to_openai(restored, count_only=True)[0]["messages"]
        self.assertEqual(counted, gw.to_openai(restored)[0]["messages"])

    def test_clock_changes_do_not_break_image_or_time_history(self):
        gw.STABLE_TIME = True
        original = body()
        original["system"] = "Now 2026-11-02 17:00:00 +13:00 [Pacific/Auckland]"
        first = gw.to_openai(gw.restore_images(original)[0])[0]["messages"]
        incoming = prune(original, 0)
        incoming["system"] = "Now 2026-11-02 17:01:00 +13:00 [Pacific/Auckland]"
        restored, count = gw.restore_images(incoming)
        self.assertEqual(count, 1)
        self.assertEqual(gw.to_openai(restored)[0]["messages"], first)

    def test_disabled_restoration_is_a_noop(self):
        original = body()
        gw.restore_images(original)
        gw.RESTORE_IMAGES = False
        incoming = prune(original, 0)
        restored, count = gw.restore_images(incoming)
        self.assertIs(restored, incoming)
        self.assertEqual(count, 0)

    def test_memory_limit_evicts_old_messages(self):
        original = body()
        gw.RESTORE_MAX = len(json.dumps(original["messages"][-1])) + 1
        gw.restore_images(original)
        newer = body()
        newer["system"] = "Another conversation"
        gw.restore_images(newer)
        self.assertEqual(len(gw.ORIGINALS), 1)
        self.assertLessEqual(gw.ORIGINALS_BYTES, gw.RESTORE_MAX)
        self.assertEqual(gw.restore_images(prune(original, 0))[1], 0)
        self.assertEqual(gw.restore_images(prune(newer, 0)), (newer, 1))

    def test_oversized_image_message_is_not_kept(self):
        gw.RESTORE_MAX = 1
        original = body()
        gw.restore_images(original)
        self.assertFalse(gw.ORIGINALS)
        self.assertEqual(gw.ORIGINALS_BYTES, 0)
        self.assertEqual(gw.restore_images(prune(original, 0))[1], 0)

    def test_restoration_never_pushes_request_over_image_limit(self):
        gw.RESTORE_MAX_IMAGES = 5
        gw.restore_images(body(5))
        incoming = prune(body(6), 0)
        self.assertEqual(gw.restore_images(incoming), (incoming, 0))

    def test_image_limit_allows_room_for_restored_images(self):
        gw.RESTORE_MAX_IMAGES = 6
        gw.restore_images(body(5))
        self.assertEqual(gw.restore_images(prune(body(6), 0)), (body(6), 1))

    def test_image_limit_counts_new_direct_images(self):
        gw.RESTORE_MAX_IMAGES = 1
        original = body()
        gw.restore_images(original)
        incoming = prune(original, 0)
        incoming["messages"].append({"role": "user", "content": [image("new-direct")]})
        self.assertEqual(gw.restore_images(incoming), (incoming, 0))

    def test_image_limit_applies_across_multiple_restorations(self):
        gw.restore_images(body(3))
        gw.RESTORE_MAX_IMAGES = 4
        restored, count = gw.restore_images(prune(body(5), 0, 1, 2))
        self.assertEqual(count, 2)
        self.assertEqual(sum(gw._image_count(m) for m in restored["messages"]), 4)
        self.assertEqual(result(restored, 2), result(prune(body(5), 2), 2))

    def test_marked_omissions_also_obey_image_limit(self):
        gw.restore_images(body())
        gw.RESTORE_MAX_IMAGES = 1
        incoming = prune(body(2), 0)
        result(incoming)["content"].append({"type": "text", "text": gw.RESTORE_MARK})
        self.assertEqual(gw.restore_images(incoming), (incoming, 0))

    def test_oversized_replacement_drops_stale_image_memory(self):
        original = body()
        gw.restore_images(original)
        gw.RESTORE_MAX = gw.ORIGINALS_BYTES
        updated = copy.deepcopy(original)
        result(updated)["content"][1] = image("large-update" * gw.RESTORE_MAX)
        self.assertEqual(gw.restore_images(updated), (updated, 0))
        self.assertFalse(gw.ORIGINALS)
        self.assertEqual(gw.ORIGINALS_BYTES, 0)
        self.assertEqual(gw.restore_images(prune(updated, 0))[1], 0)

    def test_rewritten_tool_result_structure_is_not_restored(self):
        original = body()
        gw.restore_images(original)
        incoming = prune(original, 0)
        result(incoming)["content"] = result(original)["content"][0]["text"]
        self.assertEqual(gw.restore_images(incoming), (incoming, 0))

    def test_request_log_counts_direct_and_tool_result_images(self):
        request = body(2)
        request["messages"][0]["content"] = [image("direct"), {"type": "text", "text": "Look"}]
        self.assertIn("images 3 ", gw.request_shape(request, 100))

    def test_http_routes_restore_before_forwarding(self):
        seen = []

        def upstream(path, payload, **kwargs):
            seen.append((path, payload))
            if path == "/tokenize":
                return io.BytesIO(b'{"count": 123}')
            if payload.get("stream"):
                chunk = {"id": "test", "model": "test", "choices": [
                    {"delta": {"content": "OK"}, "finish_reason": "stop"}]}
                return io.BytesIO(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
            reply = {"id": "test", "model": "test", "choices": [
                {"message": {"content": "OK"}, "finish_reason": "stop"}]}
            return io.BytesIO(json.dumps(reply).encode())

        with (patch.object(gw, "upstream", upstream), patch.object(gw, "incoming"),
              patch.object(gw, "capture"), patch.object(gw.Handler, "_log") as logs,
              ThreadingHTTPServer(("127.0.0.1", 0), gw.Handler) as server):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def post(path, payload):
                request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}",
                                                 data=json.dumps(payload).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    return response.read()

            try:
                post("/v1/messages", body(5))
                previous = seen[-1][1]["messages"]
                incoming = prune(body(6), 0)
                self.assertEqual(json.loads(post("/v1/messages/count_tokens", incoming))["input_tokens"], 123)
                counted = seen[-1][1]["messages"]
                self.assertIn(b"event: message_stop", post("/v1/messages", {**incoming, "stream": True}))
                generated = seen[-1][1]["messages"]
                self.assertEqual(generated, gw.to_openai(body(6))[0]["messages"])
                self.assertEqual(generated, counted)
                self.assertEqual(generated[:len(previous)], previous)
                self.assertTrue(any("restored 1 image message" in str(call) for call in logs.call_args_list))
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_first_image_keeps_text_history_as_strings(self):
        # The engine's image path rejoins every list-content message with "\n\n" (its text path with ""),
        # so text-only history must arrive as strings or the first image changes the whole prefix.
        gw.STABLE_TIME = True
        history = [
            {"role": "user", "content": [{"type": "text", "text": "<system-reminder>notes</system-reminder>"},
                                         {"type": "text", "text": "Take a screenshot."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Taking"}, {"type": "text", "text": " it."},
                                              {"type": "tool_use", "id": "shot", "name": "read", "input": {}}]},
        ]
        system = "Now 2026-11-02 17:00:00 +13:00 [Pacific/Auckland]"
        text = {"system": system, "max_tokens": 2, "messages": history + [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "shot", "content": "no image"}]}]}
        first = gw.to_openai(text)[0]["messages"]
        with_image = copy.deepcopy(text)
        with_image["messages"][-1]["content"][0]["content"] = [{"type": "text", "text": "shot"}, image("pixels")]
        later = gw.to_openai(with_image)[0]["messages"]
        self.assertEqual(later[:3], first[:3])
        self.assertTrue(first[1]["content"].startswith("<system-reminder>notes</system-reminder>Take a screenshot."))
        self.assertEqual(first[2]["content"], "Taking it.")
        for message in later:
            parts = message.get("content")
            if isinstance(parts, list):
                self.assertTrue(any(p.get("type") == "image_url" for p in parts), message)

    def test_missing_history_stays_unchanged(self):
        incoming = prune(body(), 0)
        restored, count = gw.restore_images(incoming)
        self.assertIs(restored, incoming)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
