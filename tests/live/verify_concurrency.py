#!/usr/bin/env python3
"""verify_concurrency.py [BASE]: two simultaneous requests through the gateway (default http://127.0.0.1:8080), after
`scripts/serve.sh start` with --parallel 1 (config/inkling.env PARALLEL): both must succeed, and the engine's done line for
the second must show queued > 0 (the lane host held it until the first ended), never "one stream at a time".
Standard library only; run it on the head node (it reads logs/r0.log and waits for the engine to be idle on :8888)."""
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
LOG = Path(__file__).resolve().parents[2] / "logs" / "r0.log"
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def idle():
    for _ in range(300):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8888/v1/load", timeout=5) as r:
                d = json.load(r)
            if d["running"] == 0 and d["waiting"] == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def call(name, path, body, out):
    t = time.monotonic()
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out[name] = (r.status, r.read().decode(errors="replace"), time.monotonic() - t)
    except urllib.error.HTTPError as e:
        out[name] = (e.code, e.read().decode(errors="replace"), time.monotonic() - t)
    except Exception as e:                        # noqa: BLE001
        out[name] = (0, repr(e), time.monotonic() - t)


check("the engine is idle before the test", idle())
start = LOG.stat().st_size if LOG.exists() else 0
first = {"model": "inkling-2-small", "max_tokens": 512, "stream": True, "messages": [
    {"role": "user", "content": "Write a 12-line poem about the sea. Number the lines."}]}
second = {"model": "inkling-2-small", "max_tokens": 128, "messages": [{"role": "user", "content": "Say READY and nothing else."}]}
out = {}
a = threading.Thread(target=call, args=("first (stream)", "/v1/chat/completions", first, out))
b = threading.Thread(target=call, args=("second (side request)", "/v1/messages", second, out))
a.start()
time.sleep(0.5)                                   # the side request arrives while the first one runs
b.start()
a.join()
b.join()
for name in ("first (stream)", "second (side request)"):
    status, text, took = out[name]
    check(f"{name}: 200, no 'one stream at a time'", status == 200 and "one stream at a time" not in text,
          f"status {status}, {took:.1f} s, {text[:100]!r}")
time.sleep(1)
lines = LOG.read_text(errors="replace")[start:].splitlines() if LOG.exists() else []
done = [(float(m.group(1)), line) for line in lines if " done " in line and (m := re.search(r"\bqueued=([0-9.]+)", line))]
for queued, line in done:
    print("   " + line[:200])
check("the engine logged both requests' done lines", len(done) >= 2, f"{len(done)} done lines in {LOG}")
check("the second waited in the engine's queue (queued > 0)", len(done) >= 2 and done[1][0] > 0,
      f"queued = {[q for q, _ in done]}")
check("no NoFreeSlot refusal in the engine log", not any("one stream at a time" in line for line in lines))
print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
sys.exit(1 if failures else 0)
