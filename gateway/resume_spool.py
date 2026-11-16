"""Opt-in, process-local resumable Messages rounds; no tool execution or request replay.

Use only behind the gateway's existing private tailnet boundary. UUIDs are bearer
capabilities and must not be logged. Spools are private temporary files, not a
restart journal. A restart/expired GET fails closed; clients must never recreate it.
"""
import hashlib
import json
import os
import socket
import re
import urllib.error
import tempfile
import threading
import time
import uuid

ENABLED = (os.environ.get("GATEWAY_STREAM_RESUME") == "1" or
           os.path.isfile(os.path.join(os.path.dirname(__file__), "stream-resume.enabled")))
MAX_JOBS, MAX_BYTES, MAX_EVENTS = 8, 16 * 1024 * 1024, 65536
RETENTION, MAX_AGE = 1800, 3600
PREFIX = "/v1/messages/resume/"
LOCK = threading.Lock()
JOBS = {}
EXPIRED = {}
MAX_IDS, ID_RETENTION = 4096, 86400


def valid_id(value):
    try:
        parsed = uuid.UUID(value)
        return str(parsed) == value and parsed.version == 4
    except (ValueError, TypeError, AttributeError):
        return False


class Job:
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.file = tempfile.TemporaryFile(mode="w+b")
        self.cv = threading.Condition()
        self.offsets, self.size, self.readers, self.pins = [], 0, 0, 0
        self.created = time.monotonic()
        self.finished = None
        self.cancelled = False
        self.response = None

    def cancel(self):
        with self.cv:
            self.cancelled = True
            self.cv.notify_all()
            response = self.response
        # HTTPResponse.close may wait behind readline; shutting down its private
        # upstream socket interrupts that read without touching any other round.
        if response is not None:
            try:
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass

    def append(self, value, terminal=False):
        data = value.encode("utf-8")
        with self.cv:
            reserve = 0 if terminal else 1024
            if len(self.offsets) >= MAX_EVENTS - (0 if terminal else 1) or self.size + len(data) + 32 > MAX_BYTES - reserve:
                raise ValueError("Resume spool limit exceeded")
            record = f"id: {len(self.offsets) + 1}\n".encode() + data
            self.file.seek(self.size)
            self.file.write(record)
            self.file.flush()
            self.offsets.append((self.size, len(record)))
            self.size += len(record)
            self.cv.notify_all()

    def fail(self, event, message):
        self.append(event("error", {"type": "error", "error": {"type": "api_error", "message": message}}), terminal=True)


def reap_locked():
    now = time.monotonic()
    for key, expires in list(EXPIRED.items()):
        if expires < now:
            del EXPIRED[key]
    for key, job in list(JOBS.items()):
        with job.cv:
            if now - job.created > MAX_AGE:
                job.cancel()
            if job.finished is not None and not job.readers and not job.pins and now - job.finished > RETENTION:
                job.file.close()
                del JOBS[key]
                EXPIRED[key] = now + ID_RETENTION


def janitor():
    while True:
        time.sleep(30)
        with LOCK:
            reap_locked()


if ENABLED:
    threading.Thread(target=janitor, daemon=True, name="resume-expiry").start()


def error_detail(failure):
    kind = "api_error"
    message = "Gateway round interrupted, unavailable, or beyond the resume spool limit. No request was replayed."
    if isinstance(failure, (TimeoutError, socket.timeout)):
        return {"type": "timeout_error", "message": "Upstream model request timed out. No request was replayed."}
    if isinstance(failure, urllib.error.HTTPError):
        status = failure.code
        kind = {400: "invalid_request_error", 413: "invalid_request_error", 429: "rate_limit_error", 502: "overloaded_error", 503: "overloaded_error", 504: "timeout_error", 529: "overloaded_error"}.get(status, kind)
        message = f"Upstream model returned HTTP {status}."
        try:
            raw = failure.read(16385)
            parsed = json.loads(raw) if len(raw) <= 16384 else {}
            detail = parsed.get("error", parsed)
            if isinstance(detail, dict):
                candidate = detail.get("type", "")
                if isinstance(candidate, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", candidate):
                    kind = candidate
                text = detail.get("message", "")
                if isinstance(text, str) and len(text) <= 2048 and "<" not in text and "{" not in text:
                    text = re.sub(r"(?i)\b(?:https?://|www\.)\S+", "[redacted URL]", text)
                    text = re.sub(r"(?i)\b(?:bearer|basic)\s+\S+", "[redacted credential]", text)
                    text = re.sub(r"(?i)\b(?:authorization|api[-_ ]?key|token|password|secret|cookie)\s*[:=]\s*[^\s,;]+", "[redacted credential]", text)
                    text = re.sub(r"\b(?:sk-|ghp_|github_pat_|tskey-)[A-Za-z0-9_-]+", "[redacted credential]", text)
                    text = re.sub(r"\b[A-Za-z0-9_+/=-]{32,}\b", "[redacted value]", text)
                    text = " ".join("".join(c if c.isprintable() else " " for c in text).split())[:512]
                    if text:
                        message += " " + text
        except Exception:
            pass
        finally:
            failure.close()
    return {"type": kind, "message": message}


def produce(job, req, upstream, converter, event):
    try:
        if job.cancelled:
            job.fail(event, "Gateway round cancelled before submission.")
            return
        conv = converter()
        received = 0
        # Only this worker submits the round. Reattachment never calls upstream.
        with upstream("/v1/chat/completions", req, timeout=180) as response:
            # Bound connection/header setup, then allow long reasoning pauses.
            # The independent subscriber heartbeat handles its own idle timeout;
            # cancellation/MAX_AGE shut down only this response socket.
            try:
                response.fp.raw._sock.settimeout(MAX_AGE)
            except AttributeError:
                pass
            with job.cv:
                job.response = response
            while True:
                if job.cancelled:
                    job.fail(event, "Gateway round cancelled or expired.")
                    return
                raw = response.readline(2 * 1024 * 1024 + 1)
                received += len(raw)
                if len(raw) > 2 * 1024 * 1024 or received > 32 * 1024 * 1024:
                    raise ValueError("Upstream stream exceeds size limit")
                if not raw:
                    if conv.finish in ("stop", "length", "tool_calls", "content_filter", "refusal"):
                        for value in conv.finish_events():
                            job.append(value, terminal=value.startswith("event: message_stop\n"))
                        return
                    raise ValueError("Upstream stream ended before completion")
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    for value in conv.finish_events():
                        job.append(value, terminal=value.startswith("event: message_stop\n"))
                    return
                for value in conv.feed(json.loads(data)):
                    job.append(value)
    except Exception as failure:
        # Only bounded structured upstream diagnostics, never raw exceptions.
        try:
            detail = {"type": "api_error", "message": "Gateway round cancelled or expired."} if job.cancelled else error_detail(failure)
            job.append(event("error", {"type": "error", "error": detail}), terminal=True)
        except Exception:
            pass
    finally:
        with job.cv:
            job.response = None
            job.finished = time.monotonic()
            job.cv.notify_all()


def send(handler, job, cursor):
    with job.cv:
        if cursor < 0 or cursor > len(job.offsets):
            return handler._anthropic_error(409, "Resume cursor is outside this round.")
        if job.readers >= 2:
            return handler._anthropic_error(429, "Too many attached readers for this round.")
        job.readers += 1
    try:
        handler.connection.settimeout(20)
        handler._stream_start()
        while True:
            with job.cv:
                if cursor < len(job.offsets):
                    offset, length = job.offsets[cursor]
                    job.file.seek(offset)
                    data = job.file.read(length)
                    cursor += 1
                elif job.finished is not None:
                    break
                else:
                    job.cv.wait(10)
                    data = b": resume keepalive\n\n"
            handler.wfile.write(data)
            handler.wfile.flush()
    except (OSError, TimeoutError):
        pass  # The model worker deliberately survives this subscriber.
    finally:
        with job.cv:
            job.readers -= 1
        handler._log(200, "resumable stream " + getattr(handler, "_resume_note", ""))


def cursor(handler):
    value = handler.headers.get("Last-Event-ID", "0")
    if not value.isascii() or not value.isdigit() or len(value) > 10:
        raise ValueError("Invalid resume cursor")
    return int(value)


def attach(handler, path, cancel=False):
    if not ENABLED:
        return handler._anthropic_error(404, "Stream resume is not enabled.")
    key = path[len(PREFIX):]
    if not valid_id(key):
        return handler._anthropic_error(400, "Invalid round identifier.")
    with LOCK:
        reap_locked()
        job = JOBS.get(key)
        if job is None:
            if cancel:
                # Stop may overtake a POST while its body is still being read or
                # converted. Tombstone the exact capability under the same lock
                # as create(), so that late registration cannot start inference.
                if key not in EXPIRED and len(JOBS) + len(EXPIRED) >= MAX_IDS:
                    return handler._anthropic_error(503, "Resume identifier capacity is full. No round can be started.")
                EXPIRED[key] = time.monotonic() + ID_RETENTION
                return handler._json(202, {"cancelled": True})
            return handler._anthropic_error(410, "Round is unavailable or expired. It was not restarted.")
        if cancel:
            job.cancel()
            return handler._json(202, {"cancelled": True})
        try:
            after = cursor(handler)
        except ValueError:
            return handler._anthropic_error(400, "Invalid resume cursor.")
        # Pin against expiry until send() acquires its reader reference.
        with job.cv:
            job.pins += 1
    try:
        return send(handler, job, after)
    finally:
        with job.cv:
            job.pins -= 1


def create(handler, body, req, upstream, converter, event):
    if not ENABLED:
        return handler._anthropic_error(404, "Stream resume is not enabled.")
    key = handler.headers.get("X-Tailnet-Request-ID", "")
    if not valid_id(key) or not req.get("stream"):
        return handler._anthropic_error(400, "Resume requires a UUIDv4 round identifier and stream=true.")
    try:
        after = cursor(handler)
    except ValueError:
        return handler._anthropic_error(400, "Invalid resume cursor.")
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).digest()
    with LOCK:
        reap_locked()
        job = JOBS.get(key)
        if job is None:
            if key in EXPIRED:
                return handler._anthropic_error(410, "Round expired. It was not restarted.")
            if len(JOBS) + len(EXPIRED) >= MAX_IDS:
                return handler._anthropic_error(503, "Resume identifier capacity is full. No round was started.")
            # Retried POSTs must carry Last-Event-ID, including zero. Unknown
            # reattachments must never cause another model invocation.
            if "Last-Event-ID" in handler.headers:
                return handler._anthropic_error(410, "Round is unavailable or expired. It was not restarted.")
            if len(JOBS) >= MAX_JOBS:
                for old_key, old in sorted(JOBS.items(), key=lambda item: item[1].created):
                    with old.cv:
                        if old.finished is not None and not old.readers and not old.pins:
                            old.file.close()
                            del JOBS[old_key]
                            EXPIRED[old_key] = time.monotonic() + ID_RETENTION
                            break
            if len(JOBS) >= MAX_JOBS:
                return handler._anthropic_error(503, "Resume capacity is full. No round was started.")
            job = Job(fingerprint)
            JOBS[key] = job
            threading.Thread(target=produce, args=(job, req, upstream, converter, event), daemon=True, name="resume-round").start()
        elif job.fingerprint != fingerprint:
            return handler._anthropic_error(409, "Round identifier already belongs to a different request.")
        with job.cv:
            job.pins += 1
    handler._handed()
    try:
        return send(handler, job, after)
    finally:
        with job.cv:
            job.pins -= 1
