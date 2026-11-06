"""Resident llama.cpp workers: each CPU model is loaded and hashed once, not once per candidate.

scripts/look_worker.py and scripts/draft_worker.py load the GGUF files of the frozen per-call scripts
(scripts/look_local.py, scripts/generate_local.py) with the same settings, then answer one request
per stdin line. A request names the same input, output and log files the per-call scripts use, so a
job directory holds the same records whichever path produced them. The server keeps one worker per
kind: started on first use, restarted after a crash, closed on exit. A worker that cannot start
raises WorkerUnavailable, and the caller runs the per-call script instead.

Both sides use only the standard library and Python 3.10, so the worker side runs in the
llama-cpp-python environment and the client side in the server's.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

START_TIMEOUT = 300  # load and hash the model; a cold disk read of the 1.7 GB VLM takes well under this
RETRY_AFTER = 300  # after a failed start, use the per-call path this long before trying again
STOP_TIMEOUT = 10


class WorkerUnavailable(RuntimeError):
    """The worker could not start; the per-call script runs instead."""


class WorkerCrashed(RuntimeError):
    """The worker died during a request; this request runs per call and the next one restarts it."""


def sha256(path) -> str:
    """Chunked SHA-256; hashlib.file_digest needs Python 3.11."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The worker side

@contextmanager
def redirected(path):
    """Point file descriptors 1 and 2 at one request's log, so llama.cpp's C output lands there."""
    if not path:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with open(path, "wb") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                yield
            finally:
                sys.stderr.flush()
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
    finally:
        os.close(saved[0])
        os.close(saved[1])


def serve(load, answer) -> int:
    """Run a worker: load() once, then answer(state, request) for each stdin line until stdin closes.

    load() returns (state, record); the record (file names, hashes, load time) is sent in the ready
    line. answer() writes request["output"] and may return extra fields for the reply. Protocol lines
    go to a private copy of the original stdout; descriptor 1 is pointed at stderr first, before
    llama_cpp is imported, so nothing the library prints can reach the protocol. Ctrl+C in the
    server's console is ignored here: the server ends the worker by closing its stdin.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    protocol = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    if os.name == "nt":  # a DLL with its own C runtime takes stdout from the Win32 handle, not descriptor 1
        import ctypes
        import msvcrt
        ctypes.windll.kernel32.SetStdHandle(-11, msvcrt.get_osfhandle(2))  # STD_OUTPUT_HANDLE
    sys.stdout = sys.stderr

    def send(value):
        protocol.write((json.dumps(value) + "\n").encode("utf-8"))

    try:
        state, record = load()
    except Exception as exc:
        traceback.print_exc()
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    send({"ready": True, "pid": os.getpid(), **record})
    for line in sys.stdin.buffer:
        if not line.strip():
            continue
        tick = time.monotonic()
        request = {}
        try:
            request = json.loads(line.decode("utf-8"))
            with redirected(request.get("log")):
                try:
                    extra = answer(state, request) or {}
                except Exception:
                    traceback.print_exc()
                    raise
            send({"id": request.get("id"), "ok": True, "seconds": round(time.monotonic() - tick, 3), **extra})
        except Exception as exc:
            send({"id": request.get("id") if isinstance(request, dict) else None, "ok": False,
                  "error": f"{type(exc).__name__}: {exc}"})
    return 0


def now():
    return datetime.now(timezone.utc).isoformat()


def provenance(record: dict, request: int) -> dict:
    """The "worker" entry of an output record: which resident process answered, and its one-time costs."""
    return {"pid": os.getpid(), "request": request, "started_at": record["started_at"],
            "load_seconds": record["load_seconds"], "hash_seconds": record["hash_seconds"]}


# The server side

class Worker:
    """One long-lived worker process, reached over JSON lines on its stdin and stdout."""

    def __init__(self, kind: str, command: list[str], env: dict, cwd: Path, log_path: Path):
        self.kind, self.command, self.env, self.cwd, self.log_path = kind, list(command), dict(env), cwd, Path(log_path)
        self.process = self.lines = self.record = None
        self.serial = self.requests = self.starts = 0
        self.failed = None  # (monotonic time, message) of the last failed start
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self):
        self.stop()
        if self.failed and time.monotonic() - self.failed[0] < RETRY_AFTER:
            raise WorkerUnavailable(self.failed[1])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.log_path.open("ab") as log:
                log.write(f"--- {now()} starting {self.kind} worker: {subprocess.list2cmdline(self.command)}\n".encode("utf-8"))
                log.flush()
                self.process = subprocess.Popen(self.command, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=log)
        except OSError as exc:
            self.process = None
            raise self._failure(f"cannot run {self.command[0]}: {exc}") from exc
        self.lines = queue.Queue()
        threading.Thread(target=self._read, args=(self.process.stdout, self.lines), daemon=True,
                         name=f"tracebrief-{self.kind}-worker").start()
        try:
            ready = self._receive(START_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.stop(force=True)
            raise self._failure(f"not ready after {START_TIMEOUT} s") from None
        if not ready or not ready.get("ready"):
            code = self.stop()
            raise self._failure((ready or {}).get("error") or f"exited with code {code}")
        self.failed, self.record = None, ready
        self.starts += 1

    def _failure(self, error) -> WorkerUnavailable:
        """Record a failed start, so the per-call path is used for RETRY_AFTER seconds."""
        message = f"The resident {self.kind} worker did not start ({error}); see {self.log_path}."
        self.failed = (time.monotonic(), message)
        return WorkerUnavailable(message)

    @staticmethod
    def _read(stream, lines):
        try:
            for line in stream:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            stream.close()
            lines.put(None)

    def _receive(self, timeout):
        """The next protocol message; None once the worker's stdout has closed."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise subprocess.TimeoutExpired(self.command, timeout) from None
            if line is None:
                return None
            try:
                message = json.loads(line.decode("utf-8"))
            except ValueError:
                continue  # not a protocol line
            if isinstance(message, dict):
                return message

    def run(self, request: Path, response: Path, log: Path, timeout: float) -> dict:
        """Have the worker read request and write response; returns its reply line.

        A worker error is a RuntimeError, a stuck worker is killed and raises subprocess.TimeoutExpired
        (as the per-call script would), and a crash raises WorkerCrashed.
        """
        with self.lock:
            if not self.alive():
                self.start()
            self.serial += 1
            line = json.dumps({"id": self.serial, "input": str(request), "output": str(response), "log": str(log)})
            try:
                self.process.stdin.write((line + "\n").encode("utf-8"))
                self.process.stdin.flush()
            except OSError as exc:
                self.stop()
                raise WorkerCrashed(f"The resident {self.kind} worker stopped accepting requests: {exc}") from exc
            deadline = time.monotonic() + timeout
            while True:
                try:
                    reply = self._receive(max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    self.stop(force=True)
                    raise
                if reply is None:
                    code = self.stop()
                    raise WorkerCrashed(f"The resident {self.kind} worker exited with code {code} during a request; see {self.log_path}.")
                if reply.get("id") == self.serial:
                    break
            self.requests += 1
            if not reply.get("ok"):
                raise RuntimeError(f"The resident {self.kind} worker could not answer: {reply.get('error')}")
            return reply

    def stop(self, force=False):
        """Close stdin (the worker exits at end of input), then terminate or kill if it does not; returns the exit code.

        force: the worker is stuck in a request, so it is terminated without waiting for it.
        """
        process, self.process = self.process, None
        if process is None:
            return None
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=0 if force else STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return process.returncode

    def close(self):
        with self.lock:
            self.stop()

    def status(self) -> dict:
        process = self.process  # read once; a request may replace it meanwhile
        running = process is not None and process.poll() is None
        return {"kind": self.kind, "running": running, "pid": process.pid if running else None,
                "starts": self.starts, "requests": self.requests, "error": self.failed[1] if self.failed else None}


WORKERS: dict[str, Worker] = {}
REGISTRY = threading.Lock()


def get(kind: str, command: list[str], env: dict, cwd: Path, log_path: Path) -> Worker:
    """The process's one worker of this kind; a new command or environment replaces it."""
    with REGISTRY:
        current = WORKERS.get(kind)
        if current is not None and (current.command, current.env) != (list(command), dict(env)):
            current.close()
            current = None
        if current is None:
            current = WORKERS[kind] = Worker(kind, command, env, cwd, log_path)
        return current


def close_all():
    """Stop every worker; registered with atexit and called when the server shuts down."""
    with REGISTRY:
        running = list(WORKERS.values())
        WORKERS.clear()
    for worker in running:
        try:
            worker.close()
        except Exception as exc:
            print(f"TraceBrief: stopping the {worker.kind} worker failed: {exc}", file=sys.stderr)


atexit.register(close_all)
