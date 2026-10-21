"""The worker's end of the live channel (#2025, ADR-0102 D4) — stdlib only.

This module runs INSIDE the training container / on the remote box, where
``cairn`` itself may not be installed (the operator wheel is). So it carries no
cairn import: it produces the exact event shapes the control plane's typed
reporter (:mod:`cairn.core.operator.reports`) produces — ``progress`` ·
``metrics`` · ``scalars`` · ``checkpoint`` · ``divergence``, each stamped
``node`` + a per-node monotonic ``seq`` — and gets them to the control plane
over one of two transports:

* **HTTP** — ``POST {CAIRN_ENGINE_URL}/runs/{run}/events/ingest`` in batches,
  ``GET …/runs/{run}/control`` for control (rank-0 only, at most one poll per
  5 s, silent on failure — the Kubeflow-callback pattern);
* **NATS** — ``CAIRN_JOB_CHANNEL=nats://…``: publish each event to the run's
  events subject, read control from its control subject (needs ``nats-py`` on
  the box; imported lazily).

Every event is first appended to a **local spool** (a JSONL file beside the
job) and only marked shipped once the control plane acknowledged it, so a
network cut of any length is a replay on reconnect — the control plane's
ingest is idempotent on ``(node, seq)`` and the store ends up exact. Nothing
here ever raises into the trainer: a report is best-effort, the train is not.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

_log = logging.getLogger("operator_training.live")

PROGRESS, METRICS, SCALARS, CHECKPOINT, DIVERGENCE = "progress", "metrics", "scalars", "checkpoint", "divergence"

#: Control actions the control plane may send down (mirrors report_ingest.CONTROL_ACTIONS).
ABORT, CHECKPOINT_AND_STOP = "abort", "checkpoint-and-stop"

ENV_RUN_ID, ENV_NODE = "CAIRN_RUN_ID", "CAIRN_NODE"
ENV_ENGINE_URL, ENV_ENGINE_TOKEN = "CAIRN_ENGINE_URL", "CAIRN_ENGINE_TOKEN"
ENV_JOB_CHANNEL = "CAIRN_JOB_CHANNEL"
ENV_SPOOL = "CAIRN_LIVE_SPOOL"
#: The former prefix (ADR-0122). This module cannot import cairn's resolver, so it mirrors
#: its fallback for the two transport names: ``CAIRN_<NAME>`` first, then ``OBS_<NAME>``.
#: The control plane sends both job-channel names until 1.0, when the twins go with the
#: rebuilt train-gpu image (#2429).
_LEGACY_PREFIX = "OBS_"


def _env(env: Any, name: str) -> str:
    """``name`` (a ``CAIRN_*`` variable), else its ``OBS_*`` spelling — stripped, ``""`` if unset."""
    return (env.get(name, "").strip()
            or env.get(_LEGACY_PREFIX + name[len("CAIRN_"):], "").strip())


# ─── the spool ──────────────────────────────────────────────────────────


class Spool:
    """Append-only JSONL of ``{"type", "data"}`` with a shipped cursor.

    ``append`` writes and fsyncs; ``pending`` is everything after the cursor;
    ``ack(n)`` advances it. A worker that restarts re-reads the file and
    replays everything unacked — duplicates are the control plane's problem,
    and it solves them by ``seq``."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._items: list[dict[str, Any]] = []
        self._acked = 0
        self._lock = threading.Lock()
        if path and os.path.exists(path):
            self._reload(path)

    def _reload(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._items.append(json.loads(line))
        except Exception:  # noqa: BLE001 — a torn spool replays what it can
            _log.warning("spool %s unreadable; starting empty", path, exc_info=True)

    def append(self, item: dict[str, Any]) -> None:
        with self._lock:
            self._items.append(item)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(item, separators=(",", ":")) + "\n")
                        fh.flush()
                except Exception:  # noqa: BLE001 — the in-memory copy still ships
                    _log.debug("spool append to %s failed", self.path, exc_info=True)

    def pending(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._items[self._acked:self._acked + limit])

    def ack(self, n: int) -> None:
        with self._lock:
            self._acked = min(len(self._items), self._acked + max(0, int(n)))

    @property
    def unshipped(self) -> int:
        with self._lock:
            return len(self._items) - self._acked

    @property
    def last_seq(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for it in self._items:
                d = it.get("data") or {}
                node, seq = str(d.get("node") or ""), int(d.get("seq") or 0)
                if seq > out.get(node, 0):
                    out[node] = seq
        return out


# ─── transports ─────────────────────────────────────────────────────────


class Transport:
    """``send(batch) -> ok`` and ``control(after_seq) -> {seq, action} | None``."""

    def send(self, batch: list[dict[str, Any]]) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def control(self, after_seq: int) -> Optional[dict[str, Any]]:  # pragma: no cover - interface
        return None

    def close(self) -> None:
        return None


class HttpTransport(Transport):
    """The control plane's own API. ``post``/``get`` are injectable so the
    network-cut acceptance test can fail them for a window."""

    def __init__(self, engine_url: str, run_id: str, token: str = "", *, timeout_s: float = 10.0,
                 post: Optional[Callable[[str, dict[str, Any]], dict[str, Any]]] = None,
                 get: Optional[Callable[[str], dict[str, Any]]] = None) -> None:
        self.base = engine_url.rstrip("/")
        self.run_id = run_id
        self.token = token
        self.timeout_s = timeout_s
        self._post = post or self._http_post
        self._get = get or self._http_get

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _http_post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 — the control plane's own URL
            return json.loads(resp.read() or b"{}")

    def _http_get(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 — same
            return json.loads(resp.read() or b"{}")

    def send(self, batch: list[dict[str, Any]]) -> bool:
        try:
            res = self._post(f"{self.base}/runs/{self.run_id}/events/ingest", {"events": batch})
        except (urllib.error.URLError, OSError, ValueError) as e:
            _log.debug("ingest post failed: %s", e)
            return False
        except Exception:  # noqa: BLE001 — a transport must never raise into the trainer
            _log.debug("ingest post failed", exc_info=True)
            return False
        return isinstance(res, dict) and "accepted" in res

    def control(self, after_seq: int) -> Optional[dict[str, Any]]:
        try:
            res = self._get(f"{self.base}/runs/{self.run_id}/control?after_seq={int(after_seq)}&wait_s=0")
        except Exception:  # noqa: BLE001 — silent on failure: the trainer keeps going
            return None
        return (res or {}).get("control") if isinstance(res, dict) else None


class NatsTransport(Transport):
    """Publish to the run's JetStream subjects through the SAME channel class the
    control plane drains (``cairn.core.operator.job_channel.NatsJobChannel``),
    on a private event loop in a helper thread — the trainer's callbacks are
    synchronous. Needs ``nats-py`` on the box; without it, ``send`` reports
    failure and the spool keeps everything for a later transport."""

    def __init__(self, url: str, run_id: str, *, channel: Any = None) -> None:
        self.url, self.run_id = url, run_id
        self._chan = channel
        self._loop = None
        self._thread: Optional[threading.Thread] = None

    def _ensure(self) -> Any:
        if self._chan is None:
            from cairn.core.operator.job_channel import NatsJobChannel  # noqa: PLC0415 — optional on the box; only when NATS is the channel
            self._chan = NatsJobChannel(self.url)
        if self._loop is None:
            import asyncio  # noqa: PLC0415 — only this transport needs a loop
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="cairn-live-nats", daemon=True)
            self._thread.start()
        return self._chan

    def _run(self, coro: Any, timeout_s: float = 10.0) -> Any:
        import asyncio  # noqa: PLC0415 — same
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout_s)

    def send(self, batch: list[dict[str, Any]]) -> bool:
        try:
            chan = self._ensure()
            for item in batch:
                self._run(chan.publish_event(self.run_id, {"type": item["type"], "data": item["data"]}))
            return True
        except Exception:  # noqa: BLE001 — never into the trainer
            _log.debug("nats publish failed", exc_info=True)
            return False

    def control(self, after_seq: int) -> Optional[dict[str, Any]]:
        try:
            ctl = self._run(self._ensure().latest_control(self.run_id))
        except Exception:  # noqa: BLE001
            return None
        if not ctl or not ctl.get("action"):
            return None
        return {"seq": int(ctl.get("seq") or 0) or after_seq + 1, **ctl}

    def close(self) -> None:
        try:
            if self._chan is not None and self._loop is not None:
                self._run(self._chan.aclose())
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:  # noqa: BLE001
            pass


# ─── the reporter ───────────────────────────────────────────────────────


class LiveReporter:
    """The control plane's reporter surface, worker-side: ``stage`` · ``series``
    · ``scalars`` · ``checkpoint`` · ``divergence``. Every call appends to the
    spool; ``flush`` ships what is pending (at most every ``flush_every_s``
    unless forced); ``control`` asks the control plane what it wants (rate
    limited to one poll per ``control_every_s``). ``close`` is a final flush.
    """

    def __init__(self, run_id: str, node: str, transport: Optional[Transport], *,
                 spool: Optional[Spool] = None, flush_every_s: float = 5.0,
                 control_every_s: float = 5.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.run_id, self.node = run_id, node
        self.transport = transport
        self.spool = spool or Spool()
        self.flush_every_s, self.control_every_s = flush_every_s, control_every_s
        self._clock = clock
        self._seq = max(0, self.spool.last_seq.get(node, 0))
        self._last_flush = -1e9
        self._last_control = -1e9
        self._control_seq = 0
        self._lock = threading.Lock()

    # ── the surface (same names, same payloads as cairn.core.operator.reports) ──
    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            self.spool.append({"type": event_type, "data": {"node": self.node, "seq": self._seq, **payload}})
        self.flush()

    def stage(self, name: str, percent: Optional[float] = None, eta_s: Optional[float] = None,
              cost_usd: Optional[float] = None) -> None:
        p: dict[str, Any] = {"stage": name}
        if percent is not None:
            p["percent"] = max(0.0, min(100.0, float(percent)))
        if eta_s is not None:
            p["eta_s"] = float(eta_s)
        if cost_usd is not None:
            p["cost_usd"] = float(cost_usd)
        self._emit(PROGRESS, p)

    def series(self, step: int, **values: float) -> None:
        clean = {k: float(v) for k, v in values.items() if isinstance(v, (int, float)) and v == v}
        if clean:
            self._emit(METRICS, {"step": int(step), "series": clean})

    def scalars(self, **values: Any) -> None:
        self._emit(SCALARS, {"values": dict(values)})

    def checkpoint(self, ref: str, step: int, metric: Optional[float] = None) -> None:
        p: dict[str, Any] = {"ref": ref, "step": int(step)}
        if metric is not None:
            p["metric"] = float(metric)
        self._emit(CHECKPOINT, p)

    def divergence(self, kind: str, step: int, value: Optional[float] = None) -> None:
        p: dict[str, Any] = {"kind": kind, "step": int(step)}
        if value is not None:
            p["value"] = float(value)
        self._emit(DIVERGENCE, p)

    @property
    def seq(self) -> int:
        return self._seq

    # ── shipping ──
    def flush(self, *, force: bool = False) -> int:
        """Ship pending spool entries. Returns how many were acknowledged."""
        if self.transport is None:
            return 0
        now = self._clock()
        if not force and now - self._last_flush < self.flush_every_s:
            return 0
        self._last_flush = now
        shipped = 0
        while True:
            batch = self.spool.pending()
            if not batch:
                break
            if not self.transport.send(batch):
                break  # the link is down: keep the spool, try again next time
            self.spool.ack(len(batch))
            shipped += len(batch)
        return shipped

    def control(self, *, force: bool = False) -> Optional[str]:
        """The newest control action the control plane sent since the last one
        seen — ``abort`` / ``checkpoint-and-stop`` — or ``None``."""
        if self.transport is None:
            return None
        now = self._clock()
        if not force and now - self._last_control < self.control_every_s:
            return None
        self._last_control = now
        ctl = self.transport.control(self._control_seq)
        if not ctl:
            return None
        self._control_seq = max(self._control_seq, int(ctl.get("seq") or 0))
        return str(ctl.get("action") or "") or None

    def close(self) -> None:
        for _ in range(3):
            if self.flush(force=True) == 0 and self.spool.unshipped == 0:
                break
            time.sleep(0.5)
        if self.transport is not None:
            self.transport.close()


class NullReporter(LiveReporter):
    """A reporter with nowhere to send — every call is a no-op."""

    def __init__(self) -> None:
        super().__init__(run_id="", node="", transport=None)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        return None


def reporter_from_env(environ: Optional[dict[str, str]] = None, *, node: Optional[str] = None) -> LiveReporter:
    """The reporter a container entrypoint or a flow kernel uses: the run id
    (``CAIRN_RUN_ID``, or the flow engines' ``CAIRN_PARENT_RUN``), the node
    (``node`` from the caller — a flow stage names itself — else ``CAIRN_NODE``),
    and either ``CAIRN_JOB_CHANNEL=nats://…`` or the control plane's URL
    (``CAIRN_ENGINE_URL`` — the box's variable, and the control plane's own setting
    when the process IS on the control plane, as the builtin flow engine's subprocess
    is). Each falls back to its ``OBS_*`` spelling. No run id →
    :class:`NullReporter` (a standalone invocation)."""
    env = environ if environ is not None else os.environ
    run_id = env.get(ENV_RUN_ID, "").strip() or env.get("CAIRN_PARENT_RUN", "").strip()
    if not run_id:
        return NullReporter()
    node = (node or "").strip() or env.get(ENV_NODE, "").strip() or "train"
    spool = Spool(env.get(ENV_SPOOL) or f"/tmp/cairn-live-{run_id}-{node}.jsonl")
    chan = _env(env, ENV_JOB_CHANNEL)
    url = _env(env, ENV_ENGINE_URL)
    transport: Optional[Transport]
    if chan.startswith("nats://"):
        transport = NatsTransport(chan, run_id)
    elif url:
        transport = HttpTransport(url, run_id, env.get(ENV_ENGINE_TOKEN, "").strip())
    else:
        _log.warning("live channel: %s set but neither %s nor %s — events stay in the spool only",
                     ENV_RUN_ID, ENV_JOB_CHANNEL, ENV_ENGINE_URL)
        transport = None
    return LiveReporter(run_id, node, transport, spool=spool)


# ─── system sample (the `system` series the training operators declare) ───


def system_sample() -> dict[str, float]:
    """GPU util / VRAM / temperature / power via ``nvidia-smi``, CPU / RAM via
    ``psutil`` — whichever is present. Missing tools yield fewer keys, never an
    error."""
    out: dict[str, float] = {}
    if shutil.which("nvidia-smi"):
        try:
            q = subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            first = (q.stdout or "").strip().splitlines()[:1]
            if first:
                util, used, total, temp, power = [float(x.strip() or 0) for x in first[0].split(",")[:5]]
                out.update({"gpu_util_pct": util, "gpu_mem_gb": used / 1024.0, "gpu_mem_total_gb": total / 1024.0,
                            "gpu_temp_c": temp, "gpu_power_w": power})
        except Exception:  # noqa: BLE001 — a failed probe is an absent key
            pass
    try:
        import psutil  # noqa: PLC0415 — optional on the box

        out["cpu_pct"] = float(psutil.cpu_percent(interval=None))
        out["ram_gb"] = float(psutil.virtual_memory().used) / 1e9
    except Exception:  # noqa: BLE001 — same
        pass
    return out


__all__ = [
    "ABORT", "CHECKPOINT_AND_STOP", "CHECKPOINT", "DIVERGENCE", "METRICS", "PROGRESS", "SCALARS",
    "HttpTransport", "LiveReporter", "NatsTransport", "NullReporter", "Spool", "Transport",
    "reporter_from_env", "system_sample",
]
