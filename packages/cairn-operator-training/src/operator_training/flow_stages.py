"""ML-flow kernels (ADR-0082 D7/P1b, #1074).

The training pack's stage KERNELS for ``kind: ml`` flows, registered under the
engine-neutral ``cairn.flow_stages`` entry-point group. Same contract as every
flow kernel (ADR-0081 §5)::

    stage(workdir: Path, config: dict, inputs: list[str]) -> dict   # scalar stats

Kernels are thin adapters around the PROVEN operator bodies — ``train.qlora``
wraps ``StubTrainOp``/``LocalTrainOp`` (which itself routes validate/CPU/GPU
and calls ``train_entry.main`` on a real CUDA device), never a re-implementation.
Engines never know which kernels exist: builtin, SSH, SkyPilot, or a Metaflow
bridge all resolve THIS registry (the seam that keeps cairn engine-neutral).

Chaining convention for ML stages: the working record is
``workdir/train_result.json`` (a TrainResult dump) — the ML analogue of the
data kernels' ``current.jsonl``. Heavy trainer deps stay INSIDE ``LocalTrainOp``
(D5); the stub path is hermetic (no torch), which is what pack tests and the
conformance path run.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from operator_training import dataset_fetch, live
from operator_training.publish import publish_adapter_dir


class _Ctx:
    """Minimal OperationContext stand-in for reused operator bodies (mirrors
    the dataset_checks kernels): kernels run engine-side with no graph.
    ``report`` (#2025) is the stage's :class:`~operator_training.live.LiveReporter`,
    built from the env the flow engine handed this process — the same surface
    the operator bodies call as ``ctx.report``. ``workdir`` is the job's working
    directory, where ``train/local`` writes the trainer's output
    (``checkpoint_sync.trainer_output_dir``)."""
    pipeline_id = "ml-flow-stage"
    node_name = "kernel"
    compute = None

    def __init__(self, report: Any = None, workdir: Path | None = None) -> None:
        self.report = report if report is not None else live.NullReporter()
        self.workdir = workdir


_RESULT_FILE = "train_result.json"

# #1322 — observed live: a 0.5B QLoRA on the t2.large peaked ~3.4GB RSS
# before the kernel OOM-killed it 45 minutes in. ~7 bytes/param covers
# weights + LoRA grads/optimizer + activations for the CPU path.
_TRAIN_BYTES_PER_PARAM = 7.0
# cairn-extensions#7 — on a CUDA box the trainer is the Unsloth GPU path: the weights live in
# VRAM, and host RAM holds the process, the tokenizer and the data. Measured on a g4dn.xlarge
# (T4, 15 GiB RAM): Qwen3-4B-Base 4-bit QLoRA peaked at 2.3 GiB RSS over a full epoch (0.58
# B/param), where the CPU rate refused it for an estimated 26.1 GiB. 1.0 B/param keeps ~70 %
# headroom over that measurement.
_GPU_TRAIN_BYTES_PER_PARAM = 1.0


def _host_bytes_per_param(config: dict) -> float:
    """The host-RAM rate for the path this train will take: the GPU path when a CUDA device is
    there and the run is not a CPU validation (the same routing ``LocalTrainOp`` applies)."""
    if config.get("validate_pipeline") or os.environ.get("CAIRN_TRAIN_VALIDATE"):
        return _TRAIN_BYTES_PER_PARAM
    try:
        import torch  # noqa: PLC0415 - a cheap cuda probe; absent on a CPU-only control plane
        return _GPU_TRAIN_BYTES_PER_PARAM if torch.cuda.is_available() else _TRAIN_BYTES_PER_PARAM
    except Exception:  # noqa: BLE001 — no torch / probe failure ⇒ the CPU path's rate
        return _TRAIN_BYTES_PER_PARAM


def _param_count_from_name(base_model: str) -> float | None:
    """Billions-of-params from names like ``Qwen2.5-0.5B-Instruct`` /
    ``Llama-3-8b``; None when the name carries no size (check skipped)."""
    import re  # noqa: PLC0415 - breaking none: tiny stdlib helper kept local to the one caller

    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z0-9])", base_model)
    return float(m.group(1)) * 1e9 if m else None


def _available_ram_bytes() -> float | None:
    try:
        import psutil  # noqa: PLC0415 - optional dependency: not in the base install
        return float(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 — feasibility is best-effort, never a crash
        try:
            return float(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError, AttributeError):
            return None


def _refuse_if_ram_infeasible(base_model: str, config: dict) -> None:
    """#1322 pre-flight: refuse a train the box cannot hold BEFORE spending
    the training time. Heuristic and overridable (``ignore_ram_check: true``
    or ``CAIRN_TRAIN_IGNORE_RAM=1``); unparseable model names and unknowable
    free RAM skip the check rather than block."""
    if config.get("ignore_ram_check") or os.environ.get("CAIRN_TRAIN_IGNORE_RAM"):
        return
    params = _param_count_from_name(base_model)
    avail = _available_ram_bytes()
    if params is None or avail is None:
        return
    rate = _host_bytes_per_param(config)
    need = params * rate
    if need > avail:
        path = "GPU path" if rate == _GPU_TRAIN_BYTES_PER_PARAM else "CPU path, calibrated on the live #1322 OOM"
        raise RuntimeError(
            f"train.qlora refused before starting: {base_model} needs an "
            f"estimated {need / 2**30:.1f} GiB RSS (~{rate:.1f} B/param, {path}) "
            f"but only {avail / 2**30:.1f} GiB is "
            f"free — the kernel OOM-killer would end this run mid-train. "
            f"Use a smaller base model / bigger box, or set ignore_ram_check: true "
            f"to proceed anyway (swap may absorb it, slowly).")


def _result_path(workdir: Path) -> Path:
    return workdir / _RESULT_FILE


def _read_result(workdir: Path) -> dict:
    p = _result_path(workdir)
    if not p.is_file():
        raise RuntimeError(
            f"{_RESULT_FILE} missing in workdir — train.qlora must run before "
            "this stage (ML-flow chaining convention)"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _stub_selected(config: dict) -> bool:
    if "stub" in config:
        return bool(config["stub"])
    return os.environ.get("CAIRN_TRAIN_STUB", "").strip().lower() in ("1", "true", "yes", "on")


#: What ``sample.fidelity`` leaves behind for ``train.qlora`` — the subsampled file
#: plus the record of how it was produced (#2029). The ML-flow analogue of the
#: data kernels' ``current.jsonl``: a stage hands the next one a FILE, never a
#: variable, because engines do not share a process.
#:
#: The kernel is NOT called ``sample.rows``: that name is taken in this same
#: shared registry by dataset_checks' dataflow reservoir sampler, and a kernel
#: name is global across packs. The two share the ALGORITHM instead
#: (:mod:`cairn.core.dataflow.fidelity`), which is the thing that must not be duplicated.
_FIDELITY_FILE = "fidelity_sample.jsonl"
_FIDELITY_RECORD = "fidelity.json"


def _fidelity_record(workdir: Path) -> dict:
    p = workdir / _FIDELITY_RECORD
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover — a corrupt record is no record
        return {}


def sample_fidelity(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """The FIDELITY knob (T10, #2029): train on a deterministic subsample.

    A run on a subsample is only worth having if two such runs are COMPARABLE.
    So this stage keeps the rows :func:`cairn.core.dataflow.fidelity.plan` chooses — the
    ``n`` with the smallest ``H(seed ‖ row_hash)``, a choice that depends on the
    corpus and the seed and on nothing else — and records ``seed``, ``n`` and an
    order-independent **row-hash digest** in two places: the stage's stats (the
    flow leg's findings) and ``TrainSpec.extra``, which ADR-0103 D3 records
    verbatim as the run's ``parameters`` event. Re-running with the same seed
    reproduces the digest; two runs whose digests differ trained on different
    data and the record says so.

    A no-op — no fetch, no read, nothing written — when no fidelity is set, so
    a full-fidelity flow is byte-identical to one without this stage.
    """
    from cairn.core.dataflow.fidelity import Fidelity, plan  # noqa: PLC0415 - kept off the registry-enumeration path

    fid = Fidelity.from_config(config)
    if not fid.is_set():
        return {"fidelity_sampled": False, "fidelity_n": 0, "fidelity_total": 0,
                "fidelity_seed": fid.seed, "fidelity_digest": "", "fidelity_rows": 0,
                "fidelity_fraction": 0.0, "skipped": True}
    uri = str(config.get("dataset_uri") or (inputs[0] if inputs else "") or "")
    if not uri:
        raise RuntimeError("sample.fidelity: a fidelity is set but no dataset_uri reached the stage")
    local = dataset_fetch.resolve_dataset_uri(uri)
    src = Path(local)
    if not src.is_file():
        raise RuntimeError(
            f"sample.fidelity: {uri} resolved to {local!r}, which is not a local file — the "
            f"fidelity knob subsamples a JSONL corpus, so a Hub id or a directory must be "
            f"exported to a JSONL file first (the dataset stages' export.jsonl)")
    with src.open(encoding="utf-8") as fh:
        rows = [line for line in fh if line.strip()]
    result = plan(rows, fid)
    out = workdir / _FIDELITY_FILE
    with out.open("w", encoding="utf-8") as fh:
        for line in result.kept:
            fh.write(line if line.endswith("\n") else line + "\n")
    stats = result.as_stats(fid)
    (workdir / _FIDELITY_RECORD).write_text(
        json.dumps({**stats, "dataset_uri": uri, "path": str(out)}, indent=2), encoding="utf-8")
    return {**stats, "output": str(out)}


def _spec_from_config(config: dict, inputs: list[str], workdir: Path | None = None):
    from operator_training.schemas.train import TrainSpec  # noqa: PLC0415 - keep pydantic models out of registry enumeration (D5)

    fields = {k: v for k, v in config.items() if k in TrainSpec.model_fields}
    if not fields.get("dataset_uri") and inputs:
        # the upstream artifact (the promoted dataset REFERENCE, ADR-0080)
        fields["dataset_uri"] = str(inputs[0])
    # #2029 — a preceding `sample.rows` REPLACES the dataset for this run and
    # its record rides `extra`, which ADR-0103 D3 writes into the run's
    # `parameters` event: the fidelity is on the record of the run that used it,
    # not only in a stage's stats.
    # cairn-extensions#4: a governed train split is the training data; dataset_uri mirrors it
    # when absent, so TrainSpec (and train.run/1.0.0's required dataset_uri) still validate.
    if fields.get("train_uri") and not fields.get("dataset_uri"):
        fields["dataset_uri"] = fields["train_uri"]
    record = _fidelity_record(workdir) if workdir is not None else {}
    if record.get("fidelity_sampled") and record.get("path"):
        # the subsample REPLACES whichever source trains: train_uri when governed, else dataset_uri
        fields["train_uri" if fields.get("train_uri") else "dataset_uri"] = str(record["path"])
        extra = dict(fields.get("extra") or {})
        extra.update({k: v for k, v in record.items() if k.startswith("fidelity_")})
        fields["extra"] = extra
    # A pipeline `with:` list arrives STRINGIFIED through the config
    # substitution boundary ("['mlflow']") — caught live on the first
    # e2e-fullstack smoke. Coerce the common shapes back to a list; a plain
    # comma string ("mlflow,wandb") is accepted for the same reason.
    for _key in ("report_to", "target_modules"):
        rt = fields.get(_key)
        if isinstance(rt, str):
            fields[_key] = [t for t in
                            (p.strip(" '\"") for p in rt.strip("[] ").split(","))
                            if t]
    return TrainSpec.model_validate(fields)


def train_qlora(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """QLoRA/LoRA fine-tune — wraps the proven train bodies.

    ``config.stub`` (or ``CAIRN_TRAIN_STUB``) → ``StubTrainOp`` (hermetic, no
    torch — the conformance/test tier). Otherwise ``LocalTrainOp``, which owns
    the validate/CPU/GPU routing and calls ``train_entry.main`` on real CUDA.
    Writes ``train_result.json`` for the downstream stages; returns scalars.
    """
    spec = _spec_from_config(config, inputs, workdir)
    if _stub_selected(config):
        from operator_training.operators.stub import StubTrainOp  # noqa: PLC0415 - operator bodies own their deps (D5)
        op = StubTrainOp()
    else:
        _refuse_if_ram_infeasible(spec.base_model, config)  # #1322 — before, not after, 45 min
        from operator_training.operators.local import LocalTrainOp  # noqa: PLC0415 - trainer deps live behind the [local] extra (D5)
        op = LocalTrainOp()
    reporter = live.reporter_from_env(node="train.qlora")  # #2025: the live curve, as this stage
    try:
        res = asyncio.run(op.run(spec, _Ctx(reporter, workdir=workdir)))
    finally:
        reporter.close()
    if res.status != "succeeded":
        _result_path(workdir).write_text(res.model_dump_json(), encoding="utf-8")
        raise RuntimeError(f"train.qlora failed: {res.error or res.status}")

    # #1715 — HONOR artifact_target ON THIS PATH. The direct path publishes in
    # local_entry; the flow path had no equivalent, so a run declaring
    # `artifact_target: s3` ended with adapter_uri = /tmp/<name> — a path on the
    # very box the pipeline's `release` finalizer stops, destroying the adapter
    # the run exists to produce. Nothing fetches the adapter dir back here (only
    # the manifests come home), so the published URI REPLACES the local one as
    # the canonical adapter_uri, and a publish failure is FATAL: continuing would
    # hand `package.adapter` — and then promotion — a path that is about to
    # vanish. The local dir stays on the record as local_adapter_dir for debugging.
    local_dir = res.adapter_uri or ""
    pure_stub = _stub_selected(config) and not (getattr(spec, "extra", None) or {}).get("mock_adapter_uri")
    if pure_stub and not Path(local_dir).is_dir():
        # #2189 (T04) — the stub tier leaves DETERMINISTIC adapter bytes behind
        # (the spec's knobs as adapter_config.json + a placeholder weights file
        # derived from them, plus the data's own hash), so the candidate identity
        # computed downstream is the real hashing rule on real files. A real
        # trainer never yields byte-identical adapters twice; what the stub makes
        # testable is the REGISTRY rule — identical bytes register once. The
        # stub still publishes nothing: `local_dir` handed to publish stays the
        # reflected URI, exactly as before.
        res.local_adapter_dir = str(_write_stub_adapter(workdir, spec))
    else:
        res.local_adapter_dir = local_dir
    published = publish_adapter_dir(
        spec.artifact_target, local_dir,
        output_name=spec.output_name or "", hf_repo=getattr(spec, "hf_repo", "") or "",
    )
    if published:
        res.adapter_uri = published

    # written AFTER the publish so package.adapter (which reads this file) and the
    # returned scalars agree on one canonical URI.
    _result_path(workdir).write_text(res.model_dump_json(), encoding="utf-8")
    metrics = res.metrics or {}
    return {
        "status": res.status,
        "backend": res.backend,
        "adapter_uri": res.adapter_uri or "",
        "local_adapter_dir": res.local_adapter_dir,
        "train_loss": metrics.get("train_loss"),
        "eval_loss": metrics.get("eval_loss"),
        "steps": metrics.get("steps"),
        "base_model": spec.base_model,
        "method": spec.method,
        # cairn-extensions#7 — present only with baseline_eval (the value, or why it was skipped).
        **{k: metrics[k] for k in ("baseline_eval_loss", "baseline_skipped") if k in metrics},
    }


def eval_checkpoints(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Checkpoint-level acceptance on the TRAINING metrics (v1).

    This is the engine-side seam of ADR-0082 §7 Q4: quick metric thresholds
    that decide whether packaging is worth doing at all (fail fast — don't
    package a candidate whose loss never converged). cairn's governed model
    gate / benchmark on the packaged candidate stays OUTSIDE the flow (D4).

    Config: ``max_train_loss`` / ``max_eval_loss`` (optional float ceilings;
    a set ceiling with a MISSING metric fails closed) · ``fail_on_reject``
    (default true → a rejected checkpoint fails the stage, aborting the flow).
    """
    result = _read_result(workdir)
    metrics = result.get("metrics") or {}
    reasons: list[str] = []
    for key, ceiling_key in (("train_loss", "max_train_loss"), ("eval_loss", "max_eval_loss")):
        ceiling = config.get(ceiling_key)
        # unset OR non-positive = no ceiling (operation bindings default the
        # knob to 0 via $default — 0 must never mean "reject everything")
        if ceiling in (None, "") or float(ceiling) <= 0:
            continue
        value = metrics.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            reasons.append(f"{key} missing but {ceiling_key}={ceiling} set (fail-closed)")
        elif float(value) > float(ceiling):
            reasons.append(f"{key}={value} > {ceiling_key}={ceiling}")
    passed = result.get("status") == "succeeded" and not reasons
    stats = {
        "evaluated": True,
        "passed": passed,
        "train_loss": metrics.get("train_loss"),
        "eval_loss": metrics.get("eval_loss"),
        "reason": "; ".join(reasons),
    }
    if not passed and config.get("fail_on_reject", True):
        raise RuntimeError(f"eval.checkpoints rejected the candidate: {'; '.join(reasons) or 'training did not succeed'}")
    return stats


#: The files that ARE the candidate: weights and the adapter's own config. Logs,
#: checkpoints, tokenizer copies and manifests are not identity (#2189).
_CANDIDATE_FILES = ("adapter_config.json", "config.json")
_CANDIDATE_SUFFIXES = (".safetensors", ".bin")
_STUB_KNOBS = ("base_model", "method", "lora_r", "lora_alpha", "learning_rate", "epochs",
               "max_steps", "seed", "target_modules", "max_seq_len", "output_name")


def _write_stub_adapter(workdir: Path, spec) -> Path:
    """The stub tier's adapter directory: ``adapter_config.json`` = the spec's
    knobs, ``adapter_model.safetensors`` = a placeholder derived from them.
    Deterministic by construction — identity tests are about the hashing rule,
    not about a trainer."""
    d = workdir / "adapter"
    d.mkdir(parents=True, exist_ok=True)
    knobs = {k: getattr(spec, k, None) for k in _STUB_KNOBS}
    # the DATA is part of the identity: the dataset's own content hash when the
    # bytes are local (the readiness export always is), else its reference —
    # two stub trains on different data must never share a candidate id.
    src = str(getattr(spec, "dataset_uri", "") or "")
    local = src[len("file://"):] if src.startswith("file://") else src
    if local and Path(local).is_file():
        knobs["dataset_sha256"] = hashlib.sha256(Path(local).read_bytes()).hexdigest()
    else:
        knobs["dataset_uri"] = src
    cfg = json.dumps(knobs, sort_keys=True, default=str).encode("utf-8")
    (d / "adapter_config.json").write_bytes(cfg)
    (d / "adapter_model.safetensors").write_bytes(b"CAIRN-STUB-ADAPTER\n" + hashlib.sha256(cfg).hexdigest().encode("ascii"))
    return d


def hash_candidate_dir(path: Path) -> "tuple[str, int] | None":
    """Content identity of an adapter/model directory: sha256 over the sorted
    candidate files (relative name + bytes). ``None`` when nothing hashable is
    there — a remote-only URI has no bytes here, and that is reported, never
    faked. Hash the safetensors, not the logs (#2189 risk note)."""
    if not path.is_dir():
        return None
    files = sorted(p for p in path.rglob("*")
                   if p.is_file() and (p.name in _CANDIDATE_FILES or p.suffix in _CANDIDATE_SUFFIXES))
    if not files:
        return None
    h = hashlib.sha256()
    total = 0
    for p in files:
        h.update(str(p.relative_to(path)).encode("utf-8") + b"\0")
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                total += len(chunk)
    return h.hexdigest(), total


def package_adapter(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Package the accepted candidate: a ``candidate.json`` manifest naming the
    adapter URI + provenance, and — #2189 (T04, C6) — the candidate's CONTENT
    IDENTITY as the stage's product (``output`` + ``content_hash`` + ``kind:
    model``), which the runner turns into the model descriptor the control
    plane registers. The flow PACKAGES a candidate; it never registers or
    promotes (D4 — registration/promotion stay cairn's gated runs, #1052).
    """
    result = _read_result(workdir)
    adapter_uri = result.get("adapter_uri") or ""
    if not adapter_uri:
        raise RuntimeError("package.adapter: no adapter_uri in train_result.json")
    local_dir = str(result.get("local_adapter_dir") or "")
    hashed = hash_candidate_dir(Path(local_dir)) if local_dir else None
    if hashed is None and adapter_uri.startswith("/"):
        hashed = hash_candidate_dir(Path(adapter_uri))
    cfg = Path(local_dir) / "adapter_config.json" if local_dir else None
    spec_digest = hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg is not None and cfg.is_file() else ""
    base_model = (result.get("metrics") or {}).get("base_model") or config.get("base_model", "")
    candidate = {
        "adapter_uri": adapter_uri,
        "base_model": base_model,
        "backend": result.get("backend", ""),
        "metrics": result.get("metrics") or {},
        "content_hash": hashed[0] if hashed else "",
        "spec_digest": spec_digest,
        "packaged_at": datetime.now(timezone.utc).isoformat(),
    }
    out = workdir / "candidate.json"
    out.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    stats = {
        "packaged": True,
        "candidate_uri": adapter_uri,
        "candidate_manifest": str(out),
        "candidate_hashed": hashed is not None,
        "base_model": base_model,
        "method": str((result.get("metrics") or {}).get("method") or ""),
        "spec_digest": spec_digest,
    }
    if hashed is not None:
        # the descriptor keys the runner reads (#1442): a real product with a
        # content identity, kind=model so the registry knows what it is
        stats.update({"output": adapter_uri, "content_hash": hashed[0], "bytes": hashed[1], "kind": "model"})
    return stats


# ─── stage contracts (#158, ADR-0118 D1/M3a) ──────────────────────────────────
# The four `sft-train` kernels implement their shared stage contract
# (cairn-contracts-training-stages): `contract://training.stage.<kernel>/1.0.0`, recorded at the
# launch door, checked on the control plane and in the runner. An engine that does not declare
# `carries_stage_contracts` refuses the flow (D7) — builtin, SSH and Metaflow do. The other
# kernels in this module (train.distill, dataset.dpo_pairs, …) stay legacy until their slices.
for _fn in (sample_fidelity, eval_checkpoints, package_adapter):
    _fn.implements = (f"contract://training.stage.{_fn.__name__}/1.0.0",)
# cairn-extensions#4 / #7 — train.qlora implements 1.2.0: the governed splits (train/eval/test URIs)
# and the batch, accumulation, checkpointing and dropout knobs (1.1.0; `extra` is not
# configuration), and baseline_eval, the base model's eval loss with the adapter disabled (1.2.0).
train_qlora.implements = ("contract://training.stage.train_qlora/1.2.0",)


_ABSTAIN = "I don't know."


def dataset_dpo_pairs(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """Mine DPO preference pairs from a closed-book measure's graded answers
    (#1463 round 4). Reads ``current.jsonl`` (the ``<tag>_answers.jsonl`` rows
    ingested by ``ingest.normalize``: question/answer/split/label/model_answer),
    rewrites it as ``{prompt, chosen, rejected}`` pairs for ``export.jsonl``.

    Pair kinds (the mixed-pairs design — defend recall while suppressing
    fabrication, not just teach abstention):

    * ``abstain``  — held-out (``eval``) hallucinations: the model CANNOT know
      these facts → chosen ``I don't know.``, rejected its own fabrication.
    * ``correct``  — trained-fact (``paraphrase_eval``) hallucinations: it
      SHOULD know these → chosen the gold answer, rejected the fabrication.
      Control hallucinations get the same treatment.
    * ``recall``   — trained facts it got right: chosen its own correct answer
      (on-policy), rejected ``I don't know.`` — the explicit recall defense.

    Config: ``max_per_kind`` (0 = all), ``seed`` (deterministic shuffle).
    """
    import random  # noqa: PLC0415 - breaking none: tiny stdlib helper kept local to the one caller

    src = workdir / "current.jsonl"
    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    kinds: dict[str, list[dict]] = {"abstain": [], "correct": [], "recall": []}
    for r in rows:
        q, gold = r.get("question"), str(r.get("answer", ""))
        label, split, ans = r.get("label"), r.get("split"), str(r.get("model_answer", ""))
        if not q or label in ("error", None):
            continue
        base = {"prompt": q, "split": split, "fact_id": r.get("fact_id", "")}
        if label == "hallucinated" and split == "eval":
            kinds["abstain"].append({**base, "kind": "abstain", "chosen": _ABSTAIN, "rejected": ans})
        elif label == "hallucinated" and split in ("paraphrase_eval", "control"):
            kinds["correct"].append({**base, "kind": "correct", "chosen": gold, "rejected": ans})
        elif label == "correct" and split == "paraphrase_eval":
            kinds["recall"].append({**base, "kind": "recall", "chosen": ans or gold, "rejected": _ABSTAIN})
    cap = int(config.get("max_per_kind", 0) or 0)
    rng = random.Random(int(config.get("seed", 42)))
    pairs: list[dict] = []
    for k, group in kinds.items():
        rng.shuffle(group)
        pairs.extend(group[:cap] if cap else group)
    rng.shuffle(pairs)
    with src.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return {"rows": len(pairs),
            "abstain_pairs": len(kinds["abstain"][:cap] if cap else kinds["abstain"]),
            "correct_pairs": len(kinds["correct"][:cap] if cap else kinds["correct"]),
            "recall_pairs": len(kinds["recall"][:cap] if cap else kinds["recall"])}


def train_dpo(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """DPO alignment of an EXISTING adapter on preference pairs (#1463 r4) —
    wraps ``dpo_entry.main`` the way ``train.qlora`` wraps the SFT bodies.
    ``config.stub`` short-circuits with a canned result (hermetic tests; the
    real path needs the GPU backend's unsloth/trl)."""
    spec = _spec_from_config(config, inputs)
    if _stub_selected(config):
        result = {"adapter_uri": f"/tmp/{spec.output_name}", "metrics": {"train_loss": 0.1}, "run_url": ""}
    else:
        from operator_training import dpo_entry  # noqa: PLC0415 - trainer deps live on the GPU backend only (D5)
        result = dpo_entry.main(spec.model_dump())
    metrics = result.get("metrics") or {}
    _result_path(workdir).write_text(json.dumps({
        "status": "succeeded", "backend": "unsloth-dpo",
        "adapter_uri": result.get("adapter_uri", ""), "metrics": metrics,
    }), encoding="utf-8")
    return {
        "status": "succeeded",
        "adapter_uri": result.get("adapter_uri", ""),
        "train_loss": metrics.get("train_loss"),
        "base_model": spec.base_model,
        "init_adapter_uri": spec.init_adapter_uri,
        "beta": spec.beta,
        **{k: metrics[k] for k in ("registered_model_version", "mlflow_run_id", "checkpoint_uri") if k in metrics},
    }


def train_distill(workdir: Path, config: dict, inputs: list[str]) -> dict:
    """GKD distillation of a teacher into a student adapter (#2643) — the ml-flow
    door for ``compress/distill``, so the trainer runs on the ml engine's GPU target
    and never in the control plane (a ``task:`` leg runs in-process, #2182).
    Calls the operator's own body (``distill_body``); ``config.stub`` is its
    deterministic tier. Stats are flat scalars (D3)."""
    from operator_training.operators.distill import distill_body  # noqa: PLC0415 - operator bodies own their deps (D5)
    from operator_training.schemas.distill import DistillSpec  # noqa: PLC0415 - keep pydantic models out of registry enumeration (D5)

    fields = {k: v for k, v in config.items() if k in DistillSpec.model_fields}
    if not fields.get("dataset_uri") and inputs:
        fields["dataset_uri"] = str(inputs[0])
    res = distill_body(DistillSpec.model_validate(fields))
    out = {"status": res.status, "backend": res.backend, "adapter_uri": res.adapter_uri,
           "teacher_model": res.teacher_model, "student_model": res.student_model}
    # every scalar metric rides as a flat stat (D3) — strings too: mlflow_run_id,
    # registered_model_version, checkpoint_uri and the #1709 `tracking_error` soft-failure
    # signal are what an approver needs; nested values never cross the boundary
    out.update({k: v for k, v in (res.metrics or {}).items()
                if isinstance(v, (str, int, float, bool)) and k not in out})
    return out


# risk classes (ADR-0081 D2 fail-closed discipline, ADR-0082 D7):
# train.qlora = paid compute + writes adapters to hub/S3 → external_write
# (structurally forces the compile-required approval gate on every ML flow
# that trains — intended). package.adapter emits the publishable candidate →
# external_write. eval.checkpoints only reads metrics. train.dpo trains →
# external_write; dataset.dpo_pairs only reshapes rows already in the workdir.
train_qlora.risk_class = "external_write"
eval_checkpoints.risk_class = "read_only"
package_adapter.risk_class = "external_write"
train_dpo.risk_class = "external_write"
train_distill.risk_class = "external_write"   # paid GPU + writes the student adapter
dataset_dpo_pairs.risk_class = "read_only"
