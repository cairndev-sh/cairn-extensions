"""dataset/provenance — check a dataset's declared metadata.

Sources metadata in priority order, then verifies the provenance facts a
readiness gate cares about:

  1. **HF ``DatasetInfo``** (when ``path`` is given) — license / homepage→source /
     citation that a Hub dataset card already carries. We don't re-declare what
     ``datasets`` already provides.
  2. **sidecar JSON** (``metadata_path``) — for local datasets that carry no
     DatasetInfo, and for the governance flags `datasets` lacks
     (``synthetic`` / ``synthetic_validated``). Overrides/fills (1).

``provenance_ok`` = no required field missing AND (not synthetic OR validated).
Deterministic. Full Croissant (``mlcroissant``) parsing is the deferred
``[provenance]`` extra.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from cairn.core.operator import (
    Operation,
    OperationContext,
    register_operator,
)
from cairn.core.operator.reports import ReportSpec

from operator_dataset_checks._io import dataset_info
from operator_dataset_checks.schemas.provenance import (
    ProvenanceConfig,
    ProvenanceInput,
    ProvenanceOutput,
)


def _nonempty(meta: dict, key: str) -> bool:
    return bool(str(meta.get(key, "")).strip())


def _read_sidecar(source: str) -> str:
    """Read the metadata sidecar from a local path OR an http(s) URL — parity with
    the remote-aware dataset loader (#913). ``Path(url).read_text()`` collapses the
    URL's ``//`` into a bogus local path (``http:/…`` → FileNotFoundError), so
    http(s) must be fetched explicitly; ``file://`` is stripped to a plain path."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310 — governed metadata URL
            return resp.read().decode("utf-8")
    if source.startswith("file://"):
        source = source[len("file://"):]
    return Path(source).read_text(encoding="utf-8")


@register_operator(
    name="dataset/provenance",
    version="0.1.0",
    capabilities=["dataset.provenance"],
    implements=["contract://dataset.provenance/1.0.0"],  # #2242 V1-B — cairn-contracts-dataset
)
class ProvenanceOp(Operation[ProvenanceInput, ProvenanceOutput]):
    """Verify declared license, source, and synthetic-data validation."""

    input_schema = ProvenanceInput
    output_schema = ProvenanceOutput
    reports = ReportSpec.NONE  # ADR-0103: short-running, nothing to report
    config_schema = ProvenanceConfig  # ADR-0070 — GOVERNANCE: which metadata fields are required (env-owned)

    async def run(self, inp: ProvenanceInput, ctx: OperationContext) -> ProvenanceOutput:
        meta: dict = {}
        sources: list[str] = []

        # 1. Harvest HF DatasetInfo (license / homepage→source / citation).
        if inp.path:
            try:
                info = dataset_info(inp.path)
                harvested = {}
                if str(getattr(info, "license", "") or "").strip():
                    harvested["license"] = info.license
                if str(getattr(info, "homepage", "") or "").strip():
                    harvested["source"] = info.homepage
                if str(getattr(info, "citation", "") or "").strip():
                    harvested["citation"] = info.citation
                if harvested:
                    meta.update(harvested)
                    sources.append("dataset_info")
            except Exception:  # noqa: BLE001 — not loadable / no info → use sidecar
                pass

        # 2. Sidecar fills/overrides and carries governance flags.
        if inp.metadata_path:
            sidecar = json.loads(_read_sidecar(inp.metadata_path))
            meta.update(sidecar)
            sources.append("sidecar")

        missing = [f for f in inp.required_fields if not _nonempty(meta, f)]
        synthetic = bool(meta.get("synthetic", False))
        synthetic_validated = bool(meta.get("synthetic_validated", False))
        provenance_ok = (not missing) and (not synthetic or synthetic_validated)

        return ProvenanceOutput(
            metadata_sources=sources,
            license_present=_nonempty(meta, "license"),
            source_present=_nonempty(meta, "source"),
            synthetic=synthetic,
            synthetic_validated=synthetic_validated,
            missing_fields=missing,
            provenance_ok=provenance_ok,
        )
