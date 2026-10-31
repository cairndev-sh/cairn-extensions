#!/usr/bin/env bash
# demos/laptop/install.sh — one virtualenv with everything the laptop demo runs (see README.md).
#
#   demos/laptop/install.sh                  # creates demos/laptop/.venv
#   LAPTOP_DEMO_VENV=/path/to/venv demos/laptop/install.sh
#
# The released engine (cairndev 0.3.3) plus this checkout's packs and operators, installed
# editable: the fixes the demo needs (eval-lm 0.3.0's CPU/MPS device and `heldout-loss` preset,
# training 0.1.1's file:// adapter hash, lm-eval[api]) are on main and not yet on PyPI. torch comes
# from PyPI (macOS arm64 wheels carry MPS; Linux wheels are the CUDA build and run on CPU too, but
# are large; set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu for the small CPU build).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
venv="${LAPTOP_DEMO_VENV:-$here/.venv}"
core="${CAIRNDEV_VERSION:-0.3.3}"

command -v uv >/dev/null || { echo "install.sh: uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

echo "==> virtualenv: $venv (Python 3.12)"
uv venv -q "$venv" --python 3.12 --allow-existing
export VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH"

extra_index=()
[ -n "${TORCH_INDEX_URL:-}" ] && extra_index=(--extra-index-url "$TORCH_INDEX_URL" --index-strategy unsafe-best-match)

# The demo's event chain needs cairndev 0.3.3 (pipeline `emits:`, ADR-0161; the trigger switch,
# ADR-0160). Before that release, CAIRNDEV_SOURCE=/path/to/a/cairn/checkout installs it from there.
if [ -n "${CAIRNDEV_SOURCE:-}" ]; then
  echo "==> the core from $CAIRNDEV_SOURCE [train, dataflow-engine]"
  uv pip install -q "${extra_index[@]+"${extra_index[@]}"}" -e "$CAIRNDEV_SOURCE[train,dataflow-engine]" \
    -e "$CAIRNDEV_SOURCE/packages/cairn-client"
else
  echo "==> the released core: cairndev==$core [train, dataflow-engine]"
  uv pip install -q "${extra_index[@]+"${extra_index[@]}"}" "cairndev[train,dataflow-engine]==$core"
fi

echo "==> this checkout's contracts, operators and packs"
# The contract wheels are installed from the checkout too: dataset-checks pins a stages version
# (1.5.0) that is not on PyPI yet.
uv pip install -q "${extra_index[@]+"${extra_index[@]}"}" \
  "$repo/packages/cairn-contracts-dataset" \
  "$repo/packages/cairn-contracts-dataset-stages" \
  "$repo/packages/cairn-contracts-training" \
  "$repo/packages/cairn-contracts-training-stages" \
  -e "$repo/packages/cairn-operator-dataset-checks" \
  -e "$repo/packages/cairn-operator-training[local]" \
  -e "$repo/packages/cairn-operator-eval-lm[box]"
# The packs are data wheels (YAML and a profile): installed as wheels, not editable (an editable
# data wheel is a copy made at install time). Re-run this script after editing a pipeline.
uv pip install -q --reinstall-package cairn-pack-dataset-prep --reinstall-package cairn-pack-finetune \
  "$repo/packs/cairn-pack-dataset-prep" "$repo/packs/cairn-pack-finetune"

echo "==> a fake S3 for the registered versions and the adapter: moto[server]"
uv pip install -q "moto[server]>=5"

echo "==> check"
python - <<'PY'
import importlib.metadata as m
for d in ("cairndev", "cairn-operator-training", "cairn-operator-eval-lm", "cairn-pack-finetune", "torch"):
    print(f"  {d} {m.version(d)}")
import torch
print("  accelerator:", "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
PY
echo "installed. Next: demos/laptop/run.sh"
