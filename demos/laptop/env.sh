# demos/laptop/env.sh — the environment the laptop demo runs in. `run.sh` sources it; source it
# yourself (`source demos/laptop/env.sh`) to drive the same server from another shell.
#
# Everything here is local: a throwaway S3 (moto) on 127.0.0.1, a sqlite file, an audit file. The
# AWS values are dummies that moto accepts; no cloud account or API key is used or needed.
# Override any of these before sourcing (for example LAPTOP_DEMO_PORT when 8765 is taken).

_laptop_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export LAPTOP_DEMO_DIR="$_laptop_dir"
export LAPTOP_DEMO_VENV="${LAPTOP_DEMO_VENV:-$_laptop_dir/.venv}"
# the working directory of one demo run (sqlite db, audit log, logs, moto's memory is not on disk)
export LAPTOP_DEMO_WORK="${LAPTOP_DEMO_WORK:-${TMPDIR:-/tmp}/cairn-laptop-demo}"
export LAPTOP_DEMO_PORT="${LAPTOP_DEMO_PORT:-8765}"
export LAPTOP_DEMO_S3_PORT="${LAPTOP_DEMO_S3_PORT:-5765}"

if [ -d "$LAPTOP_DEMO_VENV/bin" ]; then
  export VIRTUAL_ENV="$LAPTOP_DEMO_VENV" PATH="$LAPTOP_DEMO_VENV/bin:$PATH"
fi

# the profile: audit sink + builtin engines + the s3 artifact store (no ml_target: every leg runs here)
_pack="$(cd "$_laptop_dir/../.." && pwd)/packs/cairn-pack-finetune"
export CAIRN_PROFILE="$_pack/profiles/finetune-local.toml"
export CAIRN_STORAGE="sqlite:///$LAPTOP_DEMO_WORK/cairn.db"
export CAIRN_AUDIT_LOG="$LAPTOP_DEMO_WORK/audit.jsonl"
# the deployment's policies: two example policies in the finetune pack (docs/policies/laptop: a person owns them, no pack loads them)
export CAIRN_POLICIES_DIR="$_pack/docs/policies/laptop"

# the registered dataset versions and the adapter live in a local S3 (moto_server)
export CAIRN_ARTIFACT_S3_BUCKET="${CAIRN_ARTIFACT_S3_BUCKET:-cairn-laptop-demo}"
export CAIRN_ARTIFACT_S3_ENDPOINT="http://127.0.0.1:$LAPTOP_DEMO_S3_PORT"   # the control plane reads this name
export AWS_ENDPOINT_URL="$CAIRN_ARTIFACT_S3_ENDPOINT"                         # the trainer and the evaluation read this one
export CAIRN_CHECKPOINT_BUCKET="$CAIRN_ARTIFACT_S3_BUCKET"
export CAIRN_CHECKPOINT_PREFIX="${CAIRN_CHECKPOINT_PREFIX:-checkpoints}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-moto}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-moto}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export CAIRN_ARTIFACT_S3_REGION="$AWS_DEFAULT_REGION"

# the model is downloaded once from the Hugging Face Hub unless --base-model names a local directory
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# MPS ops without a kernel fall back to the CPU instead of failing (Apple silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# the control plane's address; `cairn dev` creates the user dev/dev, and `cairn remote login` keeps its
# token in the work dir, not in ~/.cairn
export CAIRN_URL="http://127.0.0.1:$LAPTOP_DEMO_PORT"
export CAIRN_CONFIG_HOME="$LAPTOP_DEMO_WORK/cairn-home"
