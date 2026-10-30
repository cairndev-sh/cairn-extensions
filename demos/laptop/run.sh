#!/usr/bin/env bash
# demos/laptop/run.sh — a governed fine-tune, end to end, on this machine (see README.md).
#
#   demos/laptop/run.sh                       # interactive: you approve each human gate
#   demos/laptop/run.sh --auto-approve        # answers the human gates as `dev` (CI, or to just watch it run)
#   demos/laptop/run.sh --tiny --auto-approve # a throwaway 0.1M-parameter model, built offline
#   demos/laptop/run.sh --base-model <hub id or local model directory>
#
# Two declared pipelines in their own packs, joined by an event, nothing in between:
# you launch `dataset-prep/prepare`; when its data gate passes it emits `dataset.prepared` with
# the train/val/test versions (ADR-0161), and `finetune/sft-local` starts on that event by its own
# trigger (ADR-0131 D4), which this script turns on once (ADR-0160). Everything below is a
# `cairn` command you can type yourself. Run demos/laptop/install.sh once first.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

auto=0; tiny=0; base_model=""; timeout_s="${LAPTOP_DEMO_TIMEOUT:-3600}"
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; cat <<'USAGE'

options:
  --auto-approve        answer every human gate as `dev` instead of waiting for you
  --base-model X        a Hugging Face model id or a local model directory (default: sft-local's, SmolLM2-135M)
  --tiny                build a tiny random-weight model offline and use it
  -h, --help            this text
environment: LAPTOP_DEMO_PORT (8765), LAPTOP_DEMO_S3_PORT (5765), LAPTOP_DEMO_WORK (a new temp dir),
             LAPTOP_DEMO_TIMEOUT (seconds before giving up, 3600); see env.sh
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --auto-approve) auto=1; shift ;;
    --tiny) tiny=1; shift ;;
    --base-model) [ $# -ge 2 ] || { usage >&2; exit 2; }; base_model="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

export LAPTOP_DEMO_WORK="${LAPTOP_DEMO_WORK:-$(mktemp -d "${TMPDIR:-/tmp}/cairn-laptop-demo.XXXXXX")}"
mkdir -p "$LAPTOP_DEMO_WORK"
# shellcheck source=env.sh
. "$here/env.sh"

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'run.sh: %s\n' "$*" >&2; exit 1; }
command -v cairn >/dev/null || die "cairn is not on PATH: run demos/laptop/install.sh first"
command -v moto_server >/dev/null || die "moto_server is not on PATH: run demos/laptop/install.sh first"
t0=$SECONDS

pids=()
cleanup() {
  local p
  for p in ${pids[@]+"${pids[@]}"}; do kill "$p" 2>/dev/null || true; done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'echo; echo "interrupted: stopping the demo servers (work dir: $LAPTOP_DEMO_WORK)"; exit 130' INT TERM

port_free() { ! python - "$1" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(0.5)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}
port_free "$LAPTOP_DEMO_PORT" || die "port $LAPTOP_DEMO_PORT is in use (set LAPTOP_DEMO_PORT)"
port_free "$LAPTOP_DEMO_S3_PORT" || die "port $LAPTOP_DEMO_S3_PORT is in use (set LAPTOP_DEMO_S3_PORT)"

say "work dir: $LAPTOP_DEMO_WORK"
cd "$LAPTOP_DEMO_WORK"

if [ "$tiny" = 1 ]; then
  say "a tiny random-weight model, built offline"
  python "$repo/tests/fixtures/tiny_model.py" "$LAPTOP_DEMO_WORK/tiny-model" >/dev/null 2>&1 || die "could not build the tiny model"
  base_model="$LAPTOP_DEMO_WORK/tiny-model"
fi

say "a local S3 (moto) on :$LAPTOP_DEMO_S3_PORT and the control plane on :$LAPTOP_DEMO_PORT"
moto_server -H 127.0.0.1 -p "$LAPTOP_DEMO_S3_PORT" >moto.log 2>&1 &
pids+=("$!")
for _ in $(seq 1 30); do port_free "$LAPTOP_DEMO_S3_PORT" || break; sleep 0.5; done
port_free "$LAPTOP_DEMO_S3_PORT" && die "moto_server did not start (see $LAPTOP_DEMO_WORK/moto.log)"
python - <<'PY'
import os, boto3
boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"]).create_bucket(Bucket=os.environ["CAIRN_ARTIFACT_S3_BUCKET"])
PY
cairn dev --engine-only --port "$LAPTOP_DEMO_PORT" --db "$LAPTOP_DEMO_WORK/cairn.db" >server.log 2>&1 &
pids+=("$!")
ok=0
for _ in $(seq 1 90); do
  if cairn remote login -u dev -p dev -e "$CAIRN_URL" >/dev/null 2>&1; then ok=1; break; fi
  kill -0 "${pids[1]}" 2>/dev/null || die "the control plane exited (see $LAPTOP_DEMO_WORK/server.log)"
  sleep 1
done
[ "$ok" = 1 ] || die "the control plane did not come up (see $LAPTOP_DEMO_WORK/server.log)"
echo "    logged in as dev; policies: $(ls "$CAIRN_POLICIES_DIR" | tr '\n' ' ')"

say "the chain: finetune/sft-local starts on dataset.prepared — paused until a person turns it on"
cairn remote triggers
trigger="$(cairn remote triggers | awk '$2 == "event" && $4 == "pipeline:finetune/sft-local" {print $1; exit}')"
[ -n "$trigger" ] || die "sft-local's event trigger is not stored (is cairn-pack-finetune installed?)"
cairn remote enable-trigger "$trigger"

# One read the CLI does not print yet: a pipeline's status and the steps waiting on a person.
# (`cairn remote pipeline <id>` prints the full record as JSON; a `--watch` for a running
# pipeline would make this line go.)
state() {
  cairn remote pipeline "$1" | python -c '
import json, sys
p = json.load(sys.stdin); st = p.get("step_state") or {}
wait = [s for s, v in st.items() if str(v.get("status", "")).startswith("awaiting")]
print(p.get("status", ""), " ".join(wait))'
}

# Follow PIPELINE to the end, answering each human gate (you, or --auto-approve as dev).
drive() {
  local pid="$1" status waiting g ans deadline=$((SECONDS + timeout_s)) last=""
  while :; do
    [ "$SECONDS" -lt "$deadline" ] || die "gave up after ${timeout_s}s (LAPTOP_DEMO_TIMEOUT); pipeline $pid is still running"
    read -r status waiting <<<"$(state "$pid")"
    [ "$status $waiting" != "$last" ] && echo "    $status ${waiting:+(waiting: $waiting)}" && last="$status $waiting"
    case "$status" in
      done) return 0 ;;
      failed|cancelled)
        cairn remote findings "$pid" >&2 || true
        tail -n "${LAPTOP_DEMO_LOG_LINES:-60}" "$LAPTOP_DEMO_WORK/server.log" | sed 's/^/    | /' >&2
        die "pipeline $pid $status (logs: $LAPTOP_DEMO_WORK/server.log)" ;;
    esac
    for g in $waiting; do
      echo "  gate $g is asking a person. What it has measured so far:  cairn remote findings $pid"
      echo "  approve with:  . $here/env.sh && cairn remote advance $pid $g --approver dev"
      if [ "$auto" = 1 ]; then
        cairn remote advance "$pid" "$g" --approver dev >/dev/null && echo "  --auto-approve: approved $g as dev"
      elif [ -t 0 ]; then
        printf '  Enter approves as dev, n cancels the pipeline: '
        read -r ans || ans=""
        case "$ans" in
          n|N|no|NO) cairn remote cancel-pipeline "$pid" >/dev/null; die "cancelled by you at $g" ;;
          *) cairn remote advance "$pid" "$g" --approver dev >/dev/null; echo "  approved $g as dev" ;;
        esac
      fi
    done
    sleep 2
  done
}

say "1/2  data prep: you launch dataset-prep/prepare on the pack's 200-row sample"
sample="$repo/packs/cairn-pack-dataset-prep/data/sample-ifc.jsonl"
[ -f "$sample" ] || die "sample not found: $sample"
args=(-i "path=$sample" -i register=laptop-demo -i val_ratio=0.1 -i test_ratio=0.1)
[ -n "$base_model" ] && args+=(-i "base_model=$base_model")
out="$(cairn remote launch dataset-prep/prepare "${args[@]}")" || { echo "$out" >&2; die "launch refused"; }
prep="$(printf '%s\n' "$out" | sed -n 's/^pipeline_id=\([^ ]*\).*/\1/p' | head -1)"
[ -n "$prep" ] || { echo "$out" >&2; die "no pipeline id in the launch output"; }
echo "    pipeline $prep"
drive "$prep"

say "2/2  fine-tune: finetune/sft-local, started by the dataset.prepared event — not by this script"
ft=""
for _ in $(seq 1 60); do
  ft="$(cairn remote pipelines | awk '$NF == "sft-local" {print $1; exit}')"
  [ -n "$ft" ] && break
  sleep 1
done
[ -n "$ft" ] || die "no sft-local pipeline started on dataset.prepared (see $LAPTOP_DEMO_WORK/server.log)"
echo "    pipeline $ft, its trigger:"
cairn remote pipeline "$ft" | python -c 'import json,sys; print("      ", json.load(sys.stdin).get("trigger"))'
drive "$ft"

say "result"
cairn remote findings "$ft"

say "the audit log: every gate decision, the promotion and the trigger switch, hash-chained"
cairn audit verify

say "done in $((SECONDS - t0))s. Work dir (db, logs, audit): $LAPTOP_DEMO_WORK"
