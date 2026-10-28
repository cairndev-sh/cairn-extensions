#!/usr/bin/env bash
# demos/laptop/run.sh — a governed fine-tune, end to end, on this machine (see README.md).
#
#   demos/laptop/run.sh                       # interactive: you approve each human gate
#   demos/laptop/run.sh --auto-approve        # answers the human gates as `dev` (CI, or to just watch it run)
#   demos/laptop/run.sh --tiny --auto-approve # a throwaway 0.1M-parameter model, built offline: ~1 minute
#   demos/laptop/run.sh --base-model <hub id or local model directory>
#
# It starts a local S3 (moto) and the control plane (`cairn dev --engine-only`), logs in, runs
# `dataset-prep/prepare` on the sample, then `finetune/sft-local` on the versions that registered
# (data prep -> LoRA -> held-out loss -> automatic gate -> approval -> promote), prints the result
# and checks the audit log. Run demos/laptop/install.sh once first.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

auto=0; tiny=0; base_model=""; inputs=(); timeout_s="${LAPTOP_DEMO_TIMEOUT:-3600}"
usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; cat <<'USAGE'

options:
  --auto-approve        answer every human gate as `dev` instead of waiting for you
  --base-model X        a Hugging Face model id or a local model directory (default: the pipeline's, SmolLM2-135M)
  --tiny                build a tiny random-weight model offline and use it (implies a higher learning rate)
  -i, --input K=V       pass an input to finetune/sft-local (repeatable), e.g. -i max_steps=60
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
    -i|--input) [ $# -ge 2 ] || { usage >&2; exit 2; }; inputs+=("$2"); shift 2 ;;
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
view() { python "$here/pipeline_view.py" "$@"; }
t0=$SECONDS; lap() { printf '    (%ss)\n' "$((SECONDS - t0))"; }

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
  python "$here/tiny_model.py" "$LAPTOP_DEMO_WORK/tiny-model" >/dev/null 2>&1 || die "could not build the tiny model"
  base_model="$LAPTOP_DEMO_WORK/tiny-model"
  inputs+=("learning_rate=0.002")   # random weights need a bigger step to show a gain in 30 steps
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
lap

# drive PIPELINE_ID: follow it to the end, answering the human gates (you, or --auto-approve as dev).
drive() {
  local pid="$1" line status gates failed progress last="" g ans rc deadline=$((SECONDS + timeout_s)) notty=0
  while :; do
    [ "$SECONDS" -lt "$deadline" ] || die "gave up after ${timeout_s}s (LAPTOP_DEMO_TIMEOUT); pipeline $pid is still $status"
    line="$(view state "$pid")" || die "cannot read pipeline $pid from the control plane"
    IFS='|' read -r status gates failed progress <<<"$line"
    if [ "$progress" != "$last" ]; then printf '    %s\n' "$progress"; last="$progress"; fi
    case "$status" in
      done) return 0 ;;
      failed|cancelled) die "pipeline $pid $status${failed:+ at: $failed} (legs' logs: $LAPTOP_DEMO_WORK/server.log)" ;;
    esac
    if [ -n "$gates" ]; then
      for g in $gates; do
        echo
        echo "  A person is asked:"
        view card "$pid" "$g"
        echo "  approve with:  . $here/env.sh && cairn remote advance $pid $g --approver dev"
        if [ "$auto" = 1 ]; then
          cairn remote advance "$pid" "$g" --approver dev >/dev/null
          echo "  --auto-approve: approved $g as dev"
        elif [ "$notty" = 0 ] && [ -t 0 ]; then
          printf '  Press Enter to approve here as dev, type n to cancel the pipeline, or run the command above in another terminal. '
          while :; do
            rc=0; read -r -t 3 ans || rc=$?
            if [ "$rc" = 0 ]; then
              case "$ans" in
                n|N|no|NO) cairn remote cancel-pipeline "$pid" >/dev/null; die "cancelled by you at $g" ;;
                *) cairn remote advance "$pid" "$g" --approver dev >/dev/null; echo "  approved $g as dev"; break ;;
              esac
            elif [ "$rc" = 1 ]; then notty=1; break   # stdin closed: wait for the command above
            fi
            line="$(view state "$pid")"; IFS='|' read -r status gates failed progress <<<"$line"
            case " $gates " in *" $g "*) ;; *) echo "(approved elsewhere)"; break ;; esac
          done
        else
          echo "  waiting for that command (no terminal to ask on)..."
        fi
      done
    fi
    sleep 2
  done
}

launch() { # launch REF [-i k=v ...]: print the new pipeline id
  local out pid
  out="$(cairn remote launch "$@" 2>&1)" || { printf '%s\n' "$out" >&2; die "launch refused"; }
  pid="$(printf '%s\n' "$out" | sed -n 's/^pipeline_id=\([^ ]*\).*/\1/p' | head -1)"
  [ -n "$pid" ] || { printf '%s\n' "$out" >&2; die "no pipeline id in the launch output"; }
  printf '%s\n' "$pid"
}

say "1/2  data prep: dataset-prep/prepare on the pack's 200-row sample"
sample="$repo/packs/cairn-pack-dataset-prep/data/sample-ifc.jsonl"
[ -f "$sample" ] || die "sample not found: $sample"
prep="$(launch dataset-prep/prepare -i "path=$sample" -i register=laptop-demo -i val_ratio=0.1 -i test_ratio=0.1)"
echo "    pipeline $prep"
drive "$prep"
eval "$(view versions "$prep" | sed 's/^\([a-z]*\)=\(.*\)$/v_\1="\2"/')"
echo "    registered: $v_train  $v_val  $v_test"
lap

say "2/2  fine-tune: finetune/sft-local (LoRA, ${base_model:-the default small model}, held-out loss)"
args=(-i "train_version=$v_train" -i "val_version=$v_val" -i "test_version=$v_test")
[ -n "$base_model" ] && args+=(-i "base_model=$base_model")
for kv in ${inputs[@]+"${inputs[@]}"}; do args+=(-i "$kv"); done
ft="$(launch finetune/sft-local "${args[@]}")"
echo "    pipeline $ft"
echo "    follow it from another terminal:  . $here/env.sh && cairn remote pipeline $ft"
drive "$ft"
lap

say "result"
view summary "$ft"

say "who decided what (the pipeline records, the durable trail of the gates)"
view trail "$prep"
view trail "$ft"

say "audit log: $CAIRN_AUDIT_LOG"
if [ -s "$CAIRN_AUDIT_LOG" ]; then
  echo "    $(wc -l <"$CAIRN_AUDIT_LOG" | tr -d ' ') event(s) written by the audit sink"
  chain="$(cairn lineage --json | python -c 'import json,sys; print(json.load(sys.stdin)["chain_ok"])')" || chain="unreadable"
  echo "    cairn lineage --json | .chain_ok  ->  $chain"
  [ "$chain" = "True" ] || die "the audit chain did not verify"
else
  echo "    empty: no event of this run reached the audit sink (known limit, cairndev-sh/cairn#233);"
  echo "    the approvals above are on the pipelines' records: cairn remote pipeline $ft"
fi

say "done in $((SECONDS - t0))s. Work dir (db, logs, audit): $LAPTOP_DEMO_WORK"
