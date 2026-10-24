#!/usr/bin/env bash
# Prove every wheel this repository builds against the public core alone (ADR-0128 D3).
#
#   scripts/test_wheels.sh                 # the RELEASED cairndev each wheel's pins select (PyPI)
#   scripts/test_wheels.sh --core main     # the core's main branch instead (the nightly job)
#   scripts/test_wheels.sh --wheels dist   # the wheels already built into dist/, not the source
#                                          # directories (the release job: the wheels it tests are
#                                          # the files it publishes)
#
# One fresh venv: every buildable member installed as a wheel (with its `test` extra when it
# declares one; with --wheels, that member's built wheel, which must be in DIR), then: nothing
# private is installed, the repository's own tests, the catalog is current, each wheel's tests,
# `cairn pack sync --check` per operator pack, `cairn pack check`, and the core's contract
# resolver over every installed contract wheel.
# The interpreter is $UV_PYTHON (default 3.13). Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { echo "usage: scripts/test_wheels.sh [--core release|main] [--wheels DIR]" >&2; exit 2; }
core=release
wheel_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --core) [ $# -ge 2 ] || usage; core="$2"; shift 2 ;;
    --wheels) [ $# -ge 2 ] || usage; wheel_dir="$2"; shift 2 ;;
    *) usage ;;
  esac
done
if [ -n "$wheel_dir" ] && [ ! -d "$wheel_dir" ]; then echo "test_wheels: --wheels $wheel_dir is not a directory" >&2; exit 2; fi
case "$core" in release|main) ;; *) echo "test_wheels: --core takes release or main, not $core" >&2; exit 2 ;; esac

# every member must be buildable: members.py fails on one that is not, instead of skipping it
member_list="$(python3 scripts/members.py)"
wheels=()
while IFS= read -r d; do [ -n "$d" ] && wheels+=("$d"); done <<< "$member_list"

venv="$PWD/.venv-ci"
rm -rf "$venv"
uv venv -q "$venv" --python "${UV_PYTHON:-3.13}"
export VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-ci-dummy}" CAIRN_STORAGE="" OBS_STORAGE=""

# the install requirement for one member: its source directory, or with --wheels its built wheel
# (exactly one, by its normalized name and version), with the `test` extra when it declares one
spec_of() {
  python - "$1" "$wheel_dir" <<'PY'
import pathlib, re, sys, tomllib
member, wheels = pathlib.Path(sys.argv[1]), sys.argv[2]
project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
extra = "[test]" if "test" in project.get("optional-dependencies", {}) else ""
if not wheels:
    print(f"./{member.as_posix()}{extra}")
    sys.exit()
stem = re.sub(r"[-_.]+", "_", project["name"]).lower() + "-" + project["version"]
found = sorted(pathlib.Path(wheels).glob(f"{stem}-*.whl"))
if len(found) != 1:
    sys.exit(f"test_wheels: expected one wheel {stem}-*.whl in {wheels}, found {len(found)}")
print(f"{project['name']}{extra} @ {found[0].resolve().as_uri()}")
PY
}
specs=()
for w in "${wheels[@]+"${wheels[@]}"}"; do
  spec="$(spec_of "$w")"
  specs+=("$spec")
done
if [ ${#specs[@]} -eq 0 ]; then
  echo "no wheels yet: installing the released cairndev only"
  uv pip install -q "cairndev>=0.2.0,<0.3" pytest pytest-asyncio
else
  uv pip install -q "${specs[@]}" pytest pytest-asyncio
fi
if [ "$core" = main ]; then
  uv pip install -q --reinstall-package cairndev "cairndev @ git+https://github.com/cairndev-sh/cairn@main"
fi
# the released-core job must really run on a released wheel: refuse a cairndev that came from a
# path, a git URL or a workspace source (direct_url.json exists only for those)
python - "$core" <<'PY'
import importlib.metadata as m, json, sys
d = m.distribution("cairndev"); direct = d.read_text("direct_url.json")
print("cairndev", d.version, "from", json.loads(direct)["url"] if direct else "the package index")
if sys.argv[1] == "release" and direct:
    sys.exit("test_wheels: the released-core job got cairndev from a direct URL, not the package index")
PY

python scripts/check_venv.py
python -m pytest tests -q --no-header -p no:cacheprovider
python scripts/gen_catalog.py --check
for w in "${wheels[@]+"${wheels[@]}"}"; do
  echo "== $w"
  if [ -d "$w/tests" ]; then python -m pytest "$w/tests" -q --no-header -p no:cacheprovider -o asyncio_mode=auto -m "not expensive"; fi
done
while IFS= read -r w; do
  [ -n "$w" ] && cairn pack sync --check "$w"
done <<< "$(python3 scripts/members.py --operator-packs)"
cairn pack check
# the launch door resolves every installed contract distribution; a wheel that is not data-only
# (for example one carrying *.dist-info/licenses/) is refused there, and `pack check` does not see it
python -c "from cairn.core.pack.contracts import resolve_contracts; resolve_contracts(); print('contracts resolve: ok')"
