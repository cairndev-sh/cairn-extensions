# Releasing

Every distribution in this repository (each `packages/*` and `packs/*` with a `pyproject.toml`)
publishes to PyPI on its own tag, `<distribution>/vX.Y.Z`. There is no repository-wide version
(ADR-0128 D3): the contract wheels are at 1.x and the packs at 0.1.x, and a tag names exactly
what it publishes. `.github/workflows/release.yml` does the work. It uses PyPI trusted
publishing (OIDC) and no API token, and it waits for the owner's approval on the `pypi`
environment before it uploads anything.

On a tag, the workflow:

1. **Plans.** `scripts/release.py plan` reads the distributions from the tree (none are listed
   by hand, so a new pack is picked up when it lands). It fails if the tag does not name a
   distribution in the tree at the version in its `pyproject.toml`, or if the tagged commit is
   not on `main`. It also fails if any distribution's directory changed since that
   distribution's last tag while its version stayed the same or went down. That check covers
   every distribution, not only the tagged one.
2. **Builds** an sdist and a wheel for every distribution with `uv build`.
3. **Tests** the built wheels on the released `cairndev` from PyPI with
   `scripts/test_wheels.sh --wheels dist`, so the files it tests are the files it publishes.
4. **Waits** for approval on the `pypi` environment.
5. **Publishes** the tagged distribution with `pypa/gh-action-pypi-publish`. Before uploading,
   it asks PyPI which of the distribution's files it already holds and skips those, logging
   `skip: <file> is already on PyPI`. If PyPI holds every file, it logs
   `skip: <name> <version> is already on PyPI (every file); nothing to upload` and uploads
   nothing. A rerun therefore never fails on an identical filename. Any PyPI answer other than
   200 or 404 fails the job rather than being read as "not there".
6. **Creates a GitHub Release** for the tag with that distribution's sdist and wheel.

## One-time setup (owner)

### PyPI: a trusted publisher per distribution

Go to <https://pypi.org/manage/account/publishing/> and add a **pending publisher** for each
distribution, with these fields:

| Field | Value |
|---|---|
| PyPI project name | the distribution name (list below) |
| Owner | `cairndev-sh` |
| Repository name | `cairn-extensions` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The distributions, as of this file (`python3 scripts/release.py plan` prints the current list
from the tree):

- `cairn-contracts-dataset`
- `cairn-contracts-dataset-stages`
- `cairn-contracts-training`
- `cairn-contracts-training-stages`
- `cairn-operator-dataset-checks`
- `cairn-operator-training`
- `cairn-pack-dataset-prep` (once its pull request lands)
- `cairn-pack-finetune` (once its pull request lands)

A pending publisher becomes the project's ordinary trusted publisher at the first upload.
Afterwards it appears under the project's *Settings → Publishing*. A pending publisher does not
reserve the name: register the publishers shortly before the first tags. Each later
distribution needs its own publisher before its first tag, or the upload fails with an OIDC
error. In that case, add the publisher and rerun the failed job.

### GitHub: the `pypi` environment

In *Settings → Environments*, create **`pypi`** with:

- **Required reviewers:** the owner. Leave *Prevent self-review* off, since the owner pushes the
  tag and approves the deployment.
- **Deployment branches and tags:** *Selected branches and tags*, with the tag rule `*/v*`.
- **No secrets.** Trusted publishing needs none.

Recommended: in *Settings → Rules → Rulesets*, add a tag ruleset for `*/v*` that lets only the
owner create, update or delete matching tags.

## Release a distribution

1. **Bump** `version` in the distribution's `pyproject.toml`. For a pack, bump `[pack] version`
   in its `pack.toml` to match, run `python3 scripts/gen_catalog.py`, and commit `catalog.json`.
   If another distribution pins this one exactly (the operator packs pin their contract wheels
   with `==`), bump that distribution's pin and version in the same pull request. Any file
   under a distribution's directory counts as a change, including tests and the README.
2. **Merge** the pull request to `main` with CI green.
3. **Dry run** from `main`. Use *Actions → release → Run workflow*, branch `main`,
   *publish* unchecked, or run:

   ```bash
   gh workflow run release.yml --repo cairndev-sh/cairn-extensions --ref main -f publish=false
   ```

   The dry run plans, builds and tests every distribution, and its summary lists what each
   distribution would upload (`upload: …`) or skip (`skip: …`). It publishes nothing and
   creates no release.
4. **Tag** the merged commit, and push one tag per command:

   ```bash
   git fetch origin && git tag -a cairn-contracts-dataset/v1.2.0 -m "cairn-contracts-dataset 1.2.0" origin/main
   git push origin cairn-contracts-dataset/v1.2.0
   ```

   GitHub starts no workflow run when a single push creates more than three tags, so never use
   `git push --tags` for a batch.
5. **Approve.** In the run, choose *Review deployments → pypi → Approve and deploy*.
6. **Check** the release in a clean venv: `pip install <distribution>==X.Y.Z`.

### The first release

Tag in dependency order, so that a `pip install` between runs never needs a distribution PyPI
does not have yet:

1. The contract wheels: `cairn-contracts-dataset`, `cairn-contracts-dataset-stages`,
   `cairn-contracts-training`, `cairn-contracts-training-stages`.
2. The operator packs: `cairn-operator-dataset-checks`, `cairn-operator-training`.
3. The domain packs: `cairn-pack-dataset-prep`, `cairn-pack-finetune`.

Each tag is its own run and needs its own approval. The version-bump check passes for every
distribution on the first release, because none has a tag yet.

## Reruns and mistakes

- **A failed run:** use *Re-run failed jobs*, or run the workflow from the tag (*Use workflow
  from → Tags → the tag*) with *publish* checked. Files PyPI already holds are skipped.
  `publish: true` from a branch is refused.
- **A wrong release:** never delete and re-push a tag for a version that reached PyPI, because
  PyPI never accepts the same filename twice. Yank the version on PyPI if it must not be
  installed, bump the version, and tag again.
- **The plan step failed:** the error names the distribution and the fix (for example
  `… changed since <tag> but its version is still X.Y.Z; bump it in …/pyproject.toml`).
  Fix it on `main`, then tag the new commit.
