# Security Policy

The extensions in this repository run inside cairn, a governance control plane:
operator packs touch datasets, model weights, credentials and paid compute, and
domain packs decide what a pipeline runs. A defect in them is a security defect
even when it looks like a bug. We would much rather hear about it privately first.

## Reporting a vulnerability

**Use [GitHub Private Vulnerability Reporting](https://github.com/cairndev-sh/cairn-extensions/security/advisories/new)**
— Security → Report a vulnerability. It keeps the report private until a fix
ships, and gives us a place to talk it through with you.

If that is unavailable, email **security@cairndev.sh** with `SECURITY` in the
subject. Reports sent to that address go to the maintainers.

Please do **not** open a public issue for a suspected vulnerability. A public
issue is visible to everyone the moment it is filed, including anyone who would
use it.

A vulnerability in the engine itself (dispatch, the approval gate, the control
plane, storage) belongs to [cairndev-sh/cairn](https://github.com/cairndev-sh/cairn/security/advisories/new).
If you are unsure which repository it is, report it in either; we will move it.

**What to include:** what you did, what happened, what you expected, and the
package name and version (`cairn-operator-…`, `cairn-contracts-…`,
`cairn-pack-…`) or the commit. A minimal reproduction is worth more than a long
report. If you are unsure whether something counts, send it anyway — a false
alarm costs us ten minutes.

**What to expect:** acknowledgement within 3 working days, an assessment within
10, and a fix or a written explanation of why it is not one. We will credit you
in the advisory unless you ask us not to.

cairn is a young project maintained by a very small team. We do not run a bug
bounty and cannot pay for reports.

## Scope

The parts where a defect matters most:

- **An operator that does more than it declares** — a side effect, network call
  or spend that its manifest's risk level or approval requirement does not
  cover, so the engine's gate approves something other than what runs.
- **Secret handling** — credentials from `ctx.secrets` or a connection that
  land in logs, artifacts, run state, an error message, or a training or
  evaluation output.
- **Untrusted input** — a dataset, model, config or archive that makes an
  operator execute code (unsafe deserialisation, `trust_remote_code`), read or
  write outside its working directory, or fetch from somewhere it should not.
- **Data protection checks** — a PII scan, anonymiser, decontamination or
  dataset gate that reports a pass it did not earn.
- **A domain pack that runs ungoverned** — a pipeline, flow or profile that
  routes around an approval, budget or policy gate while reporting that it is
  governed.

## Out of scope

- Anything requiring credentials, physical access, or an already-compromised
  host.
- Vulnerabilities in third-party dependencies — report those upstream. Tell us
  anyway if an extension's usage makes an upstream issue materially worse.
- Deployments that disable their own guardrails. Running ungoverned on purpose is
  a configuration decision, not a vulnerability.
- Missing hardening with no demonstrated impact, unless you can show what it lets
  someone do.
- Denial of service by brute resource exhaustion.

## Supported versions

Each extension is its own distribution, released on its own tag
([RELEASING.md](RELEASING.md)). Fixes land on `main` and ship in the next
release of the affected distribution. They are not backported; upgrade that
distribution to its latest release on PyPI.

| Version | Supported |
| ------- | --------- |
| `main`  | yes       |
| the latest release of each distribution | yes: fixes ship in its next release |
| older releases | no: upgrade |

## Disclosure

We aim to fix, release, and publish an advisory before any public discussion.
If a report is still unfixed after 90 days, you are free to disclose it — we
would appreciate a heads-up, and if we are simply slow rather than silent, a
conversation about timing is usually easier than a deadline.
