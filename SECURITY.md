# Security policy

## Supported code

Security fixes are considered for the current main branch and the latest
published release, starting with 0.2.0. Older research snapshots in Git
history are not maintained.

## Report a vulnerability privately

For a vulnerability in the Python package, CLI, dependencies, or CI workflow,
use [GitHub private vulnerability reporting](https://github.com/rktechpassionate/RESEARCH-1/security/advisories/new)
if it is enabled for this repository. Include the affected version, a minimal
reproducer, expected and observed behavior, and the likely impact. Please do
not post exploit details in a public issue before the maintainer can assess
them. If private reporting is unavailable, open a public issue requesting a
private contact channel without including exploit details.

The maintainer will assess the report, coordinate a fix and disclosure where
appropriate, and credit reporters who want acknowledgment. No response-time
guarantee is implied.

## Mathematical counterexamples

A suspected mathematical error or numerical counterexample is normally a
public research issue, not a security vulnerability. Use the
[mathematical issue template](.github/ISSUE_TEMPLATE/mathematical-correction.yml)
with the equation, Fourier convention, parameter values, precision, and an
independent derivation or reproducer. If the counterexample also exposes a
security vulnerability, use private reporting first.
