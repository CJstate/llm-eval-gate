#!/usr/bin/env bash
# One-command EvalGuard demo. A regression exit code (1) is the expected result.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

set +e
python -m evalguard demo/baseline.json demo/candidate.json --color always
status=$?
set -e

if [ "$status" -eq 1 ]; then
  echo
  echo "Demo complete: EvalGuard found the GSM8K regression and exited 1 as expected."
  exit 0
fi

echo "Demo failed: expected exit 1 (regression), got $status." >&2
exit "$status"
