"""Run EvalGuard with ``python -m evalguard``."""

from __future__ import annotations

import sys

from llm_eval_gate.cli import main

if __name__ == "__main__":
    sys.exit(main())
