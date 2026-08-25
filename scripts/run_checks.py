#!/usr/bin/env python3
"""Run every offline check and report one verdict.

    python scripts/run_checks.py              # all of them
    python scripts/run_checks.py app cli      # just those two
    python scripts/run_checks.py --list

Each check is a standalone program, not a pytest module. That is a deliberate choice and
worth explaining, because it looks like an omission: six of these checks fit real models,
four of those also save an artifact and re-load it from disk, and five build their own
directory under ``/tmp`` (``make clean-runs`` removes all five). Under pytest that becomes
a fixture graph where one session's leftovers decide whether the next assertion passes. As
scripts they can only lie to themselves, and each prints the numbers it measured, so a
failure is read rather than debugged.

``wiring`` runs first and is the odd one out: it reads the source instead of running it,
and catches what a green run cannot - a keyword argument renamed on one side only, a
``cfg.section.field`` typo in a branch nothing takes, a ``--flag`` the README promises and
the parser never had.

Nothing here needs scikit-learn, XGBoost, TensorFlow, SHAP, Optuna or Streamlit. The
model backends are replaced by numpy doubles registered from ``tests/_stubs`` and
Streamlit by a recording double, so this suite runs anywhere Python, numpy and pandas
run - which is the point. It checks the *plumbing*: that the pipeline's stages agree
about shapes, that a saved bundle predicts what the in-memory model predicted, that the
CLI's exit codes mean what they say, that the app's script executes. It cannot check
whether XGBoost is any good at intrusion detection. That is what the Colab notebook is
for, and the two are not substitutes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

#: Ordered so that a first failure points at the earliest broken stage: the static check
#: before anything is run, then the pipeline, then the things that consume its output,
#: then the two user-facing surfaces. They share no state - the ones that need an artifact
#: train their own - so any subset is valid and the order is a convenience, not a
#: dependency.
SUITE = [
    ("wiring", "the source itself: signatures, config fields, exports, documented flags"),
    ("train", "the pipeline end to end: split, select, balance, fit, save, round-trip"),
    ("inference", "scoring messy input, and reproducing the training path exactly"),
    ("explain", "attributions, and what they mean when SHAP is not installed"),
    ("narrate", "the sentences: severity, recommended action, and the report prose"),
    ("cli", "every command as a user types it, with its exit code and its files"),
    ("app", "the Streamlit logic module, and the script itself, actually executed"),
]

#: Most checks are ``tests/check_NAME.py``. The static one lives in ``scripts/`` because
#: it reads the tree instead of running it, and it goes first because a signature
#: mismatch it finds explains failures the runtime checks would otherwise report as
#: something less obvious.
ELSEWHERE = {"wiring": ROOT / "scripts" / "check_wiring.py"}


def script_for(name: str) -> Path:
    return ELSEWHERE.get(name, TESTS / f"check_{name}.py")


def run_one(name: str, *, timeout: int, verbose: bool) -> tuple[bool, float, str]:
    """Run one check. Returns (passed, seconds, last meaningful line)."""
    path = script_for(name)
    started = time.time()
    try:
        done = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT), timeout=timeout,
            capture_output=not verbose, text=True,
        )
    except subprocess.TimeoutExpired:
        return False, time.time() - started, f"timed out after {timeout}s"

    elapsed = time.time() - started
    if verbose:
        return done.returncode == 0, elapsed, ""

    output = (done.stdout or "") + (done.stderr or "")
    lines = [l.rstrip() for l in output.splitlines() if l.strip()]
    if done.returncode == 0:
        tail = next((l for l in reversed(lines) if "PASSED" in l), lines[-1] if lines else "")
        return True, elapsed, tail.strip()

    # On failure the useful line is the assertion, which is the last line of the
    # traceback - not the last line of output, because a bare `assert x, "why"` prints
    # the message there and an unexpected exception prints its type there.
    print(f"\n{'-' * 78}\n{path.name} failed (exit {done.returncode}). "
          f"Last 40 lines:\n{'-' * 78}")
    for line in lines[-40:]:
        print("  " + line)
    return False, elapsed, lines[-1].strip() if lines else "no output"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", metavar="NAME",
                        help="checks to run (default: all, in dependency order)")
    parser.add_argument("--list", action="store_true", help="show the suite and exit")
    parser.add_argument("--timeout", type=int, default=600, metavar="SECONDS",
                        help="per check (default 600)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream each check's output instead of summarising")
    args = parser.parse_args(argv)

    if args.list:
        for name, blurb in SUITE:
            print(f"  {name:<10} {blurb}")
        return 0

    known = [name for name, _ in SUITE]
    chosen = args.names or known
    unknown = [n for n in chosen if n not in known]
    if unknown:
        print(f"unknown check(s): {', '.join(unknown)}\nknown: {', '.join(known)}",
              file=sys.stderr)
        return 2
    missing = [n for n in chosen if not script_for(n).exists()]
    if missing:
        print(f"missing file(s): {', '.join(str(script_for(n)) for n in missing)}",
              file=sys.stderr)
        return 2

    print(f"running {len(chosen)} check(s) with {sys.executable.split('/')[-1]} "
          f"{sys.version.split()[0]}\n")
    results = []
    for name in chosen:
        print(f"  {name:<10} ", end="", flush=True)
        passed, elapsed, tail = run_one(name, timeout=args.timeout, verbose=args.verbose)
        results.append((name, passed, elapsed))
        if not args.verbose:
            mark = "ok  " if passed else "FAIL"
            print(f"{mark} {elapsed:6.1f}s   {tail[:80]}")

    failed = [name for name, passed, _ in results if not passed]
    total = sum(seconds for _, _, seconds in results)
    print(f"\n{'=' * 78}")
    if failed:
        print(f"{len(failed)} of {len(results)} FAILED: {', '.join(failed)}   "
              f"({total:.0f}s)")
        print("=" * 78)
        return 1
    print(f"all {len(results)} check(s) passed in {total:.0f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
