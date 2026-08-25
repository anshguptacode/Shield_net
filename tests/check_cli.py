"""Run every CLI command as a user would, and check the exit codes and the files.

A CLI is the one part of a project that cannot be tested by importing it: the failure
modes are wrong flag names, a subcommand that forgets to pass an argument through, and
an exception that escapes as a traceback where a sentence was needed. So this drives
``main(argv)`` directly - no subprocess, so a stack trace still points at the bug - and
asserts on what landed on disk.
"""
import contextlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Derived from this file, never hardcoded: an absolute path here would make a copy of the
# project silently import the original, so the copy would pass while being untested.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/_stubs"))
import stub_models; stub_models.register()

from shieldnet.cli import build_parser, main

WORK = Path("/tmp/shieldnet_cli")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
BASE = ["--root", str(WORK), "--log-level", "WARNING", "--traceback"]


def run(*argv, expect=0, show=0):
    """Call the CLI, capture stdout+stderr, assert the exit code, return the text."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = main(list(argv))
    except SystemExit as exc:
        # argparse exits instead of returning. Without this the script would die with
        # argparse's message locked inside the redirected buffer - which is exactly how
        # an unknown flag in this file looked the first time: exit code 2, no output.
        raise AssertionError(
            f"argparse rejected `shieldnet {' '.join(argv)}` (exit {exc.code}):\n"
            + buf.getvalue()[-800:]) from None
    text = buf.getvalue()
    label = " ".join(Path(a).name if a.startswith("/tmp") else a for a in argv)
    print(f"\n$ shieldnet {label}\n  -> exit {code}, {len(text):,} chars of output")
    if show:
        body = [l for l in text.splitlines() if l.strip()]
        for line in body[:show]:
            print("  | " + line[:110])
        if len(body) > show:
            print(f"  | ... {len(body) - show} more line(s)")
    assert code == expect, f"expected exit {expect}, got {code}\n{text[-2000:]}"
    return text


# ------------------------------------------------------------------ 1. the parser
print("=" * 78 + "\n=== 1. the parser itself ===\n" + "=" * 78)
parser = build_parser()
commands = sorted(parser._subparsers._group_actions[0].choices)
print(f"  {len(commands)} command(s): {', '.join(commands)}")
for expected in ("doctor", "models", "download", "prepare", "train", "info",
                 "predict", "evaluate", "explain", "serve", "config"):
    assert expected in commands, expected
# Every command must accept the common options - the whole point of attaching them to
# each sub-parser instead of the parent.
for name in commands:
    flags = {s for a in parser._subparsers._group_actions[0].choices[name]._actions
             for s in a.option_strings}
    for flag in ("--root", "--seed", "--log-level", "--quiet", "--traceback"):
        assert flag in flags, f"{name} is missing {flag}"
print("  every command accepts --root/--seed/--log-level/--quiet/--traceback")

# No arguments at all is a usage error, not a crash and not a silent success.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    assert main([]) == 2
assert "usage: shieldnet" in buf.getvalue()
print("  bare `shieldnet` prints usage and exits 2")

# ------------------------------------------------------------------ 2. doctor
print("\n" + "=" * 78 + "\n=== 2. doctor and models ===\n" + "=" * 78)
text = run("doctor", *BASE, show=6)
for section in ("libraries", "paths", "models", "schema", "data", "kaggle"):
    assert section in text, section
assert "every class has a narration profile" in text, \
    "check_profiles() reported gaps, or doctor stopped calling it"
assert "no raw CSVs" in text and "no trained artifact yet" in text
assert "shieldnet download" in text, "doctor does not say how to fix an empty data dir"
print("  doctor reports six sections and tells an empty project what to do next")

text = run("models", *BASE)
assert "stub_softmax" in text and "logistic_regression" in text
assert "unavailable" in text, "no model is marked unavailable in a bare environment"
lean = run("models", "--available", *BASE)
assert "unavailable" not in lean
print(f"  models: {len(text.splitlines())} rows, "
      f"{len(lean.splitlines())} of them runnable here")

# ------------------------------------------------------------------ 3. config
print("\n" + "=" * 78 + "\n=== 3. config resolution ===\n" + "=" * 78)
text = run("config", *BASE, "--seed", "7", "--features", "12",
           "--models", "stub_softmax,stub_centroid", "--no-tune")
resolved = json.loads(text)
assert resolved["seed"] == 7
assert resolved["features"]["n_features"] == 12
assert resolved["train"]["models"] == ["stub_softmax", "stub_centroid"]
assert resolved["train"]["primary"] == "stub_softmax", "primary did not follow --models"
assert resolved["tune"]["enabled"] is False
assert Path(resolved["paths"]["root"]) == WORK
print("  --seed/--features/--models/--no-tune all reach the resolved config")

cfg_file = WORK / "custom.json"
run("config", *BASE, "--output", str(cfg_file))
assert cfg_file.exists()
saved = json.loads(cfg_file.read_text())
saved["features"]["n_features"] = 14
saved["train"]["models"] = ["stub_softmax"]
saved["train"]["primary"] = "stub_softmax"
cfg_file.write_text(json.dumps(saved))
# A file supplies the base; flags still win over it.
text = run("config", "--config", str(cfg_file), *BASE, "--seed", "99")
merged = json.loads(text)
assert merged["features"]["n_features"] == 14, "the config file was ignored"
assert merged["seed"] == 99, "the flag did not override the file"
print("  a config file sets the base and command-line flags override it")

# ------------------------------------------------------------------ 4. prepare
print("\n" + "=" * 78 + "\n=== 4. prepare ===\n" + "=" * 78)
sample = WORK / "sample.csv"
text = run("prepare", "--config", str(cfg_file), *BASE, "--synthetic", "12000",
           "--sample", str(sample), "--sample-rows", "800", show=4)
assert sample.exists()
frame = pd.read_csv(sample)
print(f"  wrote {sample.name}: {frame.shape[0]:,} x {frame.shape[1]}")
assert len(frame) == 800
assert "Label" in frame.columns
assert frame["Label"].nunique() > 1, "the sample is single-class and useless as a test"
assert "shieldnet evaluate --input" in text, "prepare does not say what to do with it"

# ------------------------------------------------------------------ 5. train
print("\n" + "=" * 78 + "\n=== 5. train ===\n" + "=" * 78)
text = run("train", "--config", str(cfg_file), *BASE, "--synthetic", "12000",
           "--no-tune", "--no-explain", show=8)
artifacts = WORK / "artifacts"
assert (artifacts / "bundle.joblib").exists()
assert (artifacts / "manifest.json").exists()
assert "artifact:" in text and "next:" in text
manifest = json.loads((artifacts / "manifest.json").read_text())
print(f"  shipped {manifest['model_name']}: {manifest['n_features']} features, "
      f"macro F1 {manifest['metrics']['macro_f1']:.4f}")
assert manifest["n_features"] == 14, manifest["n_features"]
assert manifest["metadata"]["synthetic"] is True, \
    "a synthetic run must be recorded as synthetic in the manifest"

# ------------------------------------------------------------------ 6. info
print("\n" + "=" * 78 + "\n=== 6. info ===\n" + "=" * 78)
text = run("info", *BASE, show=5)
assert manifest["model_name"] in text
assert "features (14), in selection order:" in text
assert "classes (13):" in text
assert "recall" in text, "per-class recall is missing from info"
assert "SYNTHETIC" in text, "info does not warn that this artifact is synthetic"
with_json = run("info", *BASE, "--json")
assert '"bundle_sha256_12"' in with_json
print("  info lists features in selection order, per-class recall and provenance")

# ------------------------------------------------------------------ 7. predict
print("\n" + "=" * 78 + "\n=== 7. predict ===\n" + "=" * 78)
verdicts = WORK / "verdicts.csv"
summary_json = WORK / "run.json"
text = run("predict", *BASE, "--input", str(sample), "--output", str(verdicts),
           "--top-k", "3", "--probabilities", "--json", str(summary_json),
           "--chunk-rows", "250", show=6)
out = pd.read_csv(verdicts)
print(f"  verdicts.csv: {out.shape[0]:,} x {out.shape[1]}")
assert len(out) == len(frame), "predict changed the row count"
assert list(out["row"]) == list(range(len(frame)))
assert {"prediction", "confidence", "attack_probability", "is_attack", "severity",
        "alt1_class", "alt2_class"} <= set(out.columns)
assert sum(c.startswith("p(") for c in out.columns) == 13
assert "no row was dropped" in text
assert json.loads(summary_json.read_text())["rows"] == len(frame)

# The threshold has to reach the output, not just the summary line.
loose = WORK / "loose.csv"
run("predict", *BASE, "--input", str(sample), "--output", str(loose),
    "--threshold", "0.05", "--quiet")
strict = WORK / "strict.csv"
run("predict", *BASE, "--input", str(sample), "--output", str(strict),
    "--threshold", "0.95", "--quiet")
n_loose = int(pd.read_csv(loose)["is_attack"].sum())
n_strict = int(pd.read_csv(strict)["is_attack"].sum())
print(f"  --threshold 0.05 flags {n_loose:,}; --threshold 0.95 flags {n_strict:,}")
assert n_loose > n_strict, "--threshold had no effect on the written verdicts"

# ------------------------------------------------------------------ 8. evaluate
print("\n" + "=" * 78 + "\n=== 8. evaluate ===\n" + "=" * 78)
metrics_json = WORK / "metrics.json"
figure = WORK / "confusion.png"
text = run("evaluate", *BASE, "--input", str(sample), "--json", str(metrics_json),
           "--figure", str(figure), "--sweep", show=6)
metrics = json.loads(metrics_json.read_text())
# The scalars live under "overall" rather than at the top level, which is the right
# shape: it keeps `accuracy` from sitting beside `class_names` as though the two were
# the same kind of thing, and it means a consumer can iterate every headline metric
# without a hand-maintained list of which keys are numbers.
overall = metrics["overall"]
print(f"  macro F1 {overall['macro_f1']:.4f}, accuracy {overall['accuracy']:.4f}, "
      f"{len(metrics['per_class'])} classes")
assert metrics["split"] == "uploaded"
assert metrics["n_rows"] == len(frame)
assert len(metrics["class_names"]) == 13
assert metrics["confusion"] and metrics["binary"], "confusion/binary blocks are missing"
assert metrics["threshold_sweep"], "--sweep did not reach the JSON"
# The written numbers must be the ones printed, not a second, differently-rounded run.
assert f"{overall['macro_f1']:.4f}" in text, "the JSON and the console disagree"
assert figure.exists() and figure.stat().st_size > 10_000
assert "Confusion" in text or "confusion" in text
assert "threshold" in text.lower(), "--sweep printed no sweep"
# The narration must be prose, not a metric dump.
assert "macro F1" in text

# ------------------------------------------------------------------ 9. explain
print("\n" + "=" * 78 + "\n=== 9. explain ===\n" + "=" * 78)
text = run("explain", *BASE, "--input", str(sample), "--row", "17", "--top", "5",
           show=12)
assert "row 17:" in text
assert "ground truth:" in text
assert "top classes:" in text
manual = run("explain", *BASE, "--values",
             '{"Flow Duration": 1200000, "Total Fwd Packets": 8}', show=6)
assert "row 0:" in manual

# ------------------------------------------------------------------ 10. serve
print("\n" + "=" * 78 + "\n=== 10. serve hands over correctly ===\n" + "=" * 78)
# `serve` is the one command that cannot be run for real in a check: it blocks until the
# browser is closed. So intercept the hand-over and assert on the three things that are
# invisible until someone is sitting in front of the app.
#
# The working directory is the one that bit. Streamlit resolves .streamlit/config.toml
# against the working directory, not against the script path, so launching from the
# project root loaded none of app/.streamlit/config.toml - and maxUploadSize silently
# stayed at Streamlit's 200 MB default while that file asked for 400. The largest raw
# day-file is on the wrong side of 200 MB, so the upload most likely to be tried was the
# one being rejected by a limit the repo had already raised.
import types                                                          # noqa: E402
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
from shieldnet import cli as cli_mod                                  # noqa: E402

handovers = []
_real_call = cli_mod.subprocess.call


def _record_call(cmd, **kwargs):
    handovers.append((list(cmd), kwargs))
    return 0


cli_mod.subprocess.call = _record_call
previous_cwd = Path.cwd()
try:
    # Chdir into WORK and pass a *relative* --artifacts. If serve forwards it unresolved,
    # the app - which starts somewhere else entirely - looks in app/artifacts and reports
    # a missing bundle that is sitting right there.
    os.chdir(WORK)
    text = run("serve", *BASE, "--artifacts", "artifacts", "--no-browser",
               "--port", "8599", show=4)
finally:
    os.chdir(previous_cwd)
    cli_mod.subprocess.call = _real_call

assert len(handovers) == 1, handovers
cmd, kwargs = handovers[0]
assert "cwd" in kwargs, (
    "serve handed over without setting a working directory, so streamlit inherits the "
    "shell's. app/.streamlit/config.toml is then never read: no theme, and maxUploadSize "
    "back at 200 MB while that file asks for 400.")
workdir = Path(kwargs["cwd"])
print(f"  cwd:    {workdir}")
print(f"  argv:   {' '.join(cmd[1:])}")
assert workdir == (ROOT / "app").resolve(), workdir
config_toml = workdir / ".streamlit" / "config.toml"
assert config_toml.exists(), (
    f"{config_toml} does not exist, so serve's working directory buys nothing. Either "
    "the config moved or the directory is wrong; both make the theme and maxUploadSize "
    "silently revert to Streamlit's defaults.")
assert (workdir / cmd[4]).resolve() == (ROOT / "app" / "streamlit_app.py").resolve(), cmd
assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"], cmd
assert cmd[5:7] == ["--server.port", "8599"], cmd
assert cmd[7:] == ["--server.headless", "true"], "--no-browser did not reach streamlit"
env = kwargs["env"]
served = Path(env["SHIELDNET_ARTIFACTS"])
print(f"  artifacts env: {served}")
assert served.is_absolute(), f"{served} is relative; the child starts in {workdir}"
assert served == (WORK / "artifacts").resolve(), served
assert str(ROOT / "src") in env["PYTHONPATH"].split(os.pathsep), env["PYTHONPATH"]
assert "directory:" in text and "artifacts:" in text, \
    "serve does not print where it is starting from"
print("  serve starts in app/ (where .streamlit/config.toml is), with an absolute "
      "artifact path")

# ------------------------------------------------------------------ 11. failures
print("\n" + "=" * 78 + "\n=== 11. the failure paths a user will hit ===\n" + "=" * 78)
plain = ["--root", str(WORK), "--log-level", "ERROR"]        # no --traceback

text = run("predict", *plain, "--input", str(WORK / "does_not_exist.csv"), expect=1)
assert "FileNotFoundError" in text and "--input at a CICFlowMeter CSV" in text
assert "Traceback" not in text, "a missing file printed a stack trace"
print("  a missing input file: one sentence, exit 1, no traceback")

empty_dir = WORK / "empty_artifacts"
empty_dir.mkdir()
text = run("info", *plain, "--artifacts", str(empty_dir), expect=1)
assert "shieldnet train" in text and "Traceback" not in text
print("  a missing artifact says to run `shieldnet train`, exit 1")

text = run("explain", *plain, "--input", str(sample), "--row", "999999", expect=1)
assert "outside the file's" in text and "Traceback" not in text
print("  an out-of-range --row explains the range, exit 1")

bad_csv = WORK / "not_flows.csv"
pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad_csv, index=False)
text = run("predict", *plain, "--input", str(bad_csv), expect=1)
assert "CICFlowMeter" in text and "Traceback" not in text
print("  a file with no flow features is refused with an explanation, exit 1")

# --traceback must genuinely restore the traceback, or debugging is impossible.
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        main(["predict", *BASE, "--input", str(WORK / "nope.csv")])
except FileNotFoundError:
    print("  --traceback re-raises instead of swallowing")
else:
    raise AssertionError("--traceback did not re-raise")

# An unknown command is argparse's job, and it must not look like our own error.
try:
    with contextlib.redirect_stderr(io.StringIO()) as err:
        main(["frobnicate"])
except SystemExit as exc:
    assert exc.code == 2
    assert "invalid choice" in err.getvalue()
    print("  an unknown command exits 2 with argparse's own message")

print("\nALL CLI CHECKS PASSED")
stub_models.unregister()
