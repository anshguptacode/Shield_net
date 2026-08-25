#!/usr/bin/env python3
"""Generate ``notebooks/ShieldNet_Colab.ipynb``.

The notebook is a build product, not a hand-edited file, and this is why: a .ipynb is
JSON with every line of every cell escaped into a string list, so editing one by hand
means counting quotes inside quotes, and a diff of two versions is unreadable. Writing
the cells as ordinary Python strings here means the notebook can be reviewed as prose
and regenerated deterministically.

    python scripts/build_colab_notebook.py            # writes the notebook
    python scripts/build_colab_notebook.py --check     # fails if it is out of date

The ``--check`` mode exists so that "someone edited the notebook in Colab and the change
is now unreproducible" shows up as a failure rather than as a surprise six weeks later.
Cell ids are derived from position, so regenerating produces a byte-identical file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "ShieldNet_Colab.ipynb"

#: (kind, source) pairs, in order. ``kind`` is "md" or "py".
CELLS: List[Tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("md", source.strip("\n")))


def py(source: str) -> None:
    CELLS.append(("py", source.strip("\n")))


# ===========================================================================
# 1. title
# ===========================================================================

md(r"""
# ShieldNet on Google Colab

Explainable multi-class network intrusion detection on CICIDS2017.

This notebook does one job: turn the raw Kaggle dataset into a trained, saved artifact
you can download and run locally. It is not a tutorial and it is not the project — the
project is the `shieldnet` package, and every cell below calls its command line. If a
cell here disagrees with `shieldnet --help`, the command line is right.

**Runtime:** *Runtime → Change runtime type → T4 GPU*. The GPU is used by exactly one
model, `cnn_bilstm`; the four classical models are CPU-bound, so a CPU runtime still
produces a complete result — it just trains the CNN slowly. Budget 35–60 minutes on a
T4 with `config/colab.yaml`, most of it tuning.

**What you end up with:** `artifacts/` (the model bundle, ~10–60 MB depending on which
model wins) and `reports/` (leaderboard, per-class metrics, feature ranking, three
figures, worked examples). The last cell zips both.

**Order matters.** Run the cells top to bottom the first time. After that they are
individually re-runnable: the cleaned dataset is cached and the tuning study is on disk,
so a reconnect costs you the cell you were in, not the run.
""")

# ===========================================================================
# 2. environment
# ===========================================================================

md(r"""
## 1. What this runtime actually gives us

Worth thirty seconds before a forty-minute job. The three things that decide whether the
run finishes are the GPU (or its absence), free RAM during consolidation, and free disk
for the dataset.
""")

py(r"""
import os
import platform
import shutil
import subprocess
import sys

# Python buffers stdout when it is a pipe, which is what `!command` gives it. Without
# this the stage banners arrive in silent 8 KB gulps and a 25-minute tuning cell looks
# identical to a hung one.
os.environ["PYTHONUNBUFFERED"] = "1"

IN_COLAB = os.path.isdir("/content")
print(f"python  {sys.version.split()[0]}  on {platform.platform()}")
print(f"colab   {'yes' if IN_COLAB else 'no (paths below assume /content; adjust)'}")


def sh(cmd, timeout=30):
    try:
        done = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout)
        return done.stdout.strip()
    except Exception as exc:                                   # noqa: BLE001
        return f"({type(exc).__name__})"


gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
if gpu and not gpu.startswith("("):
    print(f"gpu     {gpu}")
else:
    print("gpu     none. Runtime > Change runtime type > T4 GPU, or drop cnn_bilstm "
          "from --models later on.")

try:
    total_kb = int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1])
    print(f"ram     {total_kb / 1e6:,.1f} GB  (the streaming load used here peaks at a "
          f"few hundred MB; training is what wants the rest)")
except Exception:                                              # noqa: BLE001
    pass

usage = shutil.disk_usage("/content" if IN_COLAB else ".")
print(f"disk    {usage.free / 1e9:,.1f} GB free  (the download needs ~2.5 GB: the zip "
      f"plus the CSVs it unpacks to)")
if usage.free < 4e9:
    print("        ^ tight. Delete /content/sample_data, or restart the runtime.")
""")

# ===========================================================================
# 3. get the code
# ===========================================================================

md(r"""
## 2. Get the project into the runtime

Colab cannot see your laptop, so the code has to arrive somehow. The cell below tries
four routes and stops at the first that works. Fill in `REPO` **or** `DRIVE_COPY` if you
have one; otherwise leave both blank and you will get an upload box — give it a zip of
the project folder (the one containing `pyproject.toml`).

The cell is safe to re-run, which matters because a pip-induced session restart wipes
the working directory and `sys.path` but not the disk.
""")

py(r"""
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = ""          # e.g. "https://github.com/<YOUR GITHUB>/shieldnet.git"
DRIVE_COPY = ""    # e.g. "/content/drive/MyDrive/shieldnet"  (mount Drive first)
PROJECT = Path("/content/shieldnet")


def is_project(path) -> bool:
    # A directory is the project only if it has both halves of it.
    path = Path(path)
    return (path / "pyproject.toml").exists() and (path / "src" / "shieldnet").is_dir()


def find_upload() -> Path:
    # Unzip an uploaded archive and locate the project inside it.
    from google.colab import files
    print("Upload a zip of the project folder (Right-click the folder > Compress on "
          "macOS, Send to > Compressed folder on Windows).")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("nothing was uploaded")
    name = next(iter(uploaded))
    scratch = Path("/content/_shieldnet_upload")
    if scratch.exists():
        shutil.rmtree(scratch)
    with zipfile.ZipFile(name) as archive:
        archive.extractall(scratch)
    # A zip of a folder contains one top-level directory; a zip of a folder's *contents*
    # does not. Finding pyproject.toml covers both, and picking the shallowest match
    # avoids landing inside a nested copy or a __MACOSX sibling.
    found = sorted((p.parent for p in scratch.rglob("pyproject.toml")),
                   key=lambda p: len(p.parts))
    found = [p for p in found if is_project(p)]
    if not found:
        raise RuntimeError(f"no pyproject.toml with a src/shieldnet beside it in {name}")
    return found[0]


if is_project(PROJECT):
    print(f"already here: {PROJECT}")
elif DRIVE_COPY and is_project(DRIVE_COPY):
    print(f"copying from Drive: {DRIVE_COPY}")
    # Copied rather than used in place: training reads the config and writes nothing
    # here, but an editable install of a Drive-backed directory is slow to import.
    shutil.copytree(DRIVE_COPY, PROJECT, dirs_exist_ok=True)
elif REPO:
    print(f"cloning {REPO}")
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(PROJECT)], check=True)
else:
    source = find_upload()
    shutil.copytree(source, PROJECT, dirs_exist_ok=True)
    print(f"unpacked {source.name} -> {PROJECT}")

assert is_project(PROJECT), f"{PROJECT} still does not look like the project"
os.chdir(PROJECT)
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))
print("\ncwd:", Path.cwd())
print("contents:", ", ".join(sorted(p.name for p in PROJECT.iterdir()
                                    if not p.name.startswith("."))))
""")

# ===========================================================================
# 4. install
# ===========================================================================

md(r"""
## 3. Install

Colab already ships pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn and
TensorFlow. Asking pip to satisfy `.[all]` makes it consider reinstalling TensorFlow —
minutes of download for no change, and occasionally a version pair that then demands a
session restart. So: install the package, then only the wheels Colab is missing.

If pip finishes with a **RESTART SESSION** button, press it. Nothing is lost but the
working directory, so re-run section 2 and continue from here.
""")

py(r"""
%pip install -q -e .
%pip install -q imbalanced-learn xgboost lightgbm optuna shap streamlit plotly kaggle pyarrow
""")

# ===========================================================================
# 5. where the run lives
# ===========================================================================

md(r"""
## 4. Where the run lives

`SHIELDNET_ROOT` is the base for `data/`, `artifacts/` and `reports/`, so setting it
once here means no later cell has to pass `--root`.

Local disk, not Drive — deliberately. The Optuna study is a SQLite file under
`artifacts/`, and SQLite locking across Drive's FUSE mount is exactly the class of thing
that fails once, forty minutes in, with a `database is locked` that loses the study.
Section 11 copies the finished artifact to Drive, which is the part worth keeping.

This must run **before** anything imports `shieldnet`: the package resolves the root at
import time, so a later change would be read by the subprocesses and ignored in-process.
""")

py(r"""
import os
from pathlib import Path

ROOT = "/content/shieldnet-run"
os.environ["SHIELDNET_ROOT"] = ROOT
Path(ROOT).mkdir(parents=True, exist_ok=True)

CONFIG = "config/colab.yaml"     # the settings behind the reported numbers
SAMPLE = f"{ROOT}/data/samples/cicids2017_sample.csv"

for line in (f"root     {ROOT}",
             f"config   {CONFIG}",
             f"sample   {SAMPLE}"):
    print(line)
""")

py(r"""
!shieldnet doctor
""")

md(r"""
`doctor` is the one command that never fails the cell — it is diagnostic output, and a
missing SHAP should be a sentence, not a traceback. Read three things from it: every
model you intend to train says *available*, the paths point under
`/content/shieldnet-run`, and the Kaggle line is not yet happy (it will be after the
next section).

If `shieldnet` is not found, the install cell did not finish or the session restarted:
`python -m shieldnet doctor` works either way.
""")

# ===========================================================================
# 6. kaggle
# ===========================================================================

md(r"""
## 5. Kaggle credentials

Two routes. The **Secrets** panel (the key icon in the left sidebar) is the better one:
add `KAGGLE_USERNAME` and `KAGGLE_KEY`, tick *Notebook access*, and it survives every
restart from now on. Get both values from kaggle.com → your avatar → *Settings* → *API*
→ *Create New Token*, which downloads a `kaggle.json` containing them.

Otherwise the cell falls back to uploading that `kaggle.json`, which works fine and has
to be redone each session.
""")

py(r"""
import json
import os
from pathlib import Path


def from_secrets() -> bool:
    # Colab's Secrets panel. A silent no-op anywhere else.
    try:
        from google.colab import userdata
    except ImportError:
        return False
    try:
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        return True
    except Exception as exc:                                   # noqa: BLE001
        # SecretNotFoundError and NotebookAccessError both land here, and the
        # distinction does not change what you do about it.
        print(f"secrets unavailable ({type(exc).__name__}); falling back to upload")
        return False


def from_upload() -> bool:
    from google.colab import files
    print("Upload kaggle.json:")
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        print("that was not kaggle.json")
        return False
    target = Path.home() / ".kaggle"
    target.mkdir(parents=True, exist_ok=True)
    (target / "kaggle.json").write_bytes(uploaded["kaggle.json"])
    # The Kaggle client refuses a token any other user can read.
    (target / "kaggle.json").chmod(0o600)
    print(f"wrote {target / 'kaggle.json'} (mode 600)")
    return True


if not from_secrets():
    from_upload()

from shieldnet.data.download import ensure_credentials
print("\ncredentials via", ensure_credentials())
""")

md(r"""
## 6. Download CICIDS2017

Roughly 1 GB across eight day-wise CICFlowMeter CSVs, Monday through Friday. A few
minutes. The command skips the download when CSVs are already present, so re-running it
after a disconnect is free — pass `--force` if you actually want a fresh copy.

The default mirror is `chethuhn/network-intrusion-dataset`, with `cicdataset/cicids2017`
as a fallback. Some Kaggle datasets require you to accept their terms in a browser once
before the API will serve them; a `403` here means that, not a bad token.
""")

py(r"""
!shieldnet download --config $CONFIG
""")

md(r"""
Eight files is the number to see. Fewer means the mirror's layout changed — the CSVs
carry the day in their names (`Monday-WorkingHours.pcap_ISCX.csv` and siblings), and
`shieldnet doctor` names any that are missing from the expected set. Any CICIDS2017
mirror unzipped into `/content/shieldnet-run/data/raw` will do.
""")

# ===========================================================================
# 7. prepare
# ===========================================================================

md(r"""
## 7. Clean, split, cache

This stage is where most published CICIDS2017 numbers go wrong, so it is worth knowing
what it does: it repairs the header whitespace and the duplicated
`Fwd Header Length.1` column, turns infinities into missing values, drops exact
duplicate flows, folds the three ultra-rare labels (Heartbleed, Infiltration, Web Attack
– SQL Injection) into a single *Rare Attacks* class, then builds a stratified working
chunk that caps the four huge classes and keeps every minority row.

**Cleaning happens before the split, and everything fitted happens after it.** The
imputer, the scaler, the feature selector and SMOTE all see training rows only. The
cached parquet is keyed by a hash of the config that produced it, so changing a cap
invalidates it and changing a model does not.

`--sample` writes a labelled slice of the held-out test split. Section 10 scores it,
section 11 packs it into the zip, and the Streamlit app offers it in a dropdown so a demo
does not depend on finding a file.
""")

py(r"""
!shieldnet prepare --config $CONFIG --sample $SAMPLE --sample-rows 5000
""")

md(r"""
Read the class table it prints. The three numbers that matter: how many rows survived
deduplication, how many classes cleared `min_class_rows`, and the imbalance ratio
between BENIGN and the smallest class. That ratio is the reason macro F1 is the metric
and accuracy is not.
""")

# ===========================================================================
# 8. train
# ===========================================================================

md(r"""
## 8. Train

The long cell. Five models, tuned with Optuna, evaluated on validation, and the best
macro F1 wins the artifact:

| stage | roughly |
|---|---|
| consolidate the eight CSVs (cached after the first run) | 4 min |
| feature selection, 25 of 77, three methods, 5 stability resamples | 3 min |
| Optuna across the classical models (`tune.timeout_seconds: 2700`) | 25 min |
| `cnn_bilstm`, 40 epochs with early stopping | 10 min |
| SHAP on 2,000 rows | 4 min |

Two things make a disconnect survivable, and both are worth knowing before you need
them. The cleaned parquet is cached, so a re-run skips straight past consolidation. The
Optuna study lives in `artifacts/optuna.db`, so a re-run continues the search instead of
restarting it. **If Colab drops, just run this cell again.**

Selection is on the **validation** split, never on test — test is scored once, for the
winner, and that is the number that goes in a report. Accuracy is deliberately not the
criterion: predicting BENIGN for every row scores about 0.80 on this data while detecting
nothing at all.
""")

py(r"""
!shieldnet train --config $CONFIG --log-file $ROOT/reports/train.log
""")

md(r"""
## 9. Read the result

Four things to check, in this order, before trusting anything:

1. **The leaderboard spread.** If every model lands within a thousandth of the others,
   the task has been made too easy somewhere — usually a leak.
2. **Per-class recall, not the headline.** A 0.99 accuracy with Bot at 0.30 recall is a
   model that cannot see the attack you care about.
3. **False alarm rate.** At CICIDS2017's benign volume, 1% is thousands of flows a day.
4. **The `synthetic` flag is `false`.** If it is `true` you trained on the generator and
   the numbers describe nothing.
""")

py(r"""
!shieldnet info --json
""")

py(r"""
import pandas as pd
from IPython.display import display
from pathlib import Path

reports = Path(ROOT) / "reports"
board = pd.read_csv(reports / "leaderboard.csv")
print("leaderboard (validation scores decide the winner):")
display(board)

ranking = pd.read_csv(reports / "feature_ranking.csv")
print(f"\ntop 15 of {len(ranking)} ranked features:")
display(ranking.head(15))
""")

py(r"""
from IPython.display import Image, display
from pathlib import Path

figures = Path(ROOT) / "reports" / "figures"
for name in ("confusion.png", "importance.png", "per_class_importance.png"):
    path = figures / name
    if path.exists():
        print(f"\n{name}  ({path.stat().st_size / 1e3:,.0f} KB)")
        display(Image(str(path)))
    else:
        # per_class_importance is absent when attribution fell back to occlusion, which
        # is a per-prediction method and has no per-class global view.
        print(f"\n{name} was not produced")
""")

# ===========================================================================
# 9. evaluate + explain
# ===========================================================================

md(r"""
## 10. Score the held-out sample, and explain one flow

`evaluate` reloads the *saved* artifact and scores the labelled sample through the same
inference path the app uses. Its macro F1 should sit close to the test score printed by
training — they are different row counts, so not identical, but a large gap means the
bundle does not round-trip and nothing downstream can be trusted.

`--sweep` prints the threshold sweep, which is the honest way to talk about the
detection/false-alarm trade-off: one threshold is a point on that curve, not a property
of the model.
""")

py(r"""
# Built as a string rather than typed as one long `!` line: a shell escape spanning
# several lines is one stray backslash away from silently running half a command, and
# printing it first means the cell shows you exactly what ran.
evaluate = (f"shieldnet evaluate --input {SAMPLE} --sweep --fpr-budget 0.01"
            f" --figure {ROOT}/reports/figures/confusion_sample.png"
            f" --json {ROOT}/reports/evaluation_sample.json")
print(evaluate + "\n")
!$evaluate
""")

py(r"""
# One flow, end to end: verdict, runner-up classes, signed per-feature contributions,
# and a sentence saying what an analyst should do about it. Change --row to taste; the
# sample is ordered as the test split was, so low row numbers are not all benign.
!shieldnet explain --input $SAMPLE --row 41 --top 10
""")

# ===========================================================================
# 10. export
# ===========================================================================

md(r"""
## 11. Take the artifact home

Everything above lives on a runtime that will be deleted. Two ways out, and doing both
costs nothing: a zip you download, and a copy in Drive.

The zip is what you unpack into a local checkout's `artifacts/` and `reports/` to run
the Streamlit app. Nothing in it is machine-specific — it is a joblib bundle, a JSON
manifest, and the reports.
""")

py(r"""
import json
import shutil
from pathlib import Path

root = Path(ROOT)
stamp = ""
manifest = root / "artifacts" / "manifest.json"
if manifest.exists():
    meta = json.loads(manifest.read_text("utf-8"))
    stamp = f"_{meta.get('model_name', 'model')}"

bundle_dir = Path("/content/shieldnet_export")
if bundle_dir.exists():
    shutil.rmtree(bundle_dir)
bundle_dir.mkdir(parents=True)
for name in ("artifacts", "reports"):
    if (root / name).is_dir():
        shutil.copytree(root / name, bundle_dir / name)

# The labelled sample goes too, and the reason is worth stating: the app's "sample flow"
# dropdown looks for data/samples next to the artifacts directory, so leaving it out of
# the zip is what turns a working demo into "please locate a CSV" in front of an examiner.
samples = root / "data" / "samples"
if samples.is_dir() and any(samples.iterdir()):
    shutil.copytree(samples, bundle_dir / "data" / "samples")

# The Optuna study is the one thing not worth carrying: it is the search history, it is
# the largest file after the model, and it is useless without the same config.
for db in (bundle_dir / "artifacts").glob("optuna.db*"):
    print(f"leaving out {db.name} ({db.stat().st_size / 1e6:,.1f} MB of search history)")
    db.unlink()

archive = shutil.make_archive(f"/content/shieldnet{stamp}", "zip", bundle_dir)
size = Path(archive).stat().st_size / 1e6
print(f"\n{archive}  ({size:,.1f} MB)")
for path in sorted(bundle_dir.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(bundle_dir)}  ({path.stat().st_size / 1e3:,.0f} KB)")
""")

py(r"""
from google.colab import files
files.download(archive)     # a browser can stall above ~100 MB; use the Drive cell if so
""")

py(r"""
# Optional: a copy in Drive. Safe to do at the end - nothing writes here again, so the
# FUSE-mount problem that kept the run off Drive does not apply.
from pathlib import Path
import shutil

from google.colab import drive
drive.mount("/content/drive")

destination = Path("/content/drive/MyDrive/shieldnet_runs")
destination.mkdir(parents=True, exist_ok=True)
copied = shutil.copy(archive, destination)
print("copied to", copied)
""")

# ===========================================================================
# 11. running the app
# ===========================================================================

md(r"""
## 12. Running the app

The Streamlit app is meant to run on your machine, against the artifact you just
downloaded:

```bash
unzip shieldnet_xgboost.zip -d .      # gives you artifacts/ and reports/
pip install -e ".[all]"
shieldnet serve                        # http://localhost:8501
```

`serve` exports the artifact path and starts Streamlit for you, which is the only reason
to prefer it over `streamlit run app/streamlit_app.py`.

Serving from Colab is possible but not recommended for a demo: Streamlit needs a public
tunnel, the usual `npx localtunnel --port 8501` now asks visitors for a password that is
the runtime's public IP, and none of that is something you want to be discovering in
front of an audience. Download the artifact instead — it is 15 seconds of unzip against
a tunnel that may or may not come up.
""")

# ===========================================================================
# 12. troubleshooting
# ===========================================================================

md(r"""
## 13. When it goes wrong

| symptom | what it actually is | what to do |
|---|---|---|
| `shieldnet: command not found` | install cell did not finish, or the session restarted | re-run sections 2–3; `python -m shieldnet …` works regardless |
| `CredentialsError: No Kaggle credentials found` | secrets exist but *Notebook access* is unticked, or the upload was skipped | re-run section 5 |
| Kaggle `403 Forbidden` | the dataset's terms have not been accepted by your account | open the dataset page in a browser, click Download once, retry |
| download reports success, no CSVs appear | the mirror's layout changed | unzip any CICIDS2017 mirror into `/content/shieldnet-run/data/raw` by hand |
| runtime crashes during *prepare* | consolidation peak exceeded RAM | lower `data.read_chunk_rows` in `config/colab.yaml`, or use a high-RAM runtime |
| Colab disconnects mid-tuning | nothing is lost | re-run the train cell; the parquet cache and `artifacts/optuna.db` resume it |
| `MissingDependency: cnn_bilstm needs tensorflow` | CPU runtime without TF | `pip install tensorflow`, or `--models xgboost,lightgbm,random_forest,logistic_regression` |
| training says attributions came from occlusion | `shap` is absent, or the winner is not tree-based | fine, and the app says which method was used — install `shap` if you want the tree path |
| accuracy 0.99, macro F1 0.6 | the class imbalance, working as designed | read per-class recall; that is what the imbalance costs you |
| `database is locked` | the run root is on Drive | keep `SHIELDNET_ROOT` on local disk (section 4) |

One rule for reading any of this: a number without its provenance is not a result. The
manifest records the seed, the row counts, the config hash and whether the data was
synthetic, and both `shieldnet info` and the app's sidebar print it. If those two
disagree about a number, the artifact is not the one you think it is.
""")


# ===========================================================================
# assembly
# ===========================================================================

def build() -> dict:
    """Assemble the notebook document."""
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        # nbformat wants a list of lines with their newlines kept, and no trailing
        # newline on the last one - otherwise editors show a phantom blank line at the
        # end of every cell.
        lines = source.splitlines(keepends=True)
        cell = {
            "cell_type": "markdown" if kind == "md" else "code",
            # Deterministic, so regenerating the notebook is a no-op diff. nbformat 4.5
            # requires ids and requires them unique; position gives both.
            "id": f"shieldnet-{index:02d}",
            "metadata": {},
            "source": lines,
        }
        if kind == "py":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            # Colab reads these two. `provenance: []` keeps it from accumulating a
            # revision history in the file, which would defeat --check.
            "colab": {"provenance": [], "toc_visible": True},
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    # Jupyter itself writes a trailing newline; matching that keeps `git diff` quiet
    # after a notebook has been opened and saved.
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the notebook on disk differs from this script")
    parser.add_argument("--output", default=str(TARGET))
    args = parser.parse_args(argv)

    text = render()
    out = Path(args.output)

    if args.check:
        if not out.exists():
            print(f"{out} does not exist; run: python {Path(__file__).name}")
            return 1
        if out.read_text("utf-8") != text:
            print(f"{out} is out of date. Regenerate it with:\n"
                  f"  python scripts/{Path(__file__).name}")
            return 1
        print(f"{out} is up to date ({len(CELLS)} cells)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    code = sum(1 for kind, _ in CELLS if kind == "py")
    print(f"wrote {out}: {len(CELLS)} cells ({code} code, {len(CELLS) - code} markdown), "
          f"{len(text) / 1e3:,.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
