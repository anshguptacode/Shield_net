# ShieldNet

Explainable multi-class network intrusion detection on CICIDS2017.

ShieldNet takes a table of network flows — the kind CICFlowMeter produces from a packet
capture — and returns, for every flow, which of thirteen classes it belongs to, how
confident it is, which features drove that verdict, and what an analyst should do next.
It is a complete project rather than a notebook: one library, one command line, one web
app, and one Colab notebook that produces the model the other three consume.

The thing that makes intrusion detection on this dataset interesting is not the
classifier. It is that 80.3% of CICIDS2017 is benign traffic, so a model that predicts
BENIGN for every single flow scores 0.803 accuracy while detecting nothing at all. Most
of the design decisions below exist to stop that number from ever being reported as a
result.

---

## Quickstart

**Does it work at all?** No dataset, no GPU, no machine-learning libraries — numpy and
pandas are enough. This trains real models against numpy test doubles, saves artifacts,
reloads them, runs every CLI command and executes the Streamlit app against a recording
double:

```bash
pip install -e .          # numpy + pandas
make check                # seven checks, a minute or two
```

If that install fails — an old setuptools, a machine with no route to PyPI — run
`make check` anyway. The check scripts put `src/` on `sys.path` themselves, so the suite is
the one thing here that needs no installed package; it has been run from a pristine copy of
this tree with nothing installed and no bytecode cache, and passes. Until the install
works, read every `shieldnet <command>` below as `PYTHONPATH=src python -m shieldnet
<command>`, which is the same entry point the console script calls.

Do not work around that failure by disabling pip's build isolation. On setuptools older
than 61, `pip install . --no-build-isolation` does not fail — it ignores the entire
`[project]` table, builds a wheel named `UNKNOWN-0.0.0` with no modules in it, prints
"Successfully installed", and leaves you with no `shieldnet` command and no error message
to search for. Ubuntu 22.04 ships setuptools 59.6, and that is verbatim what it does there.
Upgrade the build tool instead — `pip install -U setuptools` — and see the note at the top
of `pyproject.toml` for why the floor is 64 and not a rounder number.

**A real model in five minutes**, still without the dataset. Synthetic flows are
generated with the real schema, the real class imbalance and plausible per-class
distributions, so the pipeline is exercised end to end:

```bash
pip install -e ".[ml]"
shieldnet train --synthetic 40000 --models logistic_regression,random_forest --no-tune
shieldnet info
shieldnet serve                    # needs .[app] as well
```

The app will tell you, in red, at the top of the sidebar, that the model behind those
numbers was trained on synthetic data. That warning is not decoration — see
[Provenance](#provenance).

**The real thing** needs the dataset and about an hour. Open
[`notebooks/ShieldNet_Colab.ipynb`](notebooks/ShieldNet_Colab.ipynb) in Google Colab,
select a T4 runtime, and run it top to bottom; it downloads CICIDS2017 from Kaggle,
trains five models, and hands you a zip. Unpack that into a local checkout and
`shieldnet serve`. Locally the same run is:

```bash
pip install -e ".[all]"
shieldnet download                          # ~1 GB from Kaggle, needs a token
shieldnet train --config config/colab.yaml  # 35-60 min on a T4, longer on CPU
shieldnet serve
```

---

## The four surfaces

| surface | entry point | what it is for |
|---|---|---|
| command line | `shieldnet <command>` | everything: doctor, download, prepare, train, info, models, config, predict, evaluate, explain, serve |
| web app | `shieldnet serve` → `app/streamlit_app.py` | scoring a capture and reading one verdict, interactively |
| notebook | `notebooks/ShieldNet_Colab.ipynb` | producing the artifact on a GPU you do not own |
| library | `import shieldnet` | if you want the pieces rather than the pipeline |

They share one code path on purpose. The app renders numbers computed in
`app/shieldnet_ui.py`; the notebook shells out to the CLI; the CLI parses arguments and
calls one library function per command. Nothing is reimplemented in two places, which is
why the app's macro F1 and `shieldnet info`'s macro F1 cannot drift apart.

Run `shieldnet doctor` first on any new machine. It reports library versions, resolved
paths, which model backends can actually run, whether the dataset is present, whether an
artifact exists, and whether Kaggle credentials are discoverable — and it never fails,
because a command whose job is to tell you what is missing should not crash when
something is missing.

---

## The dataset

CICIDS2017 is five days of traffic from a lab network, published by the Canadian
Institute for Cybersecurity, distributed as eight day-wise CSVs of CICFlowMeter output:
**2,830,743 flows, 79 columns, 15 labels.** Two of those columns are not features: the
label, and `Fwd Header Length.1`, which is an exact duplicate of `Fwd Header Length` and
is dropped on load. That leaves **77 usable features**. Eight of those are constant across
the whole dataset and carry no information: `Bwd PSH Flags`, `Bwd URG Flags`, and the six
`{Fwd,Bwd} Avg {Bytes/Bulk, Packets/Bulk, Bulk Rate}` columns, which CICFlowMeter emits
and never populates. They are listed in `schema.KNOWN_CONSTANT_FEATURES`, but the run
drops them by measuring variance on the train split, not by trusting that list — if a
documented-constant column does vary in your data, the log says so.

Three labels are too rare to model or to evaluate honestly:

| label | rows | share |
|---|---|---|
| Infiltration | 36 | 0.0013% |
| Web Attack – Sql Injection | 21 | 0.0007% |
| Heartbleed | 11 | 0.0004% |

Sixty-eight rows in 2.8 million. A stratified split gives Heartbleed two test rows, so
its recall can only be 0.0, 0.5 or 1.0 — a number with no useful precision that will
nonetheless move macro F1 by several points. ShieldNet folds all three into one
**Rare Attacks** class, giving the **13-class scheme** used throughout. That is a
reportable modelling decision, not a silent convenience: `shieldnet info` names the
merged class, and the app's class reference table says which labels went into it.

The remaining imbalance is still severe — 2.27M BENIGN against 652 Web Attack – XSS
flows, a ratio of about 3,500:1 — and that is the real problem the project is about.

---

## How a run works

```
download        Kaggle -> data/raw/*.csv                       (8 files, ~1 GB)
   |
prepare         consolidate -> clean -> cap -> split -> cache
   |            data/interim/*.parquet, keyed by the settings that shaped it - change
   |            a cap and it rebuilds, change a model and it is reused
   |
train           preprocess -> select -> balance -> fit -> tune -> evaluate -> explain
   |            artifacts/bundle.joblib + manifest.json, reports/*
   |
predict         artifacts/ + capture.csv -> verdicts.csv
evaluate        artifacts/ + labelled.csv -> metrics, confusion matrix, threshold sweep
explain         artifacts/ + one row -> signed per-feature contributions + a sentence
```

Cleaning fixes the header whitespace, drops the duplicated column, turns the infinities
that CICFlowMeter emits for zero-duration flows into missing values, removes exact-duplicate
flows (the count is reported by the run rather than asserted here — CICIDS2017 has a lot of
them, and how many depends on which mirror of the CSVs you downloaded), and merges the rare
labels. Then a stratified working chunk caps the four huge classes (150k BENIGN, 40k DoS
Hulk, 40k PortScan, 35k DDoS) and keeps every row of every other class — 304,616 rows, of
which 39,616 are minority traffic kept in full, so sampling never costs a minority class a
single flow.

The split is 20% test, 10% validation, 70% train — both fractions of the whole chunk, not
of what the previous split left — stratified, and carved out **before** anything is fitted.
On the default chunk that is 213,230 train, 30,463 validation, 60,923 test.

---

## The decisions that matter

**The leakage boundary.** Cleaning happens before the split; everything that learns
happens after it. The imputer's medians, the scaler's means, the correlation filter, the
feature ranking and SMOTE all see training rows only. Constant-feature detection measures
variance on the train split, not the whole dataset. SMOTE never runs inside
cross-validation folds and never touches validation or test. This is the single most
common way CICIDS2017 papers produce 0.999 scores that do not survive contact with new
traffic.

**Macro F1 on validation selects the model.** Not accuracy, which is unusable here for
the reason in the first paragraph. Not test, which is scored exactly once, for the winner,
after the choice is already fixed — you can see this in the leaderboard, where the test
columns are blank for every model except the selected one. That asymmetry is the point: a
table with a test score in every row invites the reader to sort by it, and selecting on
test by eye leaves no trace in the code. Log loss is the *tuning* objective because Optuna
needs something smooth, but it is not the selection metric: log loss
rewards being well-calibrated about the majority class, so a model that is beautifully
calibrated about benign traffic and blind to Bot can win on it. Macro F1 weights all
thirteen classes equally, which is the actual deployment requirement.

**No row is ever dropped at inference.** Row *i* of the output is the verdict on row *i*
of the input, always. Bad cells are repaired — infinities become missing values, missing
values become the training median — and the repairs are counted and reported next to the
verdicts rather than hidden. A detector that silently returns 9,998 verdicts for 10,000
flows has told you nothing about the two it swallowed, and those two are exactly the ones
worth looking at.

**`P(attack) ≥ threshold` is not the same question as `argmax ≠ BENIGN`.** The first asks
whether total attack probability clears a bar; the second asks which single class is most
likely. A flow can be 60% attack while BENIGN is still the largest individual class,
because that 60% is spread across five attack types. Both numbers appear on screen and
the app reconciles them explicitly, because two different attack counts in one interface
with no explanation is worse than either number alone.

**Explanations degrade honestly.** TreeSHAP where the winner is tree-based and `shap` is
installed; KernelSHAP where it fits; occlusion attribution otherwise. All three produce
signed per-feature contributions in the same shape, and every surface that shows one says
which method produced it. Contributions are reported on the scale they were computed on,
with the raw captured value shown when it can be recovered and labelled `standardised`
when it cannot — a z-score posing as a packet count is a lie the interface would be
telling on the model's behalf.

**Warnings are for things that are wrong.** A message that fires on every correct run is
a message nobody reads, which means the one that matters is invisible when it arrives.

### Provenance

Every artifact records the seed, the config hash, the row counts, the library versions,
the timestamp, the selection metric, and whether the data was synthetic. `shieldnet info`
prints it and the app puts it in the sidebar **above** the metrics, not below them,
with a red banner for a synthetic run. A score from a smoke test must be impossible to
mistake for a CICIDS2017 result; putting the provenance under the numbers would make that
mistake a matter of scroll position.

Saving is verified rather than assumed, and the verification is a gate rather than a note.
The bundle is written to `artifacts/.staging`, reloaded from there, and asked to reproduce
the in-memory model's predictions on 64 test rows. Only if the largest probability
difference is at or below 1e-5 are the files moved into `artifacts/` itself. Above it the
run fails and the staging directory is deleted, so the previous artifact is still the one
on disk — a bundle that cannot be restored must not be the thing the app finds, and
checking after the write rather than before would have made this a log line about a file
already in place.

---

## Layout

```
src/shieldnet/
  schema.py         the dataset as code: 77 features, 15 labels, the 13-class scheme,
                    a glossary, and the reference row counts
  config.py         one dataclass tree, loaded from YAML, overridable per flag
  logging_utils.py  the stage timers and row counts every module logs through
  data/
    download.py     Kaggle, with credentials explained rather than assumed
    load.py         the eight CSVs -> one clean frame
    chunk.py        the stratified working chunk, caps and all
    synthetic.py    the real schema with plausible distributions, for testing
  preprocess.py     impute, clip, scale - fitted on train only
  features.py       mutual information, chi2, RFE, and a stability measurement
  balance.py        SMOTE with an expansion ceiling, and class weights
  models/           eleven backends behind one interface: logistic regression, naive
                    Bayes, decision tree, random forest, extra trees, XGBoost, LightGBM,
                    MLP, CNN-1D, BiLSTM, CNN-BiLSTM. Each declares what it needs and is
                    skipped with a reason when that is missing.
  tune.py           Optuna, resumable, with a sqlite study
  train.py          the pipeline, the leaderboard, and what gets saved
  evaluate.py       per-class metrics, confusion matrix, threshold sweep, calibration
  explain.py        TreeSHAP / KernelSHAP / occlusion, one interface
  narrate.py        the attack profiles: severity 1-5, recommended action, report prose
  inference.py      the Detector: load a bundle, score anything, never drop a row
  persist.py        the bundle and its manifest
  cli.py            every command
app/
  shieldnet_ui.py   every number the app renders. Imports no Streamlit.
  streamlit_app.py  the widgets. Deliberately thin.
tests/
  check_*.py        six standalone programs that run the project, no ML libraries needed
  _stubs/           numpy model doubles, and a recording Streamlit
config/
  default.yaml      the documented reference settings
  fast.yaml         for iterating: small, quick, and honest about being both
  colab.yaml        the settings behind the reported numbers
scripts/
  run_checks.py     runs all seven checks, reports one verdict
  check_wiring.py   reads the source: signatures, config fields, documented flags
  build_colab_notebook.py   generates the notebook (see below)
```

---

## Configuration

Three files in `config/`, and flags override them. `shieldnet config --models xgboost
--tune` prints the fully resolved settings without running anything, which is both the
fastest way to check a flag did what you meant and the thing to paste into a report
appendix as a run's provenance.

Every setting in `config/default.yaml` carries a comment saying why it has the value it
has. If you change one thing, `features.n_features` (25 of 77) and `data.caps` are the
two that move the numbers most.

`SHIELDNET_ROOT` relocates `data/`, `artifacts/` and `reports/` wholesale, which is what
the Colab notebook uses. `SHIELDNET_ARTIFACTS` points the app at a specific bundle;
`shieldnet serve` sets it for you from `--root` or `--artifacts`.

---

## Checking it

```bash
make check           # all seven; a minute or two, most of it in `train`
make check-fast      # the quick ones, for an edit loop
make wiring          # the static checks alone, seconds
python scripts/run_checks.py --list
```

Each check prints its own wall clock, and the runner prints the total. The times move a
lot with the machine — six of the seven fit real models — so treat the numbers above as
an order of magnitude rather than a benchmark.

Seven standalone programs, not pytest modules. Six of them fit real models. Four of those
— `train`, `inference`, `cli`, `app` — also save an artifact and reload it from disk, each
in its own directory under `/tmp`; `explain` writes its two figures to a directory of its
own and `narrate` writes nothing at all. `make clean-runs` removes all five. Under pytest
this would become a fixture graph where one session's leftovers decide whether the next
assertion passes. As scripts they can only lie to themselves, and each prints the numbers
it measured, so a failure is read rather than debugged.

None of them needs scikit-learn, XGBoost, TensorFlow, SHAP, Optuna or Streamlit. Model
backends are numpy doubles registered from `tests/_stubs`; Streamlit is a recording double
that returns scripted widget answers, so `app/streamlit_app.py` is *executed* — eight
times, through a whole user journey — rather than merely imported. Any Streamlit API the
double has not modelled is served as a no-op **and** recorded, and the test fails on it,
because a silently swallowed typo would make a green run meaningless.

`scripts/check_wiring.py` runs first and is the odd one out: it reads the source rather
than running it. Executing the project can only ever check code that executes, and the
mistakes that reach a viva are the ones on paths nobody took — a keyword argument renamed
on one side only, a `cfg.features.n_featurs` typo in an error branch, a flag the
README promises and the parser never had. So it compiles every file, imports every module,
validates keyword arguments at every call site it can resolve unambiguously, checks every
`cfg.section.field` against the dataclasses, checks every `__all__`, confirms every
`config/*.yaml` loads and that `default.yaml` really is identical to the defaults it claims
to restate, validates every command-line flag written anywhere in the tree — README,
Makefile, YAML comments, docstrings, the notebook — against argparse itself, and rejects
any string literal holding a path that exists only on the machine that wrote it.

Two of those eight earned their place by catching something. The flag check took two passes
to get right: matching `shieldnet <command> --flag` catches flags written next to their
command, and the ones that escaped were not written that way. A config comment promised a
`resume` flag for the Optuna study; a label-decoder error told the user to check their
`merge-rare` setting, spelled as a flag. Neither flag has ever existed — the study resumes
by name with no flag at all, and the setting is `data.merge_rare` in the config file — and
both read as instructions to somebody debugging at midnight. So a second pass takes every
remaining flag in the tree and asks whether *any* command has it, skipping lines that drive
some other program. Between them the two passes check 287 flag uses across 42 files. An
earlier version of the same check found the "you have no data, here is how to fix it" error
message telling the reader to run a `synth` sub-command that has never existed.

The path check is there because of a worse bug. All six `tests/check_*.py` used to open by
assigning an absolute path to this project on the machine that wrote them, and putting
`src/` from *that* path on `sys.path`. Here, everything passed. On any other machine
`make check` — the first command this README gives — would have died on
`import shieldnet` before running a single test. And because a copy of the project imported
the original, the copy passed too, which is how the bug survived being tested from a fresh
directory. The check now rejects a drive letter, a home directory, a mount point, and any
literal resolving inside the project itself; `/tmp` and Colab's `/content` are allowed,
because those are conventions rather than one person's filesystem.

Be clear about what all this does and does not establish. It checks the plumbing: that the
stages agree about shapes, that a saved bundle predicts what the in-memory model
predicted, that chunked and whole-frame scoring give identical answers, that the CLI's
exit codes mean what they say, that the app's script runs and its numbers come from the
bundle. It cannot tell you whether XGBoost is any good at intrusion detection. Only the
notebook, on the real dataset, can do that, and the two are not substitutes.

### What has actually been run

The seven checks above pass on synthetic data with stub models. **The full CICIDS2017 run
has not been executed here** — it needs the Kaggle dataset and a GPU, which is what the
notebook is for. Any accuracy, macro F1 or per-class recall you report must come from
your own run of that notebook, and the manifest it writes is the provenance for it. Do
not quote a number from a synthetic smoke test; the app marks those in red precisely so
that the mistake is hard to make by accident.

---

## The notebook is generated

`notebooks/ShieldNet_Colab.ipynb` is a build product of
`scripts/build_colab_notebook.py`. A `.ipynb` is JSON with every line of every cell
escaped into a string list, so editing one by hand means counting quotes inside quotes
and a diff between two versions is unreadable. Edit the generator:

```bash
make notebook            # regenerate
make notebook-check      # fail if the two have diverged
```

Cell ids are derived from position, so regenerating is a no-op diff. If you edited the
notebook in Colab, `make notebook-check` is what tells you before the change becomes
unreproducible.

---

## Before you submit this

Placeholders that want your details, all marked with angle brackets so a grep finds them:

```bash
grep -rn "<YOUR" --include="*.toml" --include="*.py" --include="*.ipynb" --include="*.md" .
```

They are in `pyproject.toml` (author name, email, homepage) and in the `REPO` variable of
`scripts/build_colab_notebook.py`. Edit the **generator**, not the notebook: the notebook
is a build product, so a URL typed into it in Colab is reverted by the next `make
notebook` and reported as a divergence by `make notebook-check`. Change it in the
generator and regenerate. Nothing functional depends on any of them — `REPO` is only used
by the notebook's "clone from GitHub" route, which is one of four ways it can find the
project.

---

## Dataset citation

Iman Sharafaldin, Arash Habibi Lashkari and Ali A. Ghorbani, *Toward Generating a New
Intrusion Detection Dataset and Intrusion Traffic Characterization*, 4th International
Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018.

The dataset is distributed by the Canadian Institute for Cybersecurity at the University
of New Brunswick; use it under their terms. It is not redistributed with this project and
`data/` is deliberately untracked.

## Licence

MIT, for the code in this repository. See the citation above for the data.
