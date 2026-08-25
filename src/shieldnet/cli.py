"""Command line entry point: ``shieldnet <command>`` (or ``python -m shieldnet``).

Every stage of the project is reachable from here, and nothing here contains logic of
its own - a command parses arguments, builds a :class:`~shieldnet.config.Config`, calls
one library function and prints what came back. That separation is deliberate: the
Colab notebook, the Streamlit app and this file must be able to disagree about
presentation while producing byte-identical models.

Commands, in the order you would run them:

    shieldnet doctor                     what works in this environment
    shieldnet models                     which model backends are installed
    shieldnet download                   fetch CICIDS2017 from Kaggle
    shieldnet prepare                    load, clean, split, and cache
    shieldnet train                      the full pipeline, saving an artifact
    shieldnet info                       what is in the saved artifact
    shieldnet evaluate --input FILE      score a labelled capture
    shieldnet predict  --input FILE      write verdicts for an unlabelled capture
    shieldnet explain  --input FILE --row 12
    shieldnet serve                      launch the Streamlit app

Exit codes: 0 success, 1 an expected failure with an explanation (missing artifact, no
Kaggle token, unreadable file), 2 a usage error, 130 interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .config import Config, seed_everything
from .logging_utils import configure_logging, get_logger, human_duration

log = get_logger(__name__)

EPILOG = """\
examples:
  shieldnet train --synthetic 40000 --models logistic_regression --no-tune
  shieldnet train --models xgboost,lightgbm,random_forest --tune --trials 40
  shieldnet predict --input capture.csv --output verdicts.csv --top-k 3
  shieldnet evaluate --input data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  shieldnet explain --input capture.csv --row 41
"""


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options every command accepts. Repeated per sub-parser on purpose.

    argparse can only put shared options *before* the sub-command if they live on the
    parent, which means ``shieldnet --seed 7 train`` works but ``shieldnet train
    --seed 7`` does not. The second is what anyone actually types, so the options are
    attached to each sub-parser instead.
    """
    g = parser.add_argument_group("common")
    g.add_argument("--config", metavar="FILE",
                   help="YAML or JSON config; CLI flags override its values")
    g.add_argument("--root", metavar="DIR",
                   help="project root for data/, artifacts/ and reports/")
    g.add_argument("--seed", type=int, help="global random seed (default 42)")
    g.add_argument("--jobs", type=int, metavar="N",
                   help="parallel workers where supported (-1 = all cores)")
    g.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    g.add_argument("--log-file", metavar="FILE", help="also write logs here")
    g.add_argument("--quiet", action="store_true", help="suppress stage banners")
    g.add_argument("--traceback", action="store_true",
                   help="show the full traceback instead of a one-line message")


def _add_pipeline_overrides(parser: argparse.ArgumentParser) -> None:
    """Flags that change what a run *is*, shared by ``train`` and ``config``.

    ``config`` takes them too so that ``shieldnet config --models xgboost --tune`` is a
    dry run: it prints exactly the settings ``train`` would use, which is both the
    fastest way to check a flag did what you meant and the thing to paste into a report
    appendix as the run's provenance.
    """
    g = parser.add_argument_group("pipeline")
    g.add_argument("--models", metavar="LIST",
                   help="comma-separated registry keys (see `shieldnet models`)")
    g.add_argument("--features", type=int, metavar="N",
                   help="how many features to select (default 25)")
    g.add_argument("--balance", choices=["none", "smote", "smote_tomek",
                                         "undersample", "class_weight"])
    g.add_argument("--select-metric", metavar="NAME",
                   help="validation metric that decides the winner (default macro_f1)")
    g.add_argument("--trials", type=int, metavar="N", help="tuning trials per model")
    tuning = g.add_mutually_exclusive_group()
    tuning.add_argument("--tune", dest="tune", action="store_true", default=None)
    tuning.add_argument("--no-tune", dest="tune", action="store_false")


def _config_from(args: argparse.Namespace) -> Config:
    """Assemble the config: file, then explicit flags, in that order."""
    overrides: Dict[str, Any] = {}
    if args.root:
        overrides["paths"] = {"root": str(Path(args.root).expanduser())}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.jobs is not None:
        overrides["n_jobs"] = args.jobs
    if getattr(args, "features", None):
        overrides["features"] = {"n_features": args.features}
    if getattr(args, "models", None):
        models = _split_list(args.models)
        overrides.setdefault("train", {})["models"] = models
        overrides["train"]["primary"] = models[0]
    if getattr(args, "select_metric", None):
        overrides.setdefault("train", {})["selection_metric"] = args.select_metric
    tune_flag = getattr(args, "tune", None)
    if tune_flag is not None:
        overrides.setdefault("tune", {})["enabled"] = bool(tune_flag)
    if getattr(args, "trials", None):
        overrides.setdefault("tune", {})["n_trials"] = args.trials
        overrides["tune"].setdefault("enabled", True)
    if getattr(args, "balance", None):
        overrides["balance"] = {"strategy": args.balance}
    if getattr(args, "quiet", False):
        overrides["verbose"] = False

    cfg = Config.load(args.config, **overrides) if args.config else Config.load(**overrides)
    seed_everything(cfg.seed)
    return cfg


def _split_list(value: str) -> List[str]:
    """``"a,b , c"`` and ``"a b c"`` both mean three names."""
    return [part for part in value.replace(",", " ").split() if part]


def _artifacts_dir(args: argparse.Namespace, cfg: Optional[Config] = None) -> Path:
    if getattr(args, "artifacts", None):
        return Path(args.artifacts).expanduser()
    cfg = cfg or _config_from(args)
    return cfg.paths.resolve("artifacts")


def _read_input(path: str) -> Any:
    from .inference import read_flows
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(
            f"{target} does not exist. Point --input at a CICFlowMeter CSV (or a "
            "Parquet file); `shieldnet prepare --synthetic 20000 --sample demo.csv` "
            "writes one you can practise on - the --sample flag is what produces the "
            "file, so prepare on its own will not leave you a CSV."
        )
    return read_flows(target)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    """Report on the environment before anything expensive is attempted."""
    from . import schema as sch
    from .models import registry
    from .narrate import check_profiles
    from .persist import dependency_report

    cfg = _config_from(args)
    print(f"ShieldNet {__version__}")
    print(f"python {sys.version.split()[0]} on {sys.platform}")

    print("\nlibraries")
    rows = dependency_report()
    for row in rows:
        mark = " " if row["version"] not in ("absent", "BROKEN") else "!"
        print(f"  {mark} {row['module']:<12} {row['version']:<12} {row['unlocks']}")
    missing = [r for r in rows if r["version"] == "absent"]
    broken = [r for r in rows if r["version"] == "BROKEN"]
    if broken:
        # Installed but unimportable is a different problem from not installed, and it
        # needs a different fix: reinstall, not install. Usually a wheel built for
        # another interpreter or another CPU.
        print(f"\n  {len(broken)} package(s) are installed but fail to import: "
              f"{', '.join(r['module'] for r in broken)}")
        print(f"    pip install --force-reinstall "
              f"{' '.join(r['pip'] for r in broken)}")
    if missing:
        print(f"\n  {len(missing)} absent. To get all of them:")
        print(f"    pip install {' '.join(r['pip'] for r in missing)}")
    else:
        print("\n  every optional dependency is present")

    print("\npaths")
    for key, path in cfg.resolved_paths().items():
        state = "exists" if path.exists() else "will be created"
        count = ""
        if path.exists() and path.is_dir():
            files = [p for p in path.iterdir() if p.is_file()]
            count = f", {len(files)} file(s)" if files else ", empty"
        print(f"  {key:<10} {path}  ({state}{count})")

    print("\nmodels")
    print(textwrap.indent(registry.describe_all(), "  "))

    print("\nschema")
    print(f"  {len(sch.CANONICAL_FEATURES)} canonical features, "
          f"{len(sch.RAW_LABELS)} raw labels -> "
          f"{len(set(sch.canonical_label(l) for l in sch.RAW_LABELS))} classes")
    problems = check_profiles()
    if problems:
        print("  narration profiles INCOMPLETE:")
        for line in problems:
            print(f"    - {line}")
    else:
        print("  every class has a narration profile, severity and recommended action")

    print("\ndata")
    raw = cfg.paths.resolve("raw")
    csvs = sorted(raw.rglob("*.csv")) if raw.exists() else []
    if csvs:
        total = sum(p.stat().st_size for p in csvs)
        print(f"  {len(csvs)} raw CSV(s) in {raw} ({total / 1e6:,.0f} MB)")
        expected = set(sch.EXPECTED_RAW_FILES)
        found = {p.name for p in csvs}
        if expected - found:
            print(f"  missing {len(expected - found)} of the 8 CICIDS2017 day files: "
                  f"{', '.join(sorted(expected - found))}")
    else:
        print(f"  no raw CSVs in {raw} - run `shieldnet download`, or use "
              "`--synthetic N` to try the pipeline without them")

    manifest_path = cfg.paths.resolve("artifacts") / "manifest.json"
    if not manifest_path.exists():
        print("  no trained artifact yet - run `shieldnet train`")
    else:
        # A truncated manifest is the exact situation doctor exists for - a run killed
        # mid-save, a Colab disconnect during the copy - so parsing it must not be the
        # thing that stops doctor from printing the rest of the diagnosis.
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError(f"top level is {type(manifest).__name__}, not an object")
            metrics = manifest.get("metrics")
            f1 = metrics.get("macro_f1") if isinstance(metrics, dict) else None
            f1_text = f"{f1:.4f}" if isinstance(f1, (int, float)) else "not recorded"
            print(f"  artifact present: {manifest.get('model_name', 'unnamed')} "
                  f"({manifest.get('n_features', '?')} features, macro F1 {f1_text})")
            # `synthetic` is written under `metadata`, not at the top level. Reading the
            # wrong key here would silently never warn, which is the one failure this
            # line exists to prevent.
            metadata = manifest.get("metadata")
            if isinstance(metadata, dict) and metadata.get("synthetic"):
                source = metadata.get("source") or "generated flows"
                print(f"  SYNTHETIC - trained on {source}, not CICIDS2017. "
                      "Its metrics are not results.")
        except Exception as exc:                                # noqa: BLE001
            print(f"  artifact present but its manifest is unreadable: "
                  f"{type(exc).__name__}: {exc}")
            print(f"    {manifest_path}")
            print("    the bundle may still load; `shieldnet info` will say. If not, "
                  "re-run `shieldnet train`.")

    print("\nkaggle")
    try:
        from .data.download import ensure_credentials
        print(f"  credentials found via {ensure_credentials()}")
    except Exception as exc:                                # noqa: BLE001
        print(f"  {type(exc).__name__}: {str(exc).splitlines()[0]}")

    # Doctor never fails the shell: it is diagnostic output, and a CI step that runs it
    # for information should not go red because SHAP is absent. Everything above this
    # line is therefore inside a try or cannot raise - a diagnostic that crashes on the
    # broken environment it was run to diagnose is worse than no diagnostic.
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .models import registry
    print(registry.describe_all(only_available=args.available))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .data.download import download_dataset

    cfg = _config_from(args)
    target = Path(args.destination).expanduser() if args.destination \
        else cfg.paths.resolve("raw")
    files = download_dataset(target, slug=args.slug, force=args.force)
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} CSV file(s) in {target} ({total / 1e6:,.0f} MB)")
    for path in files:
        print(f"  {path.name:<52} {path.stat().st_size / 1e6:>8,.1f} MB")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from .train import prepare_data

    cfg = _config_from(args)
    data = prepare_data(cfg, source=args.source, synthetic_rows=args.synthetic,
                        streaming=args.streaming, cache=not args.no_cache,
                        quiet=args.quiet)
    print("\n" + data.render())
    if args.sample:
        out = Path(args.sample).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = data.split.X_test.head(args.sample_rows).copy()
        names = data.class_names
        frame["Label"] = [names[i] for i in data.split.y_test[:len(frame)]]
        frame.to_csv(out, index=False)
        print(f"\nwrote a {len(frame):,}-row labelled sample to {out}")
        print("  use it with: shieldnet evaluate --input " + str(out))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    cfg = _config_from(args)
    if not args.quiet:
        print(cfg.describe() + "\n")
    run = train(cfg, source=args.source, synthetic_rows=args.synthetic,
                models=_split_list(args.models) if args.models else None,
                tune=args.tune, explain=not args.no_explain, save=not args.no_save,
                cache=not args.no_cache, select=args.select, quiet=args.quiet)
    print("\n" + run.render())
    if run.bundle_path:
        print(f"\nartifact: {run.bundle_path}")
        print(f"reports:  {cfg.paths.resolve('reports')}")
        print("\nnext:  shieldnet predict --input <capture.csv> --output verdicts.csv")
    if not run.succeeded:
        return 1
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .inference import Detector

    directory = _artifacts_dir(args)
    det = Detector.load(directory)
    print(det.describe())
    m = det.bundle.metrics or {}
    meta = det.bundle.metadata or {}

    print(f"\nfeatures ({det.n_features}), in selection order:")
    for i, name in enumerate(det.feature_names, 1):
        print(f"  {i:>2}. {name}")

    print(f"\nclasses ({det.n_classes}):")
    recalls = m.get("per_class_recall") or {}
    for name in det.class_names:
        r = recalls.get(name)
        print(f"  {name:<26} recall {r:.4f}" if isinstance(r, (int, float))
              else f"  {name}")

    print("\nheadline metrics:")
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_recall",
                "macro_precision", "log_loss", "mcc", "false_alarm_rate"):
        if key in m and isinstance(m[key], (int, float)):
            print(f"  {key:<20} {m[key]:.4f}")

    if meta:
        print("\nprovenance:")
        for key in ("trained_at", "seed", "rows_fit", "synthetic", "selection_metric",
                    "shieldnet_version", "artifact_version"):
            if key in meta:
                print(f"  {key:<20} {meta[key]}")
        if meta.get("synthetic"):
            print("  NOTE: this artifact was trained on SYNTHETIC data. Its numbers "
                  "describe the generator, not CICIDS2017.")
    if args.json:
        # The manifest is written at save time and is the provenance record; print it
        # verbatim rather than re-deriving it from the loaded bundle, so what you read
        # here is exactly what shipped.
        manifest = directory / "manifest.json"
        if manifest.exists():
            print("\n" + manifest.read_text("utf-8"))
        else:
            print(f"\nno manifest.json in {directory}", file=sys.stderr)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .inference import Detector

    det = Detector.load(_artifacts_dir(args), attack_threshold=args.threshold,
                        min_confidence=args.min_confidence)
    frame = _read_input(args.input)
    batch = det.predict(frame, source=Path(args.input).name,
                        chunk_rows=args.chunk_rows, quiet=args.quiet)
    print("\n" + batch.render(top=args.top))
    print("\n" + textwrap.fill(batch.narrative(), 96))

    out = Path(args.output).expanduser() if args.output else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        table = batch.frame(probabilities=args.probabilities, top_k=args.top_k)
        table.to_csv(out, index=False)
        print(f"\nwrote {len(table):,} verdict(s) x {table.shape[1]} column(s) -> {out}")
        print("  row N of that file is the verdict on row N of the input; no row was "
              "dropped, even where values had to be repaired.")
    if args.json:
        target = Path(args.json).expanduser()
        target.write_text(json.dumps(batch.to_dict(), indent=2, default=str), "utf-8")
        print(f"wrote the run summary -> {target}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .inference import Detector
    from .narrate import narrate_evaluation

    det = Detector.load(_artifacts_dir(args))
    frame = _read_input(args.input)
    report = det.evaluate(frame, label_column=args.label_column,
                          fpr_budget=args.fpr_budget, quiet=args.quiet)
    print("\n" + report.render(confusion=not args.no_confusion, sweep=args.sweep))
    print("\n" + textwrap.fill(narrate_evaluation(report), 96))
    if args.json:
        target = Path(args.json).expanduser()
        target.write_text(report.to_json(), "utf-8")
        print(f"\nwrote metrics -> {target}")
    if args.figure:
        from .train import plot_confusion
        path = plot_confusion(report, Path(args.figure).expanduser())
        print(f"wrote {path}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .inference import Detector

    det = Detector.load(_artifacts_dir(args))
    if args.input:
        frame = _read_input(args.input)
        prep = det.prepare(frame, quiet=args.quiet)
        if not 0 <= args.row < prep.rows:
            raise ValueError(f"--row {args.row} is outside the file's "
                             f"0..{prep.rows - 1} range")
        batch = det.predict(None, prepared=prep, quiet=True)
        pred = batch.prediction(args.row, explainer=det.explainer(), X=prep.X,
                                raw_values=det._raw_values(prep, args.row), narrate=True)
    else:
        values = json.loads(args.values) if args.values else {}
        if not isinstance(values, dict):
            raise ValueError("--values must be a JSON object of feature: number pairs")
        pred = det.predict_one(values)

    print(f"\nrow {pred.row}: {pred.predicted_class} at {pred.confidence:.1%} "
          f"(severity {pred.severity}, {pred.status})")
    if pred.true_class:
        print(f"ground truth: {pred.true_class}"
              + ("  [correct]" if pred.correct else "  [WRONG]"))
    print("\ntop classes:")
    for name, p in pred.top_classes(args.top_classes):
        print(f"  {name:<26} {p:>7.2%}")
    if pred.explanation is not None:
        print("\n" + pred.explanation.render(top=args.top))
    print("\n" + textwrap.fill(pred.narrative, 96))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Hand over to Streamlit, which owns the process from here."""
    app = Path(args.app).expanduser() if args.app else _find_app()
    if app is None or not app.exists():
        print("could not find app/streamlit_app.py. Pass --app PATH, or run from the "
              "project root.", file=sys.stderr)
        return 1
    app = app.resolve()
    if shutil.which("streamlit") is None:
        try:
            import streamlit  # noqa: F401
        except ImportError:
            print("streamlit is not installed. Run: pip install streamlit",
                  file=sys.stderr)
            return 1
    # Streamlit reads .streamlit/config.toml from the *working directory*, not from the
    # directory of the script it was handed. The only copy in this repo is
    # app/.streamlit/config.toml, so starting from the project root - which is what the
    # README tells you to do - silently loads none of it: no theme, and maxUploadSize
    # back at Streamlit's 200 MB default while that file asks for 400. The eight raw
    # day-files are about 1 GB together, so the largest of them is on the wrong side of
    # 200 MB and the setting that goes missing is the one the biggest upload needs.
    workdir = app.parent
    cmd = [sys.executable, "-m", "streamlit", "run", app.name,
           "--server.port", str(args.port)]
    if args.no_browser:
        cmd += ["--server.headless", "true"]
    env = dict(os.environ)
    # The app imports shieldnet; make sure it finds this checkout rather than an older
    # installed copy.
    src = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    # Resolve the artifact directory here instead of leaving the app to search for it.
    # `serve` takes --root and --config like every other command, and a flag that is
    # silently ignored is worse than one that does not exist: `shieldnet serve --root
    # /data/run-3` would otherwise start the app against whatever ./artifacts held.
    #
    # .resolve() is not decoration now that the child starts in app/: `--artifacts
    # artifacts` reaches us as a relative path, and passing it through unresolved would
    # send the app looking in app/artifacts. Everything else the app touches is already
    # absolute - Paths.resolve() builds on PROJECT_ROOT, which config.py derives from
    # __file__ at import time and never from the working directory.
    artifacts = _artifacts_dir(args).resolve()
    env["SHIELDNET_ARTIFACTS"] = str(artifacts)
    if not (artifacts / "manifest.json").exists():
        # Not fatal. The app prints a fuller version of this on screen, and starting it
        # anyway is right - someone may be about to point it elsewhere from the sidebar.
        print(f"note: no trained artifact in {artifacts}. Run `shieldnet train` first, "
              "or pass --artifacts DIR.", file=sys.stderr)
    print("starting: " + " ".join(cmd))
    print(f"directory: {workdir}")
    print(f"artifacts: {artifacts}")
    try:
        return subprocess.call(cmd, cwd=str(workdir), env=env)
    except KeyboardInterrupt:
        return 130


def _find_app() -> Optional[Path]:
    here = Path(__file__).resolve()
    for base in [Path.cwd(), *here.parents]:
        candidate = base / "app" / "streamlit_app.py"
        if candidate.exists():
            return candidate
    return None


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _config_from(args)
    if args.output:
        path = cfg.save(Path(args.output).expanduser())
        print(f"wrote {path}")
    else:
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shieldnet",
        description="Explainable multi-class network intrusion detection on CICIDS2017.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                       version=f"shieldnet {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("doctor", help="check the environment, paths and artifact")
    _add_common(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("models", help="list model backends and whether they can run")
    p.add_argument("--available", action="store_true", help="only the runnable ones")
    _add_common(p)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("download", help="fetch CICIDS2017 from Kaggle")
    p.add_argument("--destination", metavar="DIR", help="default: paths.raw")
    p.add_argument("--slug", default="chethuhn/network-intrusion-dataset",
                   help="Kaggle dataset slug (default: %(default)s)")
    p.add_argument("--force", action="store_true", help="re-download even if CSVs exist")
    _add_common(p)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("prepare", help="load, clean, split and cache")
    p.add_argument("--source", metavar="DIR", help="directory of raw CSVs")
    p.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="generate N synthetic rows instead of reading files")
    p.add_argument("--streaming", action="store_true", default=None,
                   help="force the two-pass streaming loader")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--sample", metavar="FILE",
                   help="also write a labelled sample of the test split here")
    p.add_argument("--sample-rows", type=int, default=2000)
    _add_common(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="the full pipeline, ending in a saved artifact")
    p.add_argument("--source", metavar="DIR")
    p.add_argument("--synthetic", type=int, default=0, metavar="N")
    p.add_argument("--select", default="auto", choices=["auto", "primary"],
                   help="'auto' ships the best validation score; 'primary' ships "
                        "train.primary whatever the leaderboard says")
    p.add_argument("--no-explain", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    _add_pipeline_overrides(p)
    _add_common(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("info", help="describe the saved artifact")
    p.add_argument("--artifacts", metavar="DIR")
    p.add_argument("--json", action="store_true", help="also print the full manifest")
    _add_common(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("predict", help="score a capture and write verdicts")
    p.add_argument("--input", required=True, metavar="FILE")
    p.add_argument("--output", metavar="FILE", help="verdicts CSV")
    p.add_argument("--artifacts", metavar="DIR")
    p.add_argument("--threshold", type=float, default=0.5, metavar="P",
                   help="P(attack) at or above which a flow is flagged (default 0.5)")
    p.add_argument("--min-confidence", type=float, default=0.5, metavar="P",
                   help="below this top-class probability a row is marked for review")
    p.add_argument("--top-k", type=int, default=0, metavar="K",
                   help="also write the K-1 runner-up classes per row")
    p.add_argument("--probabilities", action="store_true",
                   help="write one column per class")
    p.add_argument("--chunk-rows", type=int, default=200_000)
    p.add_argument("--top", type=int, default=15, help="classes shown in the summary")
    p.add_argument("--json", metavar="FILE", help="write the run summary as JSON")
    _add_common(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="score a labelled capture and report metrics")
    p.add_argument("--input", required=True, metavar="FILE")
    p.add_argument("--artifacts", metavar="DIR")
    p.add_argument("--label-column", default="Label")
    p.add_argument("--fpr-budget", type=float, default=0.01, metavar="RATE",
                   help="false-alarm budget for the threshold sweep (default 1%%)")
    p.add_argument("--sweep", action="store_true", help="show the threshold sweep")
    p.add_argument("--no-confusion", action="store_true")
    p.add_argument("--figure", metavar="FILE", help="write a confusion heatmap here")
    p.add_argument("--json", metavar="FILE")
    _add_common(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("explain", help="explain one flow")
    p.add_argument("--input", metavar="FILE")
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--values", metavar="JSON",
                   help='manual entry, e.g. \'{"Flow Duration": 1200000}\'')
    p.add_argument("--artifacts", metavar="DIR")
    p.add_argument("--top", type=int, default=8, help="contributions to show")
    p.add_argument("--top-classes", type=int, default=4)
    _add_common(p)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("serve", help="launch the Streamlit app")
    p.add_argument("--app", metavar="FILE")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--artifacts", metavar="DIR")
    p.add_argument("--no-browser", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("config", help="print the fully resolved configuration")
    p.add_argument("--output", metavar="FILE")
    _add_pipeline_overrides(p)
    _add_common(p)
    p.set_defaults(func=cmd_config)

    return parser


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

#: Failures that mean "the user needs to do something", not "the code is broken".
#: These print one line; everything else keeps its traceback, because an unexpected
#: exception with its stack removed is a bug report nobody can act on.
_EXPECTED = (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    configure_logging(args.log_level, logfile=args.log_file, force=True)

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except _EXPECTED as exc:
        if args.traceback:
            raise
        name = type(exc).__name__
        # KeyError stringifies with quotes around the key, which reads like a typo in
        # our own message rather than a missing key.
        detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        print(f"\n{name}: {detail}", file=sys.stderr)
        print("\n(re-run with --traceback to see where this came from)", file=sys.stderr)
        return 1


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
