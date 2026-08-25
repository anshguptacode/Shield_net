#!/usr/bin/env python3
"""Static cross-module checks: the bugs the runtime suite structurally cannot reach.

    python scripts/check_wiring.py

``tests/check_*.py`` execute the project, which is the right way to find out whether it
works - but they can only reach code that runs. A keyword argument renamed in one place
and not the other, in a branch that needs TensorFlow installed; a ``cfg.data.encodings``
typo in an error path; a ``--flag`` that a docstring promises and the parser never had.
None of those fail a green run. They fail in front of somebody.

So this reads the source instead of running it. Eight checks:

1. **compile** every file, including the ones nothing imports.
2. **import** every module in the package, which is where a circular import or a
   module-level typo surfaces.
3. **keywords**: for every call to a uniquely-named project function or method, check the
   keyword arguments against its real signature.
4. **config**: every ``cfg.section.field`` in the source exists on the dataclasses.
5. **exports**: every name in an ``__all__`` exists in its module.
6. **flags**, in two passes. Every ``shieldnet <command> --flag`` written anywhere in the
   tree - README, YAML comments, the Makefile, the notebook, docstrings - is checked
   against that command's real options. Then every *remaining* ``--flag`` in the tree is
   checked against the union of all of them plus the ``scripts/*.py`` parsers, because the
   flags that got through the first pass were not written next to a command: a config
   comment saying "``--resume`` picks it up", an error message about your
   "``--merge-rare`` settings". Neither has ever existed, and both read as instructions.
7. **config files**: every ``config/*.yaml`` loads, and ``default.yaml`` really is
   identical to the built-in defaults it says it restates.
8. **absolute paths**: no string literal names a path that exists only on the machine
   that wrote it. Every ``tests/check_*.py`` once began ``ROOT = Path("/sessions/...")``,
   which passes here and dies on ``import shieldnet`` anywhere else - and, because a copy
   of the project then imported the *original*, passed in a copy too while testing
   nothing.

Checks 6 and 8 are the ones that pay for the file. Documentation drifts silently and by
hand; a path that only works here fails silently and only for somebody else. Both now
fail loudly, here.

Exit codes: 0 clean, 1 problems found, 2 the checker itself could not run.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import inspect
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "shieldnet"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "app"))
# The model backends import xgboost and friends lazily, but the stub registry is what
# makes `import shieldnet.models` meaningful in an environment without them.
sys.path.insert(0, str(ROOT / "tests" / "_stubs"))

#: Files whose text is scanned for command lines in check 6.
TEXT_GLOBS = ("README.md", "Makefile", "config/*.yaml", "notebooks/*.ipynb",
              "src/shieldnet/**/*.py", "app/*.py", "tests/*.py", "scripts/*.py")

Problem = Tuple[str, str]        # (where, what)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def python_files() -> List[Path]:
    out: List[Path] = []
    for pattern in ("src/**/*.py", "app/*.py", "tests/**/*.py", "scripts/*.py"):
        out.extend(sorted(ROOT.glob(pattern)))
    return [p for p in out if "__pycache__" not in p.parts]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:                                          # pragma: no cover
        return str(path)


def parse(path: Path) -> Optional[ast.Module]:
    try:
        return ast.parse(path.read_text("utf-8"), filename=str(path))
    except SyntaxError:
        return None


def attribute_root(node: ast.Attribute) -> str:
    """``a.b.c.d(...)`` -> ``"a"``. Returns "" when the chain starts with a call."""
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def external_names(tree: ast.Module) -> Set[str]:
    """Names in this module that were imported from outside the project.

    A call whose root is one of these belongs to somebody else's library, so the project
    signature of the same name says nothing about it.
    """
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ("shieldnet", "shieldnet_ui"):
                    out.add(alias.asname or top)
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` and `from shieldnet.foo import y` are ours; the level
            # test is what catches the relative form, where node.module is None.
            module = node.module or ""
            if node.level or module.split(".")[0] in ("shieldnet", "shieldnet_ui"):
                continue
            for alias in node.names:
                out.add(alias.asname or alias.name)
    return out


def library_names() -> Set[str]:
    """Attribute names owned by numpy, pandas and the standard library.

    ``empty``, ``dump``, ``parse``, ``get``, ``copy``, ``run``: short, useful words that a
    project function and a DataFrame method both want. When a name is claimed by both,
    this checker has no way to tell which one a call site meant, so it declines to guess.
    """
    out: Set[str] = set(dir(__builtins__) if isinstance(__builtins__, dict)
                        else dir(__builtins__))
    for module_name in ("numpy", "pandas", "os", "os.path", "json", "re", "sys",
                        "shutil", "subprocess", "pathlib", "ast", "inspect", "time",
                        "logging", "warnings", "argparse", "collections", "itertools",
                        "math", "random", "textwrap", "datetime", "hashlib", "pickle"):
        try:
            module = importlib.import_module(module_name)
        except Exception:                                       # noqa: BLE001
            continue
        out.update(n for n in dir(module) if not n.startswith("_"))
    for qualified in ("numpy:ndarray", "pandas:DataFrame", "pandas:Series",
                      "pathlib:Path", "logging:Logger"):
        module_name, attribute = qualified.split(":")
        try:
            obj = getattr(importlib.import_module(module_name), attribute)
        except Exception:                                       # noqa: BLE001
            continue
        out.update(n for n in dir(obj) if not n.startswith("_"))
    return out


# ---------------------------------------------------------------------------
# 1. compile
# ---------------------------------------------------------------------------

def check_compiles(files: List[Path]) -> List[Problem]:
    problems: List[Problem] = []
    for path in files:
        try:
            compile(path.read_text("utf-8"), str(path), "exec")
        except SyntaxError as exc:
            problems.append((f"{rel(path)}:{exc.lineno}", f"{exc.msg}"))
    return problems


# ---------------------------------------------------------------------------
# 2. import
# ---------------------------------------------------------------------------

def check_imports() -> Tuple[List[Problem], List[Any]]:
    """Import every module in the package, plus the app's logic module."""
    problems: List[Problem] = []
    modules: List[Any] = []

    try:
        import stub_models
        stub_models.register()
    except Exception as exc:                                    # noqa: BLE001
        problems.append(("tests/_stubs/stub_models.py", f"could not register: {exc!r}"))

    names = ["shieldnet"]
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts[-1] == "__main__":
            # Importing __main__ runs the CLI. Compiling it is the check it gets.
            continue
        names.append(".".join(parts))

    for name in dict.fromkeys(names):
        try:
            modules.append(importlib.import_module(name))
        except Exception as exc:                                # noqa: BLE001
            line = traceback.format_exc().strip().splitlines()[-1]
            problems.append((name, f"import failed: {line}"))

    try:
        modules.append(importlib.import_module("shieldnet_ui"))
    except Exception as exc:                                    # noqa: BLE001
        problems.append(("app/shieldnet_ui.py", f"import failed: {exc!r}"))

    return problems, modules


# ---------------------------------------------------------------------------
# 3. keyword arguments
# ---------------------------------------------------------------------------

def collect_signatures(files: List[Path]) -> Dict[str, Tuple[inspect.Signature, str]]:
    """Name -> (signature, where), for names that are unambiguous project-wide.

    Ambiguous names are dropped rather than guessed at. Three ``render`` methods on three
    classes cannot be told apart without type inference, and a checker that guesses
    produces false positives, which is how a checker gets switched off.
    """
    seen: Dict[str, List[Tuple[ast.AST, str]]] = {}

    for path in files:
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen.setdefault(node.name, []).append(
                    (node, f"{rel(path)}:{node.lineno}"))

    out: Dict[str, Tuple[inspect.Signature, str]] = {}
    for name, entries in seen.items():
        if len(entries) != 1 or name.startswith("__"):
            continue
        node, where = entries[0]
        params: List[inspect.Parameter] = []
        a = node.args
        positional = list(a.posonlyargs) + list(a.args)
        # Drop the bound first argument: a call site writes `det.prepare(frame)`, and
        # binding that against a signature that still starts with `self` would report
        # every method call in the project as wrong.
        decorators = {d.id if isinstance(d, ast.Name) else
                      getattr(d, "attr", "") for d in node.decorator_list}
        is_method = bool(positional) and positional[0].arg in ("self", "cls")
        if is_method and "staticmethod" not in decorators:
            positional = positional[1:]
        for arg in positional:
            params.append(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                            default=inspect.Parameter.empty))
        if a.vararg:
            params.append(inspect.Parameter(a.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
        for arg in a.kwonlyargs:
            params.append(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                                            default=None))
        if a.kwarg:
            params.append(inspect.Parameter(a.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
        try:
            out[name] = (inspect.Signature(params), where)
        except ValueError:                                      # pragma: no cover
            continue
    return out


def check_keywords(files: List[Path],
                   signatures: Dict[str, Tuple[inspect.Signature, str]]) -> List[Problem]:
    """Flag keyword arguments no signature can accept."""
    problems: List[Problem] = []
    ambiguous = library_names()

    for path in files:
        tree = parse(path)
        if tree is None:
            continue
        external = external_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
                root = name
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                root = attribute_root(node.func)
            else:
                continue
            # `np.empty(shape, dtype=...)` is not the project's `empty()`, and
            # `joblib.dump(obj, path, compress=3)` is not the project's `dump()`. A name
            # matching is not a resolution; without these two guards the check reports
            # every third-party call whose name a project function happens to share, and
            # a checker with 19 false positives is a checker nobody runs.
            if root in external or name in ambiguous:
                continue
            entry = signatures.get(name)
            if entry is None:
                continue
            signature, defined = entry
            accepts = {p.name for p in signature.parameters.values()
                       if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                     inspect.Parameter.KEYWORD_ONLY)}
            open_ended = any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in signature.parameters.values())
            if open_ended:
                continue
            for keyword in node.keywords:
                if keyword.arg is None:      # **kwargs at the call site: unknowable
                    break
                if keyword.arg not in accepts:
                    problems.append((
                        f"{rel(path)}:{node.lineno}",
                        f"{name}(... {keyword.arg}=...) - not a parameter of "
                        f"{name} defined at {defined}; it takes "
                        f"{', '.join(sorted(accepts)) or '(nothing by keyword)'}"))
    return problems


# ---------------------------------------------------------------------------
# 4. config attributes
# ---------------------------------------------------------------------------

def check_config(files: List[Path]) -> List[Problem]:
    """Every ``cfg.section.field`` must exist on the dataclass tree."""
    from shieldnet.config import Config

    config = Config()
    sections: Dict[str, Set[str]] = {}
    top: Set[str] = set()
    for field in dataclasses.fields(config):
        top.add(field.name)
        value = getattr(config, field.name)
        if dataclasses.is_dataclass(value):
            sections[field.name] = {f.name for f in dataclasses.fields(value)}
            # Properties and methods count as valid attributes too: `cfg.paths.resolve`
            # is not a field but is certainly not a typo.
            sections[field.name] |= {n for n in dir(value) if not n.startswith("_")}

    problems: List[Problem] = []
    for path in files:
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            inner = node.value
            if not isinstance(inner, ast.Attribute) or not isinstance(inner.value, ast.Name):
                continue
            if inner.value.id not in ("cfg", "config", "self.cfg"):
                continue
            section, field = inner.attr, node.attr
            if section not in sections:
                if section not in top and section not in {n for n in dir(config)
                                                          if not n.startswith("_")}:
                    problems.append((f"{rel(path)}:{node.lineno}",
                                     f"cfg.{section} is not a Config field"))
                continue
            if field not in sections[section]:
                near = sorted(n for n in sections[section] if n[:3] == field[:3])
                hint = f" (did you mean {', '.join(near)}?)" if near else ""
                problems.append((f"{rel(path)}:{node.lineno}",
                                 f"cfg.{section}.{field} does not exist{hint}"))
    return problems


# ---------------------------------------------------------------------------
# 5. __all__
# ---------------------------------------------------------------------------

def check_exports(modules: List[Any]) -> List[Problem]:
    problems: List[Problem] = []
    for module in modules:
        names = getattr(module, "__all__", None)
        if not names:
            continue
        for name in names:
            if not hasattr(module, name):
                problems.append((getattr(module, "__name__", "?"),
                                 f"__all__ lists {name!r}, which the module does not define"))
    return problems


# ---------------------------------------------------------------------------
# 6. documented flags
# ---------------------------------------------------------------------------

#: `shieldnet train --flag`, `python -m shieldnet train --flag`, `$(SHIELDNET) train ...`.
_INVOCATION = re.compile(
    r"(?:\$\(SHIELDNET\)|python -m shieldnet|shieldnet)\s+"
    r"(?P<command>[a-z][a-z-]*)"
    r"(?P<rest>(?:[ \t]+(?:--?[A-Za-z][\w-]*|[^\s]+))*)")
#: `--flag`, and `--server.port` as one token rather than as `--server`. The dotted form
#: only matches when a word character follows the dot, so a flag ending a sentence -
#: "pass `--quiet`." - does not swallow the full stop.
_FLAG = re.compile(r"(?<![\w-])--[A-Za-z][\w-]*(?:\.[\w-]+)*")


def parser_flags() -> Dict[str, Set[str]]:
    """command -> the long options it accepts, straight from argparse."""
    from shieldnet.cli import build_parser

    parser = build_parser()
    subparsers = [a for a in parser._actions
                  if isinstance(a, argparse._SubParsersAction)]
    if not subparsers:                                          # pragma: no cover
        raise RuntimeError("the CLI has no sub-commands; has build_parser changed?")
    out: Dict[str, Set[str]] = {}
    for name, sub in subparsers[0].choices.items():
        flags: Set[str] = set()
        for action in sub._actions:
            flags.update(o for o in action.option_strings if o.startswith("--"))
        out[name] = flags
    out["(top level)"] = {o for a in parser._actions
                          for o in a.option_strings if o.startswith("--")}
    return out


#: Commands that are not this project's. A line that invokes one of these is that tool's
#: business, and `pip install --force-reinstall` is not a ShieldNet flag going wrong.
_OTHER_TOOLS = re.compile(
    r"(?<![\w-])(?:pip|pip3|conda|apt|apt-get|make|git|grep|rg|find|rm|sed|awk|curl|"
    r"wget|unzip|zip|tar|docker|jupyter|streamlit|nvidia-smi|kaggle|ls|cp|mv|mkdir|"
    r"nproc|du|df|echo|test|python -m pip|python -m venv|pytest|mypy|ruff|flake8)"
    r"(?![\w-])")

#: `--flag` written as a placeholder rather than as a real option.
_PLACEHOLDER_FLAGS = {"--flag", "--flags", "--some-flag"}


def script_flags() -> Set[str]:
    """Long options accepted by the project's own ``scripts/*.py``.

    ``make notebook-check`` is ``build_colab_notebook.py --check``, and ``--check`` is not
    a ShieldNet CLI flag. Introspecting these parsers rather than allow-listing the names
    means a renamed script flag is caught the same way a renamed CLI flag is.
    """
    found: Set[str] = set()
    for path in sorted((ROOT / "scripts").glob("*.py")):
        tree = parse(path)
        if tree is None:
            continue
        # add_argument("--check", ...) - read the literal, do not import and run the
        # module, because importing a script means running whatever is at module level.
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.startswith("--"):
                    found.add(arg.value)
    return found


def text_files() -> List[Path]:
    out: List[Path] = []
    for pattern in TEXT_GLOBS:
        out.extend(sorted(ROOT.glob(pattern)))
    here = Path(__file__).resolve()
    # This file's docstring contains `shieldnet <command> --flag` as an illustration of
    # what the check looks for. Scanning it would report the example as a defect.
    return [p for p in out if "__pycache__" not in p.parts and p.resolve() != here]


def check_flags() -> Tuple[List[Problem], int]:
    """Validate every command line written anywhere in the tree.

    Two passes, and the second exists because the first was not enough. Matching
    ``shieldnet <command> ... --flag`` only sees flags sitting on the same line as the
    command, and the mistakes that got through were not written that way: a config
    comment saying "``--resume`` picks it up", an error message telling the user to check
    their "``--merge-rare`` settings". Neither flag has ever existed, and both read as
    instructions. So the second pass takes every remaining ``--flag`` in the tree and asks
    whether *any* ShieldNet command has it.
    """
    commands = parser_flags()
    named = sorted(k for k in commands if not k.startswith("("))
    known = set(script_flags()) | {f for fs in commands.values() for f in fs}
    problems: List[Problem] = []
    checked = 0

    for path in text_files():
        try:
            text = path.read_text("utf-8")
        except UnicodeDecodeError:                              # pragma: no cover
            continue
        # A shell continuation joins two lines into one command, so it is flattened
        # everywhere or half of every multi-line invocation goes unchecked.
        text = text.replace("\\\n", " ")
        # A `.ipynb` is JSON: a cell's line breaks are the two characters \ and n, and
        # without expanding them the whole notebook is a handful of enormous lines. Doing
        # this to a .py file instead *corrupts* it - `"\n"` in a Python string literal is
        # those same two characters, so expanding them inserts newlines that are not
        # there and every reported line number below the first one is wrong.
        if path.suffix == ".ipynb":
            text = text.replace("\\n", "\n")

        for line_offset, line in enumerate(text.splitlines(), start=1):
            attributed: Set[int] = set()          # offsets already judged in pass 1

            for match in _INVOCATION.finditer(line):
                command = match.group("command")
                rest = match.group("rest") or ""
                base = match.start("rest")
                if command not in commands:
                    # `shieldnet --help`, `shieldnet <command>` in a table, prose like
                    # "shieldnet is a package". Only flag things that look like a real
                    # command being misspelled: a bare word followed by a flag.
                    if _FLAG.search(rest) and command.isalpha() \
                            and command not in {"is", "and", "or", "the", "a", "in"}:
                        problems.append((f"{rel(path)}:{line_offset}",
                                         f"`shieldnet {command}` is not a command "
                                         f"({', '.join(named)})"))
                        attributed.update(m.start() + base for m in _FLAG.finditer(rest))
                    continue
                for m in _FLAG.finditer(rest):
                    flag, checked = m.group(), checked + 1
                    attributed.add(m.start() + base)
                    if flag not in commands[command]:
                        problems.append((
                            f"{rel(path)}:{line_offset}",
                            f"`shieldnet {command} {flag}` - that command has no {flag} "
                            f"(it has {', '.join(sorted(commands[command]))})"))

            # Pass 2. Skipped entirely for lines that drive some other program: a line
            # containing `pip install --force-reinstall` or `git clone --depth 1` is that
            # tool's business, and validating its flags against argparse would report a
            # correct command as broken.
            if _OTHER_TOOLS.search(line):
                continue
            for m in _FLAG.finditer(line):
                flag = m.group()
                if m.start() in attributed or flag in _PLACEHOLDER_FLAGS:
                    continue
                # `--server.port`, `--server.headless`. argparse cannot make a dotted
                # option, so a dot proves the flag belongs to something else - here
                # Streamlit, invoked as a subprocess argument list whose elements sit on
                # their own lines, out of reach of the tool names on the line above.
                if "." in flag:
                    continue
                checked += 1
                if flag not in known:
                    problems.append((
                        f"{rel(path)}:{line_offset}",
                        f"{flag} is not an option of any shieldnet command or project "
                        f"script. Writing it here reads as an instruction."))
    return problems, checked


# ---------------------------------------------------------------------------
# 7. the YAML files against the dataclasses
# ---------------------------------------------------------------------------

def check_config_files() -> Tuple[List[Problem], int]:
    """Load every ``config/*.yaml``, and hold ``default.yaml`` to its own claim.

    Two failures live here. The first is a typo in a shipped config: unknown keys are a
    hard error by design, so ``n_feature: 25`` raises - but only for whoever runs that
    file, and nobody runs ``colab.yaml`` on a laptop.

    The second is ``default.yaml`` drifting from ``config.py``. That file opens by saying
    it restates the built-in defaults, which makes it documentation that can be wrong: a
    default changed in the dataclass and not in the YAML leaves a comment explaining a
    value the program no longer uses. The equality below is the assertion the file claims
    is made, so it now is.
    """
    from shieldnet.config import Config

    problems: List[Problem] = []
    files = sorted(ROOT.glob("config/*.yaml"))
    for path in files:
        try:
            Config.load(path)
        except Exception as exc:                                # noqa: BLE001
            problems.append((rel(path), f"does not load: {type(exc).__name__}: {exc}"))

    default = ROOT / "config" / "default.yaml"
    if default.is_file() and not any(p[0] == rel(default) for p in problems):
        built_in, from_yaml = Config.load(None), Config.load(default)
        for section in dataclasses.fields(built_in):
            mine = getattr(built_in, section.name)
            theirs = getattr(from_yaml, section.name)
            if mine == theirs:
                continue
            if not dataclasses.is_dataclass(mine):
                problems.append((rel(default),
                                 f"{section.name}: {theirs!r} but the default is {mine!r}"))
                continue
            for field in dataclasses.fields(mine):
                a, b = getattr(mine, field.name), getattr(theirs, field.name)
                if a != b:
                    problems.append((
                        rel(default),
                        f"{section.name}.{field.name}: {b!r} here, {a!r} in config.py - "
                        f"this file says the two are identical"))
    return problems, len(files)


# ---------------------------------------------------------------------------
# 8. absolute paths that only exist on one machine
# ---------------------------------------------------------------------------

#: Absolute POSIX or Windows paths, as they appear inside a string literal. `/tmp/x` and
#: `/usr/bin/env` are absolute too, so the judgement below is about *where* they point,
#: not about the leading slash.
_ABS_PATH = re.compile(r"^(?:/[^/\s]+(?:/[^\s]*)?|[A-Za-z]:[\\/][^\s]*)$")

#: A drive letter, which needs no allow-list: there is no portable `C:\` path. Worth
#: matching separately because every prefix below is POSIX, so a Windows path would
#: otherwise fall through all of them and be accepted.
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")

#: Absolute prefixes that are a machine, not a filesystem convention. `/tmp` is a
#: contract every POSIX system honours; `/home/priya` is one laptop. `/content/drive/` is
#: in here rather than under the Colab exemption below because a Drive path names one
#: person's Drive - the notebook's `DRIVE_COPY` is opt-in and empty by default for exactly
#: that reason.
_MACHINE_ROOTS = ("/home/", "/Users/", "/root/", "/mnt/", "/media/", "/sessions/",
                  "/content/drive/")

#: Absolute paths that are genuinely portable and deliberate. Checked after
#: ``_MACHINE_ROOTS``, so `/content` admits `/content/shieldnet-run` while
#: `/content/drive/MyDrive/...` is still rejected.
_PORTABLE_ABS = ("/tmp", "/usr/bin/env", "/usr/bin", "/proc", "/dev/null", "/etc",
                 "/content")


def check_absolute_paths(files: List[Path]) -> Tuple[List[Problem], int]:
    """Reject string literals holding a path that only resolves on the author's machine.

    This check exists because of a real bug, and a bad one: all six ``tests/check_*.py``
    opened with ``ROOT = Path("/sessions/.../shieldnet")`` and put ``ROOT / "src"`` on
    ``sys.path``. On the machine that wrote them everything passed. Anywhere else - the
    marker's laptop, a fresh clone, Colab - ``make check`` dies on ``import shieldnet``
    before running a single test, and ``make check`` is the first command the README
    gives. Worse, on the authoring machine a *copy* of the project imported the original,
    so the copy passed while being entirely untested, which is how the bug survived a
    cold-start verification.

    Two rules, and the second is the subtle one:

    * A literal under a machine-specific root (``/home``, ``/Users``, ``/sessions``, a
      mounted drive) is always wrong. Nobody else has that directory.
    * A literal that resolves *inside this project* is wrong even when it is correct,
      because it is only correct here. It should be derived from ``__file__``.

    ``/tmp/shieldnet_run`` and ``/content`` survive, deliberately: scratch space and the
    Colab mount are conventions rather than one person's home directory.

    The two tuples above are themselves lists of machine-specific prefixes, so this file
    is exempted from its own rule - but only for those two assignments, by name. Excluding
    the whole file, which is what :func:`text_files` has to do for check 6, would leave the
    checker as the one place in the tree free to hardcode a path.
    """
    exempt = {"_MACHINE_ROOTS", "_PORTABLE_ABS"}
    problems: List[Problem] = []
    scanned = 0
    root_text = str(ROOT)
    for path in files:
        tree = parse(path)
        if tree is None:
            continue
        spared: Set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in exempt for t in node.targets):
                spared.update(id(n) for n in ast.walk(node.value))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            value = node.value.strip()
            if not _ABS_PATH.match(value):
                continue
            scanned += 1
            if id(node) in spared:
                continue
            if _WINDOWS_ABS.match(value):
                problems.append((
                    f"{rel(path)}:{node.lineno}",
                    f"{value!r} is a drive-letter path, so it names one Windows machine. "
                    "Derive it: Path(__file__).resolve().parents[N]."))
                continue
            if value.startswith(_PORTABLE_ABS) and not value.startswith(_MACHINE_ROOTS):
                continue
            if value.startswith(root_text):
                problems.append((
                    f"{rel(path)}:{node.lineno}",
                    f"{value!r} is inside this project, so it is only right on this "
                    "machine. Derive it: Path(__file__).resolve().parents[N]."))
            elif value.startswith(_MACHINE_ROOTS):
                problems.append((
                    f"{rel(path)}:{node.lineno}",
                    f"{value!r} names one machine's filesystem. Nobody who clones this "
                    "repo has that directory."))
    return problems, scanned


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def report(title: str, problems: List[Problem], detail: str = "") -> bool:
    mark = "ok  " if not problems else "FAIL"
    suffix = f"   {detail}" if detail else ""
    print(f"  {mark} {title}{suffix}")
    for where, what in problems:
        print(f"         {where}: {what}")
    return not problems


def main(argv: Optional[List[str]] = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)

    print("=" * 78)
    print("static wiring checks")
    print("=" * 78)

    files = python_files()
    ok = True

    ok &= report("compiles", check_compiles(files), f"{len(files)} file(s)")

    import_problems, modules = check_imports()
    ok &= report("imports", import_problems, f"{len(modules)} module(s)")

    if import_problems:
        print("\nimports failed, so the checks that need live objects are skipped.")
        return 1

    signatures = collect_signatures(files)
    ok &= report("keyword arguments", check_keywords(files, signatures),
                 f"{len(signatures)} unambiguous signature(s)")
    ok &= report("config attributes", check_config(files))
    ok &= report("__all__ exports", check_exports(modules))

    flag_problems, flag_count = check_flags()
    ok &= report("documented flags", flag_problems,
                 f"{flag_count} flag use(s) across {len(text_files())} file(s)")

    yaml_problems, yaml_count = check_config_files()
    ok &= report("config files", yaml_problems, f"{yaml_count} file(s)")

    path_problems, path_count = check_absolute_paths(files)
    ok &= report("absolute paths", path_problems, f"{path_count} literal(s)")

    print("=" * 78)
    # scripts/run_checks.py greps a check's output for "PASSED" to build its summary
    # line, so this says more than "ok": the counts are the answer to "did it actually
    # look at anything?", which is the question to ask of any check that always passes.
    print(f"static checks PASSED - {len(files)} files, {len(modules)} modules, "
          f"{len(signatures)} signatures, {flag_count} flag uses, {yaml_count} configs, "
          f"{path_count} path literals"
          if ok else "WIRING PROBLEMS FOUND")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
