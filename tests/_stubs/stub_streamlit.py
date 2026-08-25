"""A recording stand-in for Streamlit, so that ``app/streamlit_app.py`` can be run.

Streamlit is not installed in this environment and, more importantly, would not help if it
were: ``streamlit run`` needs a browser and a websocket, which no assertion can reach. The
usual consequence is that the app is the least-tested file in a project and the only one
anybody looks at.

So this module impersonates the parts of the Streamlit API that the app uses. Widgets
return scripted answers instead of reading a browser, and every call is recorded, which
turns "does the app work" into ordinary assertions:

    st = stub_streamlit.install(script={"Score": True, "row number": 17})
    runpy.run_path("app/streamlit_app.py", run_name="__main__")
    assert st.text_containing("flagged as attack")
    assert not st.unknown, st.unknown

Three things it deliberately gets right, because getting them wrong would make a green
test meaningless:

* Unmodelled attributes are served as recording no-ops **and** logged to
  :attr:`Recorder.unknown`, so the script never dies on an API this stub has not thought
  about, but the test fails on it. A silent no-op would let a typo pass.
* Widgets default to the value the app passed as ``value=`` (or ``options[index]``), which
  is what a real first render returns. Buttons default to ``False``, which is also what a
  real first render returns - so an unscripted run exercises the app's empty state, and
  that state is worth exercising.
* Payloads handed to ``st.dataframe`` and ``st.download_button`` are kept, not just
  counted. The numbers on the screen are the point; a harness that only recorded that a
  table *happened* would prove nothing.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple


class StopScript(Exception):
    """Raised by ``st.stop()``. Real Streamlit halts the run; so do we."""


class SessionState(dict):
    """``st.session_state``: a dict that also answers to attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:                              # pragma: no cover
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class Call:
    """One recorded API call."""

    __slots__ = ("kind", "label", "args", "kwargs", "value", "payload")

    def __init__(self, kind: str, label: Any = None, args: Tuple = (),
                 kwargs: Optional[Dict[str, Any]] = None, value: Any = None,
                 payload: Any = None) -> None:
        self.kind = kind
        self.label = label
        self.args = args
        self.kwargs = kwargs or {}
        self.value = value
        self.payload = payload

    def __repr__(self) -> str:                               # pragma: no cover
        head = f"{self.kind}({self.label!r})"
        return head if self.value is None else f"{head} -> {self.value!r}"


class _ColumnConfig:
    """``st.column_config``. The app only builds descriptors; nothing renders them."""

    class ProgressColumn:
        def __init__(self, label: str = "", **kwargs: Any) -> None:
            self.label = label
            self.kwargs = kwargs

    class NumberColumn:
        def __init__(self, label: str = "", **kwargs: Any) -> None:
            self.label = label
            self.kwargs = kwargs

    class TextColumn:
        def __init__(self, label: str = "", **kwargs: Any) -> None:
            self.label = label
            self.kwargs = kwargs


class Surface:
    """Anything you can call ``.markdown()`` on.

    Columns, tabs, expanders, containers, forms and the sidebar are all this. In real
    Streamlit they are separate containers with the same method set; here they share one
    recorder, so a metric drawn inside the third column of the second tab lands in the
    same transcript as everything else. Ordering is preserved, nesting is not - and the
    app is asserted on by content, not by tree position.
    """

    def __init__(self, recorder: "Recorder", scope: str = "root") -> None:
        self._rec = recorder
        self._scope = scope

    # -- plumbing ----------------------------------------------------------------
    def __enter__(self) -> "Surface":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def _record(self, kind: str, label: Any = None, *, args: Tuple = (),
                kwargs: Optional[Dict[str, Any]] = None, value: Any = None,
                payload: Any = None) -> Any:
        self._rec.calls.append(Call(kind, label, args, kwargs, value, payload))
        return value

    def _answer(self, label: Any, default: Any, kwargs: Dict[str, Any]) -> Any:
        """Scripted answer for a widget, else the default a real first render gives."""
        key = kwargs.get("key") or label
        script = self._rec.script
        if key in script:
            return script[key]
        if label in script:
            return script[label]
        return default

    # The text methods - markdown, caption, error, warning and friends - are attached
    # below the class body by `_make_text`, because they differ only in the label they
    # record and writing sixteen near-identical three-line methods would bury the ones
    # that do something.

    def __getattr__(self, name: str) -> Any:
        """Serve anything unmodelled as a recording no-op, and remember that we did.

        The alternative - AttributeError - would abort the app run at the first
        unrecognised widget and tell the reader nothing about the rest of the script.
        This way the run completes and ``recorder.unknown`` fails the test.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        self._rec.unknown.add(name)

        def unmodelled(*args: Any, **kwargs: Any) -> None:
            self._record(f"?{name}", args[0] if args else None, args=args, kwargs=kwargs)
            return None
        return unmodelled

    # -- layout -------------------------------------------------------------------
    def columns(self, spec: Any, **kwargs: Any) -> List["Surface"]:
        n = spec if isinstance(spec, int) else len(spec)
        self._record("columns", n, kwargs=kwargs)
        return [Surface(self._rec, f"col{i}") for i in range(n)]

    def tabs(self, labels: List[str]) -> List["Surface"]:
        self._record("tabs", list(labels))
        return [Surface(self._rec, f"tab:{name}") for name in labels]

    def expander(self, label: str = "", **kwargs: Any) -> "Surface":
        self._record("expander", label, kwargs=kwargs)
        return Surface(self._rec, f"expander:{label}")

    def container(self, **kwargs: Any) -> "Surface":
        self._record("container", kwargs=kwargs)
        return Surface(self._rec, "container")

    def form(self, key: str = "", **kwargs: Any) -> "Surface":
        self._record("form", key, kwargs=kwargs)
        return Surface(self._rec, f"form:{key}")

    def spinner(self, text: str = "", **kwargs: Any) -> "Surface":
        self._record("spinner", text, kwargs=kwargs)
        return Surface(self._rec, "spinner")

    def empty(self) -> "Surface":
        self._record("empty")
        return Surface(self._rec, "empty")

    def divider(self) -> None:
        self._record("divider")

    def set_page_config(self, **kwargs: Any) -> None:
        self._record("set_page_config", kwargs.get("page_title"), kwargs=kwargs)

    def stop(self) -> None:
        self._record("stop")
        raise StopScript("st.stop()")

    # -- output -------------------------------------------------------------------
    def metric(self, label: str, value: Any = None, **kwargs: Any) -> None:
        self._record("metric", label, kwargs=kwargs, payload=value)

    def dataframe(self, data: Any = None, **kwargs: Any) -> None:
        self._record("dataframe", self._scope, kwargs=kwargs, payload=data)

    def table(self, data: Any = None, **kwargs: Any) -> None:
        self._record("table", self._scope, kwargs=kwargs, payload=data)

    def json(self, obj: Any = None, **kwargs: Any) -> None:
        self._record("json", self._scope, payload=obj)

    def bar_chart(self, data: Any = None, **kwargs: Any) -> None:
        self._record("chart:bar", self._scope, kwargs=kwargs, payload=data)

    def line_chart(self, data: Any = None, **kwargs: Any) -> None:
        self._record("chart:line", self._scope, kwargs=kwargs, payload=data)

    def area_chart(self, data: Any = None, **kwargs: Any) -> None:
        self._record("chart:area", self._scope, kwargs=kwargs, payload=data)

    def plotly_chart(self, fig: Any = None, **kwargs: Any) -> None:
        self._record("chart:plotly", self._scope, kwargs=kwargs, payload=fig)

    def pyplot(self, fig: Any = None, **kwargs: Any) -> None:
        self._record("chart:pyplot", self._scope, payload=fig)

    # -- input --------------------------------------------------------------------
    def _press(self, kind: str, label: str, kwargs: Dict[str, Any]) -> bool:
        """Shared by the three button flavours, including the ``disabled`` guard.

        A disabled button returns ``False`` no matter what the script says, because that is
        what a browser does. Ignoring ``disabled`` would let a test drive the app into a
        state a user cannot reach - pressing *Score* with no file chosen, say, and then
        blaming ``read_flows(None)`` on the app.
        """
        pressed = bool(self._answer(label, False, kwargs))
        if kwargs.get("disabled"):
            pressed = False
        return bool(self._record(kind, label, kwargs=kwargs, value=pressed))

    def button(self, label: str = "", **kwargs: Any) -> bool:
        return self._press("button", label, kwargs)

    def form_submit_button(self, label: str = "", **kwargs: Any) -> bool:
        return self._press("form_submit_button", label, kwargs)

    def download_button(self, label: str = "", data: Any = None, **kwargs: Any) -> bool:
        # `data` is kept: this is the only place the app hands over a full export, and the
        # invariant worth checking is that the download holds every row even when the
        # table on screen is filtered.
        self._record("download_button", label, kwargs=kwargs, value=False, payload=data)
        return False

    def checkbox(self, label: str = "", value: bool = False, **kwargs: Any) -> bool:
        return bool(self._record("checkbox", label, kwargs=kwargs,
                                 value=self._answer(label, value, kwargs)))

    def toggle(self, label: str = "", value: bool = False, **kwargs: Any) -> bool:
        return self.checkbox(label, value, **kwargs)

    def text_input(self, label: str = "", value: str = "", **kwargs: Any) -> str:
        return self._record("text_input", label, kwargs=kwargs,
                            value=self._answer(label, value, kwargs))

    def text_area(self, label: str = "", value: str = "", **kwargs: Any) -> str:
        return self._record("text_area", label, kwargs=kwargs,
                            value=self._answer(label, value, kwargs))

    def number_input(self, label: str = "", min_value: Any = None, max_value: Any = None,
                     value: Any = None, step: Any = None, **kwargs: Any) -> Any:
        default = value if value is not None else (min_value if min_value is not None else 0)
        answer = self._answer(label, default, kwargs)
        # Real Streamlit clamps to the declared range. Reproduced because the app derives
        # a row index from one of these, and an out-of-range index would fail here for a
        # reason the real app would never hit.
        if min_value is not None and answer is not None and answer < min_value:
            answer = min_value
        if max_value is not None and answer is not None and answer > max_value:
            answer = max_value
        return self._record("number_input", label, kwargs=kwargs, value=answer)

    def slider(self, label: str = "", min_value: Any = None, max_value: Any = None,
               value: Any = None, step: Any = None, **kwargs: Any) -> Any:
        return self._record("slider", label, kwargs=kwargs,
                            value=self._answer(label, value, kwargs))

    def select_slider(self, label: str = "", options: Any = (), value: Any = None,
                      **kwargs: Any) -> Any:
        return self._record("select_slider", label, kwargs=kwargs,
                            value=self._answer(label, value, kwargs))

    def selectbox(self, label: str = "", options: Any = (), index: int = 0,
                  **kwargs: Any) -> Any:
        opts = list(options)
        default = opts[index] if 0 <= index < len(opts) else None
        answer = self._answer(label, default, kwargs)
        # A script may name an option that this render does not offer - a sample file that
        # was not written, say. Falling back to the default keeps the run going, and the
        # recorded options let the test see what was actually available.
        if answer not in opts and opts:
            answer = default
        return self._record("selectbox", label, kwargs=kwargs, value=answer,
                            payload=opts)

    def multiselect(self, label: str = "", options: Any = (), default: Any = None,
                    **kwargs: Any) -> List[Any]:
        return self._record("multiselect", label, kwargs=kwargs,
                            value=list(self._answer(label, default or [], kwargs)),
                            payload=list(options))

    def radio(self, label: str = "", options: Any = (), index: int = 0,
              **kwargs: Any) -> Any:
        return self.selectbox(label, options, index, **kwargs)

    def file_uploader(self, label: str = "", **kwargs: Any) -> Any:
        return self._record("file_uploader", label, kwargs=kwargs,
                            value=self._answer(label, None, kwargs))


def _make_text(kind: str):
    """Build one text-rendering method. See the note inside :class:`Surface`."""
    def fn(self: Surface, body: Any = "", *args: Any, **kwargs: Any) -> None:
        self._record(kind, str(body), args=args, kwargs=kwargs)
    fn.__name__ = kind
    return fn


TEXT_METHODS = ("markdown", "caption", "write", "text", "title", "header", "subheader",
                "code", "latex", "error", "warning", "info", "success", "exception",
                "toast", "help")

for _name in TEXT_METHODS:
    setattr(Surface, _name, _make_text(_name))
del _name


class _Cache:
    """``st.cache_resource`` / ``st.cache_data``.

    Caches for real, keyed on the arguments, because the app relies on it: the detector is
    loaded once and its two thresholds are then mutated in place by
    ``ui.apply_thresholds``. A decorator that re-ran the function would hand out a fresh
    detector on every call and quietly undo the slider.
    """

    def __init__(self, recorder: "Recorder", kind: str) -> None:
        self._rec = recorder
        self._kind = kind

    def __call__(self, func: Any = None, **_kwargs: Any) -> Any:
        if func is None:                       # used as @st.cache_resource(...)
            return self.__call__
        store: Dict[Any, Any] = {}

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            if key not in store:
                self._rec.calls.append(Call(f"{self._kind}:miss", func.__name__))
                store[key] = func(*args, **kwargs)
            return store[key]

        wrapper.clear = store.clear            # type: ignore[attr-defined]
        wrapper.__name__ = getattr(func, "__name__", "cached")
        wrapper.__wrapped__ = func             # type: ignore[attr-defined]
        return wrapper


class Recorder(Surface):
    """The object that stands in for the ``streamlit`` module."""

    def __init__(self, script: Optional[Dict[Any, Any]] = None,
                 session_state: Optional[SessionState] = None) -> None:
        super().__init__(self, "root")
        self.calls: List[Call] = []
        self.script: Dict[Any, Any] = dict(script or {})
        self.unknown: set = set()
        # Passing a session_state in models Streamlit's central fact: the script reruns,
        # the state does not. Bugs that only appear on the *second* interaction - a stale
        # explanation from the previous upload, say - are invisible without it.
        self.session_state = session_state if session_state is not None else SessionState()
        self.sidebar = Surface(self, "sidebar")
        self.column_config = _ColumnConfig()
        self.cache_resource = _Cache(self, "cache_resource")
        self.cache_data = _Cache(self, "cache_data")
        self.__version__ = "0.0.0-stub"

    # -- reading the transcript -----------------------------------------------------
    def of(self, *kinds: str) -> List[Call]:
        return [c for c in self.calls if c.kind in kinds]

    def texts(self) -> List[str]:
        """Every string the app rendered, in order."""
        wanted = {"markdown", "caption", "write", "text", "title", "header", "subheader",
                  "code", "error", "warning", "info", "success", "metric"}
        return [str(c.label) for c in self.calls if c.kind in wanted]

    def text_containing(self, needle: str) -> List[str]:
        low = needle.lower()
        return [t for t in self.texts() if low in t.lower()]

    def said(self, needle: str) -> bool:
        return bool(self.text_containing(needle))

    def metrics(self) -> Dict[str, Any]:
        """label -> value, last write wins, which is what the screen shows."""
        return {str(c.label): c.payload for c in self.calls if c.kind == "metric"}

    def frames(self) -> List[Any]:
        return [c.payload for c in self.calls
                if c.kind in ("dataframe", "table") and c.payload is not None]

    def charts(self) -> List[Call]:
        return [c for c in self.calls if c.kind.startswith("chart:")]

    def downloads(self) -> Dict[str, Any]:
        return {str(c.label): c.payload for c in self.calls if c.kind == "download_button"}

    def widget_values(self) -> Dict[str, Any]:
        kinds = {"button", "checkbox", "slider", "selectbox", "number_input",
                 "text_input", "file_uploader", "form_submit_button", "radio",
                 "multiselect", "select_slider"}
        return {str(c.label): c.value for c in self.calls if c.kind in kinds}

    def errors(self) -> List[str]:
        return [str(c.label) for c in self.calls if c.kind in ("error", "exception")]

    def warnings(self) -> List[str]:
        return [str(c.label) for c in self.calls if c.kind == "warning"]

    def summary(self) -> str:
        counts: Dict[str, int] = {}
        for c in self.calls:
            counts[c.kind] = counts.get(c.kind, 0) + 1
        head = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return f"{len(self.calls)} call(s): {head}"


def install(script: Optional[Dict[Any, Any]] = None,
            session_state: Optional[SessionState] = None) -> Recorder:
    """Put a fresh recorder in ``sys.modules`` as ``streamlit`` and return it.

    Pass the same ``session_state`` to two installs to model two reruns of one browser
    session; omit it for a first visit.

    Only the top-level module is registered. Submodules the app does not import - such as
    ``streamlit.components.v1`` - are left absent on purpose: inventing API surface a stub
    has never been asked for is how it starts lying about the real thing.
    """
    rec = Recorder(script, session_state)
    sys.modules["streamlit"] = rec           # type: ignore[assignment]
    return rec


def uninstall() -> None:
    sys.modules.pop("streamlit", None)
