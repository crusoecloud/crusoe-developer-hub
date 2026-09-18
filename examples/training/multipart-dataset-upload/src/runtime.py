"""Dual-mode prompts that work in both a CLI and a Jupyter notebook.

On a real TTY, questionary renders arrow-key menus and Y/n prompts. In a notebook,
a piped shell, or when questionary is not installed, a zero-dependency fallback
prints numbered menus and reads from `input()`. Secrets are read with `getpass`,
which masks on a TTY and routes to the Jupyter password widget in a kernel.

Public surface (each blocks until the user answers):

    pick(options, prompt="Pick", default=None) -> str
    confirm(prompt, default=True) -> bool
    text(prompt, default="") -> str
    path(prompt="Path", default="") -> str
    password(prompt="Secret") -> str
    abort(message, status=1) -> NoReturn
"""

from __future__ import annotations

import getpass
import sys
from collections.abc import Callable
from typing import NoReturn, TypeVar

_T = TypeVar("_T")


class WorkflowError(Exception):
    """Stops a notebook cell cleanly (raised by abort() inside a Jupyter kernel)."""


def is_ipython_kernel() -> bool:
    """True when running inside an IPython/Jupyter kernel (JupyterLab, VS Code, Colab)."""
    try:
        shell = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    # Terminal IPython also defines get_ipython(), but only kernels carry a .kernel.
    return getattr(shell, "kernel", None) is not None


def abort(message: str, status: int = 1) -> NoReturn:
    """Stop the run: raise WorkflowError inside a Jupyter kernel, else print to stderr and exit."""
    if is_ipython_kernel():
        raise WorkflowError(message)
    print(message, file=sys.stderr)
    sys.exit(status)


# --- mode selection ----------------------------------------------------------


def _questionary():
    """The questionary module when it can drive a real TTY, else None (use input())."""
    if not sys.stdin.isatty():
        return None
    try:
        import questionary
    except ImportError:
        return None
    return questionary


def _ask(query: Callable[[object], _T], fallback: Callable[[], _T]) -> _T:
    """Answer via questionary on a TTY, else the input()-based `fallback`; Ctrl-C/EOF aborts."""
    questionary = _questionary()
    if questionary is None:
        return _guard(fallback)
    try:
        return query(questionary)
    except (KeyboardInterrupt, EOFError):
        abort("cancelled by user")
    except WorkflowError:
        raise
    except Exception:  # noqa: BLE001 - a rare questionary runtime error falls back to input()
        return _guard(fallback)


def _guard(fallback: Callable[[], _T]) -> _T:
    """Run an input()-based prompt, turning Ctrl-C/EOF into a clean abort."""
    try:
        return fallback()
    except (KeyboardInterrupt, EOFError):
        abort("cancelled by user")


# --- input() fallbacks (one per prompt kind; `read` is injected in tests) -----


def _input_pick(options: list[str], prompt: str, read: Callable[[str], str] = input) -> str:
    for index, option in enumerate(options, 1):
        print(f"  {index}. {option}")
    while True:
        raw = read(f"{prompt} [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  enter a number 1-{len(options)}")


def _input_confirm(prompt: str, default: bool, read: Callable[[str], str] = input) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = read(f"{prompt} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def _input_text(prompt: str, default: str, read: Callable[[str], str] = input) -> str:
    return read(f"{prompt} [{default}]: ").strip() or default


# --- public prompts ----------------------------------------------------------


def pick(options: list[str], prompt: str = "Pick", default: str | None = None) -> str:
    """Single-choice menu (arrow keys on a TTY, numbered list otherwise)."""
    if not options:
        raise ValueError("pick() requires a non-empty options list")
    return _ask(
        lambda q: q.select(prompt, choices=options, default=default or options[0]).unsafe_ask(),
        lambda: _input_pick(options, prompt),
    )


def confirm(prompt: str, default: bool = True) -> bool:
    """Yes/no prompt; `default` is the answer for a bare Enter."""
    return _ask(
        lambda q: bool(q.confirm(prompt, default=default).unsafe_ask()),
        lambda: _input_confirm(prompt, default),
    )


def text(prompt: str, default: str = "") -> str:
    """Free-text prompt with an optional default."""
    return _ask(
        lambda q: q.text(prompt, default=str(default)).unsafe_ask(),
        lambda: _input_text(prompt, default),
    )


def autocomplete_available() -> bool:
    """True when path prompts can offer Tab autocomplete (a TTY with questionary present)."""
    return _questionary() is not None


def path(prompt: str = "Path", default: str = "") -> str:
    """Filesystem path input with Tab autocomplete on a TTY, else plain input()."""
    return _ask(
        lambda q: q.path(prompt, default=default).unsafe_ask(),
        lambda: _input_text(prompt, default),
    )


def password(prompt: str = "Secret", read: Callable[[str], str] = getpass.getpass) -> str:
    """Masked secret input (hidden on a TTY, password widget in Jupyter); `read` is injectable for tests."""
    try:
        return read(f"{prompt}: ")
    except (KeyboardInterrupt, EOFError):
        abort("cancelled by user")
