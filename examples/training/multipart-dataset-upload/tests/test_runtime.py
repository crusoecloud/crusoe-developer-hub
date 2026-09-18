"""Unit tests for src/runtime.py. The questionary/TTY path isn't exercised here (tests
run without a TTY); the input() fallbacks carry the parsing and reprompt logic, and are
driven through an injected `read` rather than by patching builtins."""

from __future__ import annotations

import pytest

from src import runtime


def _reader(answers):
    """A read() stand-in returning successive answers, ignoring the prompt text."""
    it = iter(answers)
    return lambda _prompt: next(it)


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        pytest.param(["1"], "alpha", id="first"),
        pytest.param(["3"], "gamma", id="last"),
        pytest.param(["9", "0", "2"], "beta", id="reprompts-until-in-range"),
        pytest.param(["x", "2"], "beta", id="reprompts-on-non-number"),
    ],
)
def test_input_pick(answers, expected):
    assert runtime._input_pick(["alpha", "beta", "gamma"], "Pick", read=_reader(answers)) == expected


@pytest.mark.parametrize(
    ("answers", "default", "expected"),
    [
        pytest.param(["y"], True, True, id="yes"),
        pytest.param(["no"], True, False, id="no"),
        pytest.param([""], True, True, id="empty-takes-default-true"),
        pytest.param([""], False, False, id="empty-takes-default-false"),
        pytest.param(["maybe", "n"], True, False, id="reprompts-on-garbage"),
    ],
)
def test_input_confirm(answers, default, expected):
    assert runtime._input_confirm("Sure?", default, read=_reader(answers)) is expected


@pytest.mark.parametrize(
    ("answer", "default", "expected"),
    [
        pytest.param("hello", "", "hello", id="value"),
        pytest.param("", "fallback", "fallback", id="empty-takes-default"),
        pytest.param("  spaced  ", "", "spaced", id="stripped"),
    ],
)
def test_input_text(answer, default, expected):
    assert runtime._input_text("Name", default, read=_reader([answer])) == expected


def test_password_returns_entered_secret():
    assert runtime.password("Key", read=lambda _p: "s3cret") == "s3cret"


def test_password_aborts_on_cancel():
    def _cancel(_prompt):
        raise KeyboardInterrupt

    with pytest.raises(SystemExit):
        runtime.password("Key", read=_cancel)


def test_abort_exits_with_status():
    with pytest.raises(SystemExit) as exc:
        runtime.abort("boom", status=2)
    assert exc.value.code == 2


def test_is_ipython_kernel_false_outside_jupyter():
    assert runtime.is_ipython_kernel() is False
