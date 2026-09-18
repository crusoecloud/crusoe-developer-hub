"""Unit tests for src/parameters.py: bounded-int parsing/reprompt and gathering the upload
parameters through injected prompts (no real prompts, no network)."""

from __future__ import annotations

import pytest

from src import constants, parameters
from src.state import FileConfig, State, UploadParams


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("3", 3, id="in-range"),
        pytest.param("1", 1, id="low-bound"),
        pytest.param("10", 10, id="high-bound"),
        pytest.param("0", None, id="below-range"),
        pytest.param("11", None, id="above-range"),
        pytest.param("abc", None, id="not-an-int"),
        pytest.param("", None, id="empty"),
    ],
)
def test_parse_bounded_int(raw, expected):
    assert parameters._parse_bounded_int(raw, 1, 10) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"purpose": "nonsense"}, id="bad-purpose"),
        pytest.param({"part_size": 0}, id="part-size-zero"),
        pytest.param({"workers": 0}, id="workers-zero"),
        pytest.param({"num_parts": 0}, id="num-parts-zero"),
    ],
)
def test_upload_params_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        UploadParams(**kwargs)


def test_prompt_int_reprompts_until_in_range():
    # Below-range, then above-range, then valid: the loop must reject the first two.
    answers = iter(["0", "999", "3"])
    prompt = parameters._IntPrompt("part size", default=5, lo=1, hi=10)
    assert parameters._prompt_int(prompt, text=lambda *a, **k: next(answers)) == 3


def test_choose_params_gathers_all_fields(tmp_path):
    state = State(api_key="k", state_file_path=str(tmp_path / "state.json"))
    state.file = FileConfig(size=200 * constants.MiB)
    answers = iter(["64", "8"])  # part size (MiB), then worker count

    parameters.choose_params(state, pick=lambda *a, **k: "batch", text=lambda *a, **k: next(answers))

    assert state.params.purpose == "batch"
    assert state.params.part_size == 64 * constants.MiB
    assert state.params.workers == 8
    assert state.params.num_parts == state.expected_num_parts() == 4  # ceil(200 / 64)
