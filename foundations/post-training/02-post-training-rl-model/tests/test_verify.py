"""The verifier is pure Python plus SymPy, so these run offline in seconds."""

import pytest

from reasoning_utils.verify import extract_final_answer, grade, last_boxed, normalize_answer, split_parts

# (prediction, reference answer, expected verdict)
CASES = [
    ("3/4", r"\frac{3}{4}", True),
    ("(3)/(4)", "3/4", True),
    (r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
    (r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
    ("(1, 2)", "(1,2)", True),
    ("(2, 1)", "(1, 2)", False),
    ("(1, 2, 3)", "(1, 2)", False),
    ("0.5", "1/2", True),
    ("0.3333333333", "1/3", False),
    ("1,234/2", "617", True),
    (r"\text{2/3}", "2/3", True),
    ("50%", "1/2", False),
    (r"2\cdot 3/4", "3/2", True),
    (r"90^\circ", "90", True),
    (r"90^{\circ}", "90", True),
    (r"\left(\frac{3}{4}\right)", "3/4", True),
    ("2²", "2**2", True),
    ("[1, 2]", "(1, 2)", True),
    ("1e-3", "0.001", True),
    ("(-3)^2", "9", True),
    ("−1", "-1", True),  # unicode minus
    (r"\dfrac{14}{3}", "14/3", True),
    ("28/6", "14/3", True),
    ("x**2 + 2x + 1", "(x+1)^2", True),
    ("c. 3", "3", True),
    ("", "3", False),
    (None, "3", False),
]


@pytest.mark.parametrize("prediction,truth,expected", CASES)
def test_grade(prediction, truth, expected):
    assert grade(prediction, truth) is expected


def test_last_boxed_handles_nesting_and_whitespace():
    assert last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed {7}") == "7"
    assert last_boxed(r"first \boxed{1} then \boxed{2}") == "2"
    assert last_boxed(r"\boxed{unbalanced") is None
    assert last_boxed("no box here") is None


def test_extract_final_answer_fallbacks():
    assert extract_final_answer(r"Final: \[\boxed{\dfrac{14}{3}}\]") == r"\dfrac{14}{3}"
    assert extract_final_answer(r"\boxed{ 14/3. }") == "14/3."
    assert extract_final_answer("so the answer is 42 apples") == "42"
    assert extract_final_answer("abc 14/3 then 2.5") == "2.5"
    assert extract_final_answer("no digits at all") == "no digits at all"
    assert extract_final_answer("") == ""


def test_normalize_answer_examples():
    assert normalize_answer(r"\dfrac{14}{3}") == "(14)/(3)"
    assert normalize_answer(r"\text{\[\frac{14}{3}\]}") == "(14)/(3)"
    assert normalize_answer("$1,234$") == "1234"
    assert normalize_answer("1 1/2") == "1+1/2"
    assert normalize_answer("<|im_end|>5") == "5"


def test_split_parts():
    assert split_parts("(14/3, 2/3)") == ["14/3", "2/3"]
    assert split_parts("[a,b]") == ["a", "b"]
    assert split_parts("(1,)") == ["(1,)"]  # an empty part means it is not a tuple
    assert split_parts("14/3") == ["14/3"]
    assert split_parts("") == []
