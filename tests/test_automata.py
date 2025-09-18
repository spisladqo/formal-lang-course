from pyformlang.regular_expression.regex_objects import MisformedRegexError
import pytest
from project.finite_automata import regex_to_dfa


def test_regex_to_dfa():
    s = "a* b"
    dfa = regex_to_dfa(s)
    assert dfa.accepts("aaaab")


def test_invalid_regex_to_dfa():
    with pytest.raises(MisformedRegexError):
        regex_to_dfa("**")
