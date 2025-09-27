from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import Symbol
from scipy.sparse import coo_array
import numpy as np
from project.regular_query import AdjacencyMatrixFA


def test_init():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions([(0, "a", 1), (0, "d", 1)])
    nfa.add_start_state(0)
    nfa.add_final_state(1)
    amfa = AdjacencyMatrixFA(nfa)

    expected = {
        "a": coo_array(([1], ([0], [1])), shape=(2, 2)),
        "d": coo_array(([1], ([0], [1])), shape=(2, 2))
    }

    for s, arr in expected.items():
        assert s in amfa.adj_matrices
        assert np.array_equal(amfa.adj_matrices[s].toarray(),
                             arr.toarray())
    assert amfa.start_states == set([0])
    assert amfa.final_states == set([1])
    assert amfa.states_num == 2

def test_accepts():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions([(0, "a", 1), (0, "a", 0), (0, "d", 1)])
    nfa.add_start_state(0)
    nfa.add_final_state(1)
    amfa = AdjacencyMatrixFA(nfa)

    assert amfa.accepts("aaaaaad")

def test_not_accepts():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions([(0, "a", 1), (0, "d", 1)])
    nfa.add_start_state(0)
    nfa.add_final_state(1)
    amfa = AdjacencyMatrixFA(nfa)

    assert not amfa.accepts("pmr")

def test_accepts_word_symbols():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions([(0, "abcdefg", 1), (0, "d", 1)])
    nfa.add_start_state(0)
    nfa.add_final_state(1)
    amfa = AdjacencyMatrixFA(nfa)

    assert amfa.accepts([Symbol("abcdefg")])


def test_not_accepts_word_symbols():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions([(0, "a", 0), (0, "d", 1)])
    nfa.add_start_state(0)
    nfa.add_final_state(1)
    amfa = AdjacencyMatrixFA(nfa)

    assert not amfa.accepts([Symbol("aaa")])

def test_is_empty():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions([(0, "a", 0), (1, "d", 1)])
    nfa.add_start_state(0)
    nfa.add_final_state(1)
    amfa = AdjacencyMatrixFA(nfa)

    assert amfa.is_empty()

def test_is_not_empty():
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions([(0, "a", 0), (0, "a", 1), (1, "d", 1)])
    nfa.add_start_state(0)
    nfa.add_final_state(1)
    amfa = AdjacencyMatrixFA(nfa)

    assert not amfa.is_empty()
    assert amfa.accepts("aa")
