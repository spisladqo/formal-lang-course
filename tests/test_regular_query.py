from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import Symbol
from scipy.sparse import coo_array
import numpy as np
from project.regular_query import AdjacencyMatrixFA, intersect_automata


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


def test_intersect_automata():
    nfa1 = NondeterministicFiniteAutomaton()
    nfa1.add_start_state(0)
    nfa1.add_final_state(0)
    nfa1.add_transitions([(0, "a", 0), (1, "a", 1),(0, "b", 1), (1, "b", 0)])
    amfa1 = AdjacencyMatrixFA(nfa1)

    nfa2 = NondeterministicFiniteAutomaton()
    nfa2.add_start_state(0)
    nfa2.add_final_state(1)
    nfa2.add_transitions([(0, "a", 1), (1, "b", 0)])
    amfa2 = AdjacencyMatrixFA(nfa2)

    amfa3 = intersect_automata(amfa1, amfa2)

    expected = {
        "a": coo_array(([1, 1], ([0, 2], [1, 3])), shape=(4, 4)),
        "b": coo_array(([1, 1], ([1, 3], [2, 0])), shape=(4, 4))
    }

    for s, arr in expected.items():
        assert s in amfa3.adj_matrices
        print(s, amfa3.adj_matrices[s])
        assert np.array_equal(amfa3.adj_matrices[s].toarray(),
                             arr.toarray())

    assert amfa3.start_states== {0}
    assert amfa3.final_states == {1}
    assert amfa3.states_num == 4
