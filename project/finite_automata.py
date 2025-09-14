from typing import Set
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.finite_automaton import State, Symbol
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    re = Regex(regex)
    dfa = re.to_epsilon_nfa()
    return dfa


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()
    if not start_states:
        start_states = set(graph.nodes)
    if not final_states:
        final_states = set(graph.nodes)

    for s in start_states:
        nfa.add_start_state(State(s))

    for f in final_states:
        nfa.add_final_state(State(f))

    for u, k, v in list(graph.edges):
        nfa.add_transition(State(u), Symbol(k), State(v))

    return nfa
