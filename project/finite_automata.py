from typing import Set
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.finite_automaton import State
from pyformlang.regular_expression import Regex
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    re = Regex(regex)
    dfa = re.to_epsilon_nfa().to_deterministic()
    return dfa


def graph_to_nfa(
    graph: MultiDiGraph, start_states: Set[int], final_states: Set[int]
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton().from_networkx(graph)
    if not start_states:
        start_states = set(graph.nodes)
    if not final_states:
        final_states = set(graph.nodes)

    for s in start_states:
        nfa.add_start_state(State(s))
    for f in final_states:
        nfa.add_final_state(State(f))

    return nfa
