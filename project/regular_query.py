from typing import Iterable
import numpy as np
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import State
from dataclasses import dataclass
from scipy.sparse import coo_array

@dataclass
class AdjacencyMatrixFA:
    matrix: dict[coo_array]
    start_states: list
    final_states: list

    def __init__(self, nfa: NondeterministicFiniteAutomaton):
        self.start_states = nfa.start_states
        self.final_states = nfa.final_states
        delta = nfa.to_dict()
        print(delta)
        row = {k: [] for k in nfa.symbols}
        col = {k: [] for k in nfa.symbols}
        for v1, edge in delta.items():
            for s, v2 in edge.items():
                row[s].append(v1.value)
                for v in v2:
                    col[s].append(v.value)
                print(f"added {v1, s, v2}")
        self.matrix = {}
        for s in nfa.symbols:
            data = [1] * len(row[s])
            n = len(nfa.states)
            coo = coo_array((data, (row[s], col[s])), shape = (n, n))
            self.matrix[s] = coo
