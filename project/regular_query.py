from typing import Iterable
import numpy as np
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import Symbol
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

        # dict(node, dict[str, node]) ==
        # dict(int, dict[str, int])
        delta = nfa.to_dict()

        row = dict[list]
        col = dict[list]

        # iterate over delta.keys() (v)
        # for every v.key (s) with value v.value (u)
        # add pair of (v, u, 1) into coo_array
        # and add it to matrix[s]
        for v, edge in delta.items():
            for s, u in edge.items():
                row[s].append(v)
                col[s].append(u)

        for s in delta.symbols:
            data = [1] * len(row[s])
            coo = coo_array(data, (row[s], col[s]))
            self.matrix[s] = coo





