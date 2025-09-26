from typing import Iterable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import Symbol
from dataclasses import dataclass
from scipy.sparse import coo_array

@dataclass
class AdjacencyMatrixFA:
    adj_matrices: dict[coo_array]
    start_states: list
    final_states: list
    states_num: int

    def __init__(self, nfa: NondeterministicFiniteAutomaton):
        self.start_states = nfa.start_states
        self.final_states = nfa.final_states
        self.states_num = len(nfa.states)

        delta = nfa.to_dict()
        row = {k: [] for k in nfa.symbols}
        col = {k: [] for k in nfa.symbols}

        for v1, edge in delta.items():
            for s, v2 in edge.items():
                row[s].append(v1.value)
                for v in v2:
                    col[s].append(v.value)

        self.adj_matrices = {}
        for s in nfa.symbols:
            data = [1] * len(row[s])
            n = self.states_num
            coo = coo_array((data, (row[s], col[s])), shape = (n, n))
            self.adj_matrices[s] = coo

    def accepts(self, word: Iterable[Symbol]) -> bool:
        n = self.states_num
        row = [state.value for state in self.start_states]
        col = [0] * len(row)
        data = [1] * len(row)
        m = coo_array((data, (row, col)), shape=(1, n))

        for symbol in word:
            if symbol not in self.adj_matrices:
                return False
            m = m.dot(self.adj_matrices[symbol])

        row = [state.value for state in self.final_states]
        col = [0] * len(row)
        data = [1] * len(row)
        final = coo_array((data, (row, col)), shape=(n, 1))
        result = m.dot(final) # a (1x1) matrix

        return (result[0, 0] > 0)
