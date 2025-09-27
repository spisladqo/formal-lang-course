from typing import Iterable
import numpy as np
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import Symbol
from dataclasses import dataclass
from scipy.sparse import coo_array, csr_array, identity

@dataclass
class AdjacencyMatrixFA:
    adj_matrices: dict[coo_array] # TODO better to use dict[csr_matrix]
    start_states: set # TODO better to use csr_matrix
    final_states: set # TODO better to use csr_matrix
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
                for v in v2:
                    row[s].append(v1.value)
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

        return (result[0, 0] != 0)

    def is_empty(self) -> bool:
        n = self.states_num
        row = [state.value for state in self.start_states]
        col = [0] * len(row)
        data = [1] * len(row)
        matrix = coo_array((data, (row, col)), shape=(1, n))

        tc = self.trans_clos()
        matrix = matrix.dot(tc)

        row = [state.value for state in self.final_states]
        col = [0] * len(row)
        data = [1] * len(row)
        final = coo_array((data, (row, col)), shape=(n, 1))
        result = matrix.dot(final) # a (1x1) matrix

        return (result[0, 0] == 0)

    def trans_clos(self) -> csr_array:
        n = self.states_num
        closure = csr_array((n, n), dtype=np.int8)
        for _, m in self.adj_matrices.items():
            closure = closure + m.tocsr()
        iden = identity(n, dtype='bool', format='csr')
        print(iden)

        print(closure)
        closure = closure + iden / 2
        print(closure)

        changed = True
        while changed:
            new_closure = closure.dot(closure)
            print(new_closure)
            ineq_matr = (closure != new_closure)
            changed = (ineq_matr.nnz != 0)
            closure = new_closure

        return closure
