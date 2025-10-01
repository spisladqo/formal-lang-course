from typing import Iterable
from networkx import MultiDiGraph
import numpy as np
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton
from pyformlang.finite_automaton import State, Symbol
from dataclasses import dataclass
from project.finite_automata import graph_to_nfa, regex_to_dfa
from scipy.sparse import coo_array, csr_array, identity, kron


@dataclass
class AdjacencyMatrixFA:
    adj_matrices: dict[coo_array]
    start_states: set
    final_states: set
    states_num: int
    state_to_index: dict

    def __init__(self, nfa: NondeterministicFiniteAutomaton):
        self.start_states = nfa.start_states
        self.final_states = nfa.final_states

        states_list = list(nfa.states)
        self.states_num = len(states_list)
        self.state_to_index = {state: idx for idx, state in enumerate(states_list)}

        delta = nfa.to_dict()
        row = {symbol: [] for symbol in nfa.symbols}
        col = {symbol: [] for symbol in nfa.symbols}

        for v1, transitions in delta.items():
            for symbol, v2 in transitions.items():
                if isinstance(v2, set):
                    for state in v2:
                        row[symbol].append(v1.value)
                        col[symbol].append(state.value)
                else:
                    row[symbol].append(self.state_to_index[v1])
                    col[symbol].append(self.state_to_index[v2])

        self.adj_matrices = {}
        for symbol in nfa.symbols:
            if row[symbol]:
                data = [1] * len(row[symbol])
                coo = coo_array(
                    (data, (row[symbol], col[symbol])),
                    shape=(self.states_num, self.states_num),
                )
                self.adj_matrices[symbol] = coo

    def accepts(self, word: Iterable[Symbol]) -> bool:
        if not word:
            return any(state in self.final_states for state in self.start_states)

        n = self.states_num
        start_indices = [state.value for state in self.start_states]
        row = [0] * len(start_indices)
        col = start_indices
        data = [1] * len(start_indices)
        m = coo_array((data, (row, col)), shape=(1, n))

        for symbol in word:
            if symbol not in self.adj_matrices:
                return False
            m = m @ self.adj_matrices[symbol]

        final_indices = [state.value for state in self.final_states]
        row = final_indices
        col = [0] * len(final_indices)
        data = [1] * len(final_indices)
        final = coo_array((data, (row, col)), shape=(n, 1))
        result = m @ final

        return result[0, 0] != 0

    def is_empty(self) -> bool:
        if not self.start_states:
            return True

        n = self.states_num
        start_indices = [state.value for state in self.start_states]
        row = [0] * len(start_indices)
        col = start_indices
        data = [1] * len(start_indices)
        matrix = coo_array((data, (row, col)), shape=(1, n))

        tc = self.trans_clos()
        matrix = matrix @ tc

        final_indices = [state.value for state in self.final_states]
        row = final_indices
        col = [0] * len(final_indices)
        data = [1] * len(final_indices)
        final = coo_array((data, (row, col)), shape=(n, 1))
        result = matrix @ final

        return result[0, 0] == 0

    def trans_clos(self) -> csr_array:
        n = self.states_num
        if not self.adj_matrices:
            return identity(n, format="csr", dtype=np.bool_)

        closure = sum(
            (matrix.astype(np.bool_) for matrix in self.adj_matrices.values()),
            csr_array((n, n), dtype=np.bool_),
        )

        closure = closure + identity(n, dtype=np.bool_, format="csr")

        changed = True
        while changed:
            new_closure = closure @ closure
            changed = (new_closure != closure).nnz > 0
            closure = new_closure

        return closure


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    common_symbols = set(automaton1.adj_matrices.keys()) & set(
        automaton2.adj_matrices.keys()
    )
    adj_matrices = {}

    for symbol in common_symbols:
        adj_matrices[symbol] = kron(
            automaton1.adj_matrices[symbol], automaton2.adj_matrices[symbol]
        )

    start_states = set()
    final_states = set()

    for s1 in automaton1.start_states:
        for s2 in automaton2.start_states:
            idx1 = automaton1.state_to_index[s1]
            idx2 = automaton2.state_to_index[s2]
            new_idx = idx1 * automaton2.states_num + idx2
            start_states.add(State(new_idx))

    for s1 in automaton1.final_states:
        for s2 in automaton2.final_states:
            idx1 = automaton1.state_to_index[s1]
            idx2 = automaton2.state_to_index[s2]
            new_idx = idx1 * automaton2.states_num + idx2
            final_states.add(State(new_idx))

    automaton3 = AdjacencyMatrixFA.__new__(AdjacencyMatrixFA)
    automaton3.adj_matrices = adj_matrices
    automaton3.start_states = start_states
    automaton3.final_states = final_states
    automaton3.states_num = automaton1.states_num * automaton2.states_num

    states_list = list(range(automaton3.states_num))
    automaton3.state_to_index = {State(i): i for i in states_list}

    return automaton3


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    autom1 = AdjacencyMatrixFA(graph_nfa)
    autom2 = AdjacencyMatrixFA(regex_dfa)

    autom3 = intersect_automata(autom1, autom2)
    closure = autom3.trans_clos()

    pairs = set()

    for start_state in autom3.start_states:
        start_idx = start_state.value
        if start_idx >= closure.shape[0]:
            continue

        for final_state in autom3.final_states:
            final_idx = final_state.value
            if final_idx >= closure.shape[1]:
                continue

            if closure[start_idx, final_idx]:
                original_start_idx = start_idx // autom2.states_num
                original_final_idx = final_idx // autom2.states_num

                original_start_node = None
                original_final_node = None

                for state, idx in autom1.state_to_index.items():
                    if idx == original_start_idx:
                        original_start_node = state.value
                    if idx == original_final_idx:
                        original_final_node = state.value

                if original_start_node is not None and original_final_node is not None:
                    pairs.add((original_start_node, original_final_node))

    return pairs
