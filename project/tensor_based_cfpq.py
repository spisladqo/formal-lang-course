import itertools
from collections.abc import Iterable

import numpy as np
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol, State
from scipy.sparse import csr_matrix
import networkx as nx
import scipy.sparse as sp
from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton

from project.finite_automata import graph_to_nfa


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def _build_rsm_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    def _unwrap(x):
        return x.value if hasattr(x, "value") else x

    for symbol in rsm.boxes:
        box = rsm.get_box(symbol)

        for src, dst, label in box.dfa.to_networkx().edges(data="label"):
            nfa.add_transition(
                State((symbol, _unwrap(src))), label, State((symbol, _unwrap(dst)))
            )

        for start_state in box.start_state:
            nfa.add_start_state(State((symbol, _unwrap(start_state))))

        for final_state in box.final_states:
            nfa.add_final_state(State((symbol, _unwrap(final_state))))

    return nfa


class AdjacencyMatrixFA:
    def __init__(self, fa: NondeterministicFiniteAutomaton = None):
        if fa is None:
            self.states = {}
            self.states_count = 0
            self.start_indices = set()
            self.final_indices = set()
            self.boolean_decomposition = {}
            return

        self.states = {state: idx for idx, state in enumerate(fa.states)}
        self.states_count = len(fa.states)
        self.start_indices = {self.states[state] for state in fa.start_states}
        self.final_indices = {self.states[state] for state in fa.final_states}
        self.boolean_decomposition = self._build_boolean_decomposition(fa)

    def _build_boolean_decomposition(self, fa: NondeterministicFiniteAutomaton):
        decomposition = {}
        for src_state, transitions in fa.to_dict().items():
            for symbol, dst_states in transitions.items():
                dst_states = (
                    {dst_states} if not isinstance(dst_states, set) else dst_states
                )
                if symbol not in decomposition:
                    decomposition[symbol] = csr_matrix(
                        (self.states_count, self.states_count), dtype=bool
                    )
                src_idx = self.states[src_state]
                for dst_state in dst_states:
                    dst_idx = self.states[dst_state]
                    decomposition[symbol][src_idx, dst_idx] = True
        return decomposition

    def transitive_closure(self):
        if not self.boolean_decomposition:
            return sp.eye(self.states_count, dtype=bool, format="csr")

        closure = sp.eye(self.states_count, dtype=bool, format="csr")
        closure += sum(self.boolean_decomposition.values())
        powered = np.linalg.matrix_power(closure.toarray(), self.states_count)
        return sp.csr_matrix(powered)

    def accepts(self, word: Iterable[Symbol]) -> bool:
        configs = [(list(word), start_idx) for start_idx in self.start_indices]

        while configs:
            tape, current_state = configs.pop()
            if not tape and current_state in self.final_indices:
                return True
            if tape:
                next_states = self._get_next_states(current_state, tape[0])
                configs.extend((tape[1:], next_state) for next_state in next_states)

        return False

    def _get_next_states(self, current_state: int, symbol: Symbol):
        if symbol not in self.boolean_decomposition:
            return []
        row = self.boolean_decomposition[symbol][current_state]
        return row.nonzero()[1].tolist()

    def is_empty(self) -> bool:
        closure = self.transitive_closure()
        return all(
            not closure[start_idx, final_idx]
            for start_idx in self.start_indices
            for final_idx in self.final_indices
        )


def intersect_automata(
    fa1: AdjacencyMatrixFA, fa2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    result = AdjacencyMatrixFA()
    result.states_count = fa1.states_count * fa2.states_count

    result.states = {
        (s1, s2): (fa1.states[s1] * fa2.states_count + fa2.states[s2])
        for s1, s2 in itertools.product(fa1.states.keys(), fa2.states.keys())
    }

    result.start_indices = {
        s1 * fa2.states_count + s2
        for s1, s2 in itertools.product(fa1.start_indices, fa2.start_indices)
    }

    result.final_indices = {
        f1 * fa2.states_count + f2
        for f1, f2 in itertools.product(fa1.final_indices, fa2.final_indices)
    }

    common_symbols = fa1.boolean_decomposition.keys() & fa2.boolean_decomposition.keys()
    for symbol in common_symbols:
        result.boolean_decomposition[symbol] = sp.kron(
            fa1.boolean_decomposition[symbol],
            fa2.boolean_decomposition[symbol],
            format="csr",
        )

    return result


def _get_product_start_indices(
    intersection: AdjacencyMatrixFA, rsm_matrix: AdjacencyMatrixFA
) -> set[int]:
    rsm_start_states = {
        state
        for state, idx in rsm_matrix.states.items()
        if idx in rsm_matrix.start_indices
    }
    return {
        idx
        for (graph_state, rsm_state), idx in intersection.states.items()
        if rsm_state in rsm_start_states
    }


def _matrix_semiring_bfs(
    intersection: AdjacencyMatrixFA, rsm_matrix: AdjacencyMatrixFA
) -> sp.csr_matrix:
    n = intersection.states_count
    reachability = sp.eye(n, dtype=bool, format="csr")

    for start_idx in _get_product_start_indices(intersection, rsm_matrix):
        reachability[start_idx, start_idx] = True

    matrices = list(intersection.boolean_decomposition.values())
    changed = True
    while changed:
        previous_nnz = reachability.count_nonzero()
        for matrix in matrices:
            reachability = reachability.maximum(reachability @ matrix)
        changed = reachability.count_nonzero() > previous_nnz

    return reachability


def _get_box_label_if_valid_path(
    rsm_matrix: AdjacencyMatrixFA, start_state: State, end_state: State
) -> object | None:
    rsm_start_states = {
        state
        for state, idx in rsm_matrix.states.items()
        if idx in rsm_matrix.start_indices
    }
    rsm_final_states = {
        state
        for state, idx in rsm_matrix.states.items()
        if idx in rsm_matrix.final_indices
    }

    if start_state not in rsm_start_states or end_state not in rsm_final_states:
        return None

    try:
        start_box, _ = start_state.value
        end_box, _ = end_state.value
    except Exception:
        return None

    return start_box if start_box == end_box else None


def _update_graph_with_nonterminals(
    reachability: sp.csr_matrix,
    graph_matrix: AdjacencyMatrixFA,
    rsm_matrix: AdjacencyMatrixFA,
    intersection: AdjacencyMatrixFA,
) -> bool:
    idx_to_state = {idx: pair for pair, idx in intersection.states.items()}
    changed = False

    rows, cols = reachability.nonzero()
    for i, j in zip(rows, cols):
        graph_start, rsm_start = idx_to_state[i]
        graph_end, rsm_end = idx_to_state[j]

        box_label = _get_box_label_if_valid_path(rsm_matrix, rsm_start, rsm_end)
        if box_label is None:
            continue

        if box_label not in graph_matrix.boolean_decomposition:
            graph_matrix.boolean_decomposition[box_label] = sp.csr_matrix(
                (graph_matrix.states_count, graph_matrix.states_count), dtype=bool
            )

        start_idx = graph_matrix.states[graph_start]
        end_idx = graph_matrix.states[graph_end]
        matrix = graph_matrix.boolean_decomposition[box_label]

        if not matrix[start_idx, end_idx]:
            matrix[start_idx, end_idx] = True
            changed = True

    return changed


def _perform_tensor_iteration(
    graph_matrix: AdjacencyMatrixFA, rsm_matrix: AdjacencyMatrixFA
) -> bool:
    intersection = intersect_automata(graph_matrix, rsm_matrix)
    reachability = _matrix_semiring_bfs(intersection, rsm_matrix)
    return _update_graph_with_nonterminals(
        reachability, graph_matrix, rsm_matrix, intersection
    )


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    start_nodes = set(graph.nodes) if start_nodes is None else set(start_nodes)
    final_nodes = set(graph.nodes) if final_nodes is None else set(final_nodes)

    graph_fa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_matrix = AdjacencyMatrixFA(graph_fa)

    rsm_fa = _build_rsm_nfa(rsm)
    rsm_matrix = AdjacencyMatrixFA(rsm_fa)

    while _perform_tensor_iteration(graph_matrix, rsm_matrix):
        continue

    result = set()
    init_label = rsm.initial_label
    if init_label in graph_matrix.boolean_decomposition:
        matrix = graph_matrix.boolean_decomposition[init_label]
        rows, cols = matrix.nonzero()

        idx_to_graph_state = {idx: state for state, idx in graph_matrix.states.items()}

        def _unwrap(x):
            return x.value if hasattr(x, "value") else x

        for i, j in zip(rows, cols):
            u, v = _unwrap(idx_to_graph_state[i]), _unwrap(idx_to_graph_state[j])
            if u in start_nodes and v in final_nodes:
                result.add((u, v))

    return result
