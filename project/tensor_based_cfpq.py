import networkx as nx
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from pyformlang.rsa import RecursiveAutomaton
from scipy import sparse

from project.finite_automata import graph_to_nfa
from project.regular_query import AdjacencyMatrixFA, intersect_automata


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Translates grammar EBNF description into recursive automaton"""
    rsm = RecursiveAutomaton.from_text(ebnf)
    return rsm


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Transforsms context-free grammar into recursive automation"""
    ebnf = cfg.to_text()
    return ebnf_to_rsm(ebnf)


def rsm_to_adj(rsm: RecursiveAutomaton) -> AdjacencyMatrixFA:
    """Tranforms RecursiveAutomaton into adjacency matrix of finite automaton"""
    nfa = NondeterministicFiniteAutomaton()
    for non_terminal, box in rsm.boxes.items():
        automaton = box.dfa
        for start_state in automaton.start_states:
            state = State((non_terminal, start_state))
            nfa.add_start_state(state)
        for final_state in automaton.final_states:
            state = State((non_terminal, final_state))
            nfa.add_final_state(state)
        automaton_dict = automaton.to_dict()
        for source, inner in automaton_dict.items():
            source_state = State((non_terminal, source))
            for label, dest in inner.items():
                dest_state = State((non_terminal, dest))
                nfa.add_transition(source_state, label, dest_state)
    return AdjacencyMatrixFA(nfa)


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Reachability function, based on tensor multiplication algorithm"""
    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)
    rsa_adj = rsm_to_adj(rsm)
    graph_adj = AdjacencyMatrixFA(
        graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    )
    graph_alphabet = graph_adj.boolean_decompositions.keys()
    for non_terminal in rsm.boxes.keys():
        if non_terminal not in graph_alphabet:
            graph_adj.boolean_decompositions[non_terminal] = sparse.dok_matrix(
                (graph_adj.states_len, graph_adj.states_len), dtype=bool
            )
    old_nnz = 0
    while True:
        intersection_adj = intersect_automata(rsa_adj, graph_adj)
        transitive_closure = intersection_adj.get_transitive_closure()
        intersection_idx_to_state = {
            idx: st for st, idx in intersection_adj.state_to_idx.items()
        }
        for row_idx, column_idx in zip(*transitive_closure.nonzero()):
            row_state = intersection_idx_to_state[row_idx]
            row_inner_state, row_graph_state = row_state.value
            row_symbol, row_rsm_state = row_inner_state.value
            column_state = intersection_idx_to_state[column_idx]
            column_inner_state, column_graph_state = column_state.value
            column_symbol, column_rsm_state = column_inner_state.value
            if row_symbol != column_symbol:
                continue
            box = rsm.boxes[row_symbol]
            automaton = box.dfa
            if row_rsm_state not in automaton.start_states:
                continue
            if column_rsm_state not in automaton.final_states:
                continue
            row_graph_idx = graph_adj.state_to_idx[row_graph_state]
            column_graph_idx = graph_adj.state_to_idx[column_graph_state]
            graph_adj.boolean_decompositions[row_symbol][
                row_graph_idx, column_graph_idx
            ] = True
        new_nnz = sum(
            graph_adj.boolean_decompositions[symbol].count_nonzero()
            for symbol in graph_alphabet
        )
        if new_nnz == old_nnz:
            break
        old_nnz = new_nnz
    answer = set()
    start_label = rsm.initial_label
    for start in graph_adj.start_states:
        for finish in graph_adj.final_states:
            if graph_adj.boolean_decompositions[start_label][
                graph_adj.state_to_idx[start], graph_adj.state_to_idx[finish]
            ]:
                answer.add((start, finish))
    return answer
