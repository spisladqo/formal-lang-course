from networkx import MultiDiGraph
from project.finite_automata import regex_to_dfa, graph_to_nfa
from project.regular_query import AdjacencyMatrixFA
from scipy import sparse


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_adj = AdjacencyMatrixFA(regex_to_dfa(regex))
    if start_nodes is None:
        start_nodes = set()
    if final_nodes is None:
        final_nodes = set()
    graph_adj = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    front_blocks = []
    graph_state_to_idx = graph_adj.state_to_index
    regex_state_to_idx = regex_adj.state_to_index
    common_symbols = set(graph_adj.adj_matrices.keys()) & set(
        regex_adj.adj_matrices.keys()
    )
    for graph_start_state in graph_adj.start_states:
        graph_start_state_idx = graph_state_to_idx[graph_start_state]
        front_block = sparse.csr_array(
            (graph_adj.states_num, regex_adj.states_num), dtype=bool
        )
        for regex_start_state in regex_adj.start_states:
            regex_start_state_idx = regex_state_to_idx[regex_start_state]
            front_block[graph_start_state_idx, regex_start_state_idx] = True
        front_blocks.append(front_block)
    front = sparse.vstack(front_blocks)
    visited = front.copy()
    graph_adj_matrices = graph_adj.adj_matrices
    regex_adj_matrices = regex_adj.adj_matrices
    graph_adj_matrices_t = {
        symbol: decomposition.transpose().tocsr()
        for symbol, decomposition in graph_adj_matrices.items()
    }

    iteration = 0
    max_iterations = 1000
    while front.nnz and iteration < max_iterations:
        iteration += 1
        new_front_blocks = {}
        for symbol in common_symbols:
            this_symbol_blocks = []
            for i in range(len(graph_adj.start_states)):
                current_block = front[
                    i * graph_adj.states_num : (i + 1) * graph_adj.states_num
                ]
                step = (
                    graph_adj_matrices_t[symbol]
                    @ current_block
                    @ regex_adj_matrices[symbol]
                )
                this_symbol_blocks.append(step)
            new_front_blocks[symbol] = sparse.vstack(this_symbol_blocks)
        front = sum(new_front_blocks.values()) > visited
        visited = visited + front

    if iteration >= max_iterations:
        print(
            f"iteration count = {iteration}, which exceeds max_iterations = {max_iterations}"
        )
    result = set()
    for i, graph_start_state in enumerate(graph_adj.start_states):
        result_block = visited[
            i * graph_adj.states_num : (i + 1) * graph_adj.states_num
        ]
        for graph_final_state in graph_adj.final_states:
            graph_final_idx = graph_state_to_idx[graph_final_state]
            for regex_final_state in regex_adj.final_states:
                regex_final_idx = regex_state_to_idx[regex_final_state]
                if result_block[graph_final_idx, regex_final_idx]:
                    result.add((graph_start_state.value, graph_final_state.value))
    return result
