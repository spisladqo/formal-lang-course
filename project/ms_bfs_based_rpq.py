from networkx import MultiDiGraph
from project.finite_automata import regex_to_dfa, graph_to_nfa
from project.regular_query import AdjacencyMatrixFA
from collections import deque


def create_trans_dict(automaton: AdjacencyMatrixFA) -> dict[str, dict[int, set[int]]]:
    transitions = {}
    for symbol, matrix in automaton.adj_matrices.items():
        transitions[symbol] = {}
        coo = matrix.tocoo()
        for i, j in zip(coo.row, coo.col):
            if i not in transitions[symbol]:
                transitions[symbol][i] = set()
            transitions[symbol][i].add(j)
    return transitions


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)

    nfaG = AdjacencyMatrixFA(graph_nfa)
    dfaQ = AdjacencyMatrixFA(regex_dfa)

    nfa_transitions = create_trans_dict(nfaG)
    dfa_transitions = create_trans_dict(dfaQ)

    abc = set(nfa_transitions.keys()) & set(dfa_transitions.keys())

    nfa_index_to_state = {v: k for k, v in nfaG.state_to_index.items()}
    dfa_index_to_state = {v: k for k, v in dfaQ.state_to_index.items()}

    pairs = set()

    for start_node in start_nodes:
        if start_node not in nfaG.state_to_index:
            continue

        start_nfa_idx = nfaG.state_to_index[start_node]
        visited = set()
        queue = deque()

        for dfa_start in dfaQ.start_states:
            start_dfa_idx = dfaQ.state_to_index[dfa_start]
            start_state = (start_nfa_idx, start_dfa_idx)
            visited.add(start_state)
            queue.append(start_state)

        while queue:
            nfa_idx, dfa_idx = queue.popleft()

            nfa_state = nfa_index_to_state[nfa_idx]
            dfa_state = dfa_index_to_state[dfa_idx]

            if nfa_state in final_nodes and dfa_state in dfaQ.final_states:
                pairs.add((start_node, nfa_state))

            for symbol in abc:
                nfa_nexts = nfa_transitions[symbol].get(nfa_idx, set())
                dfa_nexts = dfa_transitions[symbol].get(dfa_idx, set())

                for nfa_next_idx in nfa_nexts:
                    for dfa_next_idx in dfa_nexts:
                        next_state = (nfa_next_idx, dfa_next_idx)
                        if next_state not in visited:
                            visited.add(next_state)
                            queue.append(next_state)

    return pairs
