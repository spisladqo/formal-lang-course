import networkx as nx
import pyformlang.cfg as cfg
import scipy.sparse as sp
from typing import Set
from project.hellings import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    transformed_cfg = cfg_to_weak_normal_form(cfg)
    production_map = {}
    for prod in transformed_cfg.productions:
        if len(prod.body) == 2:
            production_map.setdefault(tuple(prod.body), set()).add(prod.head)

    node_count = graph.number_of_nodes()
    node_index = {node: idx for idx, node in enumerate(graph.nodes)}
    index_to_node = {idx: node for idx, node in enumerate(graph.nodes)}
    matrices = {
        var: sp.csc_matrix((node_count, node_count), dtype=bool)
        for var in transformed_cfg.variables
    }

    for start, end, label in graph.edges(data="label"):
        for variable in transformed_cfg.variables:
            if any(
                prod.body[0].value == label
                for prod in transformed_cfg.productions
                if prod.head == variable and len(prod.body) == 1
            ):
                matrices[variable][node_index[start], node_index[end]] = True

    for node in graph.nodes:
        for variable in transformed_cfg.get_nullable_symbols():
            matrices[variable][node_index[node], node_index[node]] = True

    update_queue = list(matrices.keys())

    while update_queue:
        current_var = update_queue.pop(0)
        for body, heads in production_map.items():
            if current_var not in body:
                continue
            new_matrix = matrices[body[0]] @ matrices[body[1]]
            for head in heads:
                old_nnz = matrices[head].nnz
                matrices[head] = matrices[head] + new_matrix
                matrices[head].eliminate_zeros()
                if matrices[head].nnz > old_nnz:
                    if head not in update_queue:
                        update_queue.append(head)

    if transformed_cfg.start_symbol not in matrices:
        return set()

    result_matrix = matrices[transformed_cfg.start_symbol]

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    res = {
        (index_to_node[i], index_to_node[j])
        for i, j in zip(*result_matrix.nonzero())
        if index_to_node[i] in start_nodes and index_to_node[j] in final_nodes
    }

    return res
