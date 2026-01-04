from pyformlang.cfg import CFG, Terminal, Production, Epsilon
import networkx as nx


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    prods = set()

    for prod in cfg.to_normal_form().productions:
        prods.add(prod)

    for sym in cfg.get_nullable_symbols():
        prods.add(Production(sym, [Epsilon()], filtering=False))

    return CFG(start_symbol=cfg.start_symbol, productions=prods)


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    wnf = cfg_to_weak_normal_form(cfg)

    edges = set()
    for src, tgt, lbl in graph.edges.data("label"):
        edges.add((src, Terminal(lbl), tgt))

    term_to_vars, body_to_head = {}, {}
    for production in wnf.productions:
        if len(production.body) == 1 and isinstance(production.body[0], Terminal):
            term = production.body[0]
            if term not in term_to_vars:
                term_to_vars[term] = set()
            term_to_vars[term].add(production.head)
        elif len(production.body) == 2:
            body_tuple = tuple(production.body)
            if body_tuple not in body_to_head:
                body_to_head[body_tuple] = set()
            body_to_head[body_tuple].add(production.head)

    evaluated_edges = set()
    for src, term, tgt in edges:
        if term in term_to_vars:
            for var in term_to_vars[term]:
                evaluated_edges.add((src, var, tgt))

    for node in graph.nodes:
        for nullable in wnf.get_nullable_symbols():
            evaluated_edges.add((node, nullable, node))

    edge_queue = list(evaluated_edges)

    while edge_queue:
        current_edge = edge_queue.pop(0)
        new_edges = set()

        for evaluated in evaluated_edges:
            for edge_pair in [(current_edge, evaluated), (evaluated, current_edge)]:
                s1, x1, t1 = edge_pair[0]
                s2, x2, t2 = edge_pair[1]
                combined_key = (x1, x2)

                if s2 != t1 or combined_key not in body_to_head:
                    continue

                for new_var in body_to_head[combined_key]:
                    reduced_edge = (s1, new_var, t2)
                    if reduced_edge not in evaluated_edges:
                        new_edges.add(reduced_edge)
                        edge_queue.append(reduced_edge)

        evaluated_edges.update(new_edges)

    result_set = set()
    for src, var, tgt in evaluated_edges:
        if (
            src in start_nodes
            and var.value == wnf.start_symbol.value
            and tgt in final_nodes
        ):
            result_set.add((src, tgt))

    return result_set