from dataclasses import dataclass
from typing import Dict, Set, Tuple
from pyformlang import rsa
import networkx as nx


def gll_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    @dataclass(frozen=True)
    class StateMarker:
        variable: str
        substate: str

    @dataclass(frozen=True)
    class TreeNode:
        gss_node: "NodeStack"
        state: StateMarker
        node_id: int

    class NodeStack:
        def __init__(self, state: StateMarker, node_id: int):
            self.state = state
            self.node_id = node_id
            self.refs = {}
            self.visited_nodes = set()

        def pop(self, current_node_id: int) -> Set[TreeNode]:
            if current_node_id in self.visited_nodes:
                return set()
            self.visited_nodes.add(current_node_id)
            return {
                TreeNode(gss, state, current_node_id)
                for state, gss_list in self.refs.items()
                for gss in gss_list
            }

        def add_ref(
            self, return_state: StateMarker, gss_node: "NodeStack"
        ) -> Set[TreeNode]:
            if gss_node in self.refs.setdefault(return_state, set()):
                return set()
            self.refs[return_state].add(gss_node)
            return {
                TreeNode(gss_node, return_state, current_node)
                for current_node in self.visited_nodes
            }

    @dataclass
    class StateDescription:
        terminal_edges: Dict[str, StateMarker]
        variable_edges: Dict[str, Tuple[StateMarker, StateMarker]]
        is_final_state: bool

    def prepare_parser_data(rsm: rsa.RecursiveAutomaton, graph: nx.DiGraph):
        node_edges = {
            node: {
                symbol: set() for _, _, symbol in graph.edges(data="label") if symbol
            }
            for node in graph.nodes()
        }

        state_data = {
            variable: {
                state: StateDescription({}, {}, state in box.dfa.final_states)
                for state in box.dfa.to_networkx().nodes
            }
            for variable, box in rsm.boxes.items()
        }

        start_state = StateMarker(
            rsm.initial_label, rsm.boxes[rsm.initial_label].dfa.start_state.value
        )
        accept_gss_node = NodeStack(StateMarker("$", "fin"), None)
        gss_nodes = {}

        for from_node, to_node, symbol in graph.edges(data="label"):
            if symbol:
                node_edges[from_node].setdefault(symbol, set()).add(to_node)

        for variable, box in rsm.boxes.items():
            for from_state, to_state, symbol in box.dfa.to_networkx().edges(
                data="label"
            ):
                if symbol:
                    state_data_item = state_data[variable][from_state]
                    if symbol in rsm.boxes:
                        sub_box = rsm.boxes[symbol].dfa
                        state_data_item.variable_edges[symbol] = (
                            StateMarker(symbol, sub_box.start_state.value),
                            StateMarker(variable, to_state),
                        )
                    else:
                        state_data_item.terminal_edges[symbol] = StateMarker(
                            variable, to_state
                        )

        return node_edges, state_data, start_state, accept_gss_node, gss_nodes

    def process_popped_nodes(
        nodes: Set[TreeNode], previous_node: TreeNode
    ) -> Tuple[Set[TreeNode], Set[Tuple[int, int]]]:
        return {node for node in nodes if node.gss_node != accept_gss_node}, {
            (previous_node.gss_node.node_id, node.node_id)
            for node in nodes
            if node.gss_node == accept_gss_node
        }

    # Main function implementation
    node_edges, state_data, start_state, accept_gss_node, gss_nodes = (
        prepare_parser_data(rsm, graph)
    )

    def get_gss_node(state: StateMarker, node_id: int) -> NodeStack:
        if (state, node_id) not in gss_nodes:
            gss_nodes[(state, node_id)] = NodeStack(state, node_id)
        return gss_nodes[(state, node_id)]

    reachable_pairs = set()
    added_nodes = set()
    pending_nodes = set()

    def add_parse_nodes(nodes: Set[TreeNode]):
        new_nodes = nodes - added_nodes
        added_nodes.update(new_nodes)
        pending_nodes.update(new_nodes)

    for node_id in start_nodes or set(graph.nodes()):
        gss_node = get_gss_node(start_state, node_id)
        gss_node.add_ref(StateMarker("$", "fin"), accept_gss_node)
        add_parse_nodes({TreeNode(gss_node, start_state, node_id)})

    while pending_nodes:
        parse_node = pending_nodes.pop()
        state_data_item = state_data[parse_node.state.variable][
            parse_node.state.substate
        ]

        for terminal, new_state in state_data_item.terminal_edges.items():
            if terminal in node_edges[parse_node.node_id]:
                add_parse_nodes(
                    {
                        TreeNode(parse_node.gss_node, new_state, new_node_id)
                        for new_node_id in node_edges[parse_node.node_id][terminal]
                    }
                )

        for _, (
            start_rsm_state,
            return_rsm_state,
        ) in state_data_item.variable_edges.items():
            sub_node = get_gss_node(start_rsm_state, parse_node.node_id)
            popped_nodes = sub_node.add_ref(return_rsm_state, parse_node.gss_node)
            popped_nodes, sub_reachable_pairs = process_popped_nodes(
                popped_nodes, parse_node
            )
            add_parse_nodes(popped_nodes)
            add_parse_nodes({TreeNode(sub_node, start_rsm_state, parse_node.node_id)})
            reachable_pairs.update(sub_reachable_pairs)

        if state_data_item.is_final_state:
            new_parse_nodes = parse_node.gss_node.pop(parse_node.node_id)
            new_parse_nodes, start_end_pairs = process_popped_nodes(
                new_parse_nodes, parse_node
            )
            add_parse_nodes(new_parse_nodes)
            reachable_pairs.update(start_end_pairs)

    return {
        (start_node, end_node)
        for start_node, end_node in reachable_pairs
        if end_node in (final_nodes or set(graph.nodes()))
    }
