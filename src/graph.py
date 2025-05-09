import os
from typing import Set, Tuple
from graphviz import Digraph
from value import Operation, Value


def trace_graph(root: Value):
    nodes: Set[Value] = set()
    edges: Set[Tuple[Value, Value]] = set()

    def build(current: Value):
        nodes.add(current)
        for child in current._children:
            edges.add((current, child))
            build(child)

    build(root)
    print(nodes)
    print(edges)

    return nodes, edges


def plot_graph(root: Value):
    path = os.path.join(".", "graph", "graph")
    nodes, edges = trace_graph(root)
    graph = Digraph(format="png", graph_attr={"rankdir": "LR"})

    for node in nodes:
        uid = str(id(node))
        graph.node(
            name=uid,
            label=f"{{{node.label} | value: {node.data:.3f} | grad: {node.grad:.3f} }}",
            shape="record",
        )
        if node._op and node._op is not Operation.NONE:
            graph.node(name=uid + node._op.value, label=node._op.value)
            graph.edge(uid + node._op.value, uid)

    for parent, child in edges:
        puid = str(id(parent))
        cuid = str(id(child))
        graph.edge(cuid, puid + parent._op.value)

    graph.render(path, view=True)
