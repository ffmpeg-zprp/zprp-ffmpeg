from .filter_graph import Filter
from .filter_graph import SinkFilter
from .filter_graph import SourceFilter
from .filter_graph import Stream


def view(graph: Stream, filename=None) -> None:
    "Creates graph of filters"

    import networkx as nx  # type: ignore
    from matplotlib import pyplot as plt  # type: ignore

    colors = {"input": "#99cc00", "output": "#99ccff", "filter": "#ffcc00"}

    G = nx.DiGraph()

    graph_connection = []

    for node in graph._nodes:
        if isinstance(node, SourceFilter):
            graph_connection.append((node.in_path.split("/")[-1], colors["input"]))
        elif isinstance(node, SinkFilter):
            graph_connection.append((node.out_path.split("/")[-1], colors["output"]))
        elif isinstance(node, Filter):
            graph_connection.append((node.command, colors["filter"]))

    # Adding nodes
    for nodeG, color in graph_connection:
        G.add_node(nodeG, color=color)

    # Adding edges
    for i in range(len(graph_connection) - 1):
        G.add_edge(graph_connection[i][0], graph_connection[i + 1][0])

    # Set nodes to be horizontal
    pos = {}
    for i, nodeG in enumerate(graph_connection):  # type: ignore
        pos[nodeG[0]] = (i, 0)

    nx.draw(
        G, pos, with_labels=True, node_shape="s", node_size=3000, node_color=[color for _, color in graph_connection], font_weight="bold"
    )

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
