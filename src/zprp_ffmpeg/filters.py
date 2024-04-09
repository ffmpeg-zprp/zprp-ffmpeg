from .FilterGraph import Filter
from .FilterGraph import Stream


def hflip(graph: Stream) -> Stream:
    graph.append(Filter("hflip"))
    return graph
