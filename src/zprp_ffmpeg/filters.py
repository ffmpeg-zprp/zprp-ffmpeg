from typing import Iterable

from .FilterGraph import Filter
from .FilterGraph import Stream


def hflip(graph: Stream) -> Stream:
    graph.append(Filter("hflip"))
    return graph


def concat(graphs: Iterable[Stream]) -> Stream:
    filter_ = Filter("concat")
    for graph in graphs:
        filter_.add_input(graph)
    return Stream().append(filter_)
