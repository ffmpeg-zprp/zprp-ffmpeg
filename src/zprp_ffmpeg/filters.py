from typing import Iterable

from .FilterGraph import Filter
from .FilterGraph import FilterOption
from .FilterGraph import Stream

# def hflip(graph: Stream) -> Stream:
#     graph.append(Filter("hflip"))
#     return graph


def concat(graphs: Iterable[Stream]) -> Stream:
    filter_ = Filter("concat")
    for graph in graphs:
        filter_.add_input(graph)
    return Stream().append(filter_)


def overlay(main, sub, x=0, y=0) -> Stream:
    filter_ = Filter("overlay", params=[FilterOption(name="x", value=x), FilterOption(name="y", value=y)])
    filter_.add_input(main)
    filter_.add_input(sub)
    return Stream().append(filter_)
