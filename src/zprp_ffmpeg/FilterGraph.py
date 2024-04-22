"""There is no general 'Node' class, because it doesn't work well with object relations, source and sink filters are kind of orthogonal.
It slightly violates DRY, but the parameter types are different. It is what it is"""

from typing import Optional
from typing import Union


class Filter:
    """Filters can have many inputs and many outputs, holds the filter name and potential params"""

    def __init__(self, command: str, params: Optional[str] = None):
        self._out: list[AnyNode] = []
        self._in: list[AnyNode] = []
        self.command = command
        self.params = params

    def add_output(self, parent: "Filter | SinkFilter"):
        self._out.append(parent)

    def add_input(self, child: "Filter | SourceFilter"):
        self._in.append(child)


# names as per ffmpeg documentation
class SourceFilter:
    """There can't be any input node to input filter, it provides the input itself."""

    def __init__(self, in_path: str):
        self.in_path: str = in_path
        self._out: list[AnyNode] = []

    def add_output(self, parent: "Filter | SinkFilter"):
        self._out.append(parent)

    def add_input(self, child: "Filter"):
        raise NotImplementedError("This node can't have inputs")


class SinkFilter:
    """Similarly to SourceFilter, it doesn't make sense to output anything further, this is the end of graph."""

    def __init__(self, out_path: str):
        self.out_path: str = out_path
        self._in: list[AnyNode] = []

    def add_input(self, parent: "Filter | SourceFilter"):
        self._in.append(parent)

    def add_output(self, parent: "Filter"):
        raise NotImplementedError("This node can't have outputs")


# in python 3.12 there is 'type' keyword, but we are targetting 3.8
# https://stackoverflow.com/questions/76712720/typeerror-unsupported-operand-types-for-type-and-nonetype
# python >3.9 uses , instead of | idk if it works with python 3.12
AnyNode = Union[Filter, SourceFilter, SinkFilter]


class Stream:
    """One directional sequence of nodes, in future will be able to visualize them.
    Streams can be concatenated and split with certain filters."""

    def __init__(self) -> None:
        self._nodes: list[AnyNode] = []

    def append(self, node: AnyNode) -> "Stream":
        if len(self._nodes) > 0:
            # connect head with new node
            if not isinstance(self._nodes[-1], SinkFilter) and not isinstance(node, SourceFilter):
                self._nodes[-1].add_output(node)
                node.add_input(self._nodes[-1])
        self._nodes.append(node)
        return self  # fluent
