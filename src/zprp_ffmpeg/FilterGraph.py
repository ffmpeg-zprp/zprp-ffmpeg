"""There is no general 'Node' class, because it doesn't work well with object relations, source and sink filters are kind of orthogonal.
It slightly violates DRY, but the parameter types are different. It is what it is"""

from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import List
from typing import Optional
from typing import Union


class FilterType(Enum):
    VIDEO = "AVMEDIA_TYPE_VIDEO"
    AUDIO = "AVMEDIA_TYPE_AUDIO"


@dataclass
class FilterOption:
    name: str
    value: Any


class Filter:
    """Filters can have many inputs and many outputs, holds the filter name and potential params"""

    def __init__(self, command: str, params: Optional[List[FilterOption]] = None, filter_type: str = FilterType.VIDEO.value):
        self._out: List[AnyNode] = []
        self._in: List[AnyNode] = []
        self.command = command
        self.params = params if params else []
        self.filter_type = filter_type

    def add_output(self, parent: "Filter | SinkFilter"):
        self._out.append(parent)

    def add_input(self, child: "Filter | SourceFilter"):
        self._in.append(child)

    def get_command(self):
        joined_params = ":".join(p.name + "=" + str(p.value) for p in self.params if p.value)
        if joined_params:  # if there was no option, leave empty string
            joined_params = "=" + joined_params
        if self.filter_type == "AVMEDIA_TYPE_VIDEO":
            return "-vf " + self.command + joined_params

        elif self.filter_type == "AVMEDIA_TYPE_AUDIO":
            return "-af " + self.command + joined_params

        return ""  # in case no match


# names as per ffmpeg documentation
class SourceFilter:
    """There can't be any input node to input filter, it provides the input itself."""

    def __init__(self, in_path: str):
        self.in_path: str = in_path
        self._out: List[AnyNode] = []

    def add_output(self, parent: "Filter | SinkFilter"):
        self._out.append(parent)

    def add_input(self, child: "Filter"):
        raise NotImplementedError("This node can't have inputs")

    def get_command(self):
        return "-i " + self.in_path


class SinkFilter:
    """Similarly to SourceFilter, it doesn't make sense to output anything further, this is the end of graph."""

    def __init__(self, out_path: str):
        self.out_path: str = out_path
        self._in: List[AnyNode] = []

    def add_input(self, parent: "Filter | SourceFilter"):
        self._in.append(parent)

    def add_output(self, parent: "Filter"):
        raise NotImplementedError("This node can't have outputs")

    def get_command(self):
        return self.out_path


# in python 3.12 there is 'type' keyword, but we are targetting 3.8
# https://stackoverflow.com/questions/76712720/typeerror-unsupported-operand-types-for-type-and-nonetype
# python >3.9 uses , instead of | idk if it works with python 3.12
AnyNode = Union[Filter, SourceFilter, SinkFilter]


class Stream:
    """One directional sequence of nodes, in future will be able to visualize them.
    Streams can be concatenated and split with certain filters."""

    def __init__(self) -> None:
        self._nodes: List[AnyNode] = []

    def append(self, node: AnyNode) -> "Stream":
        if len(self._nodes) > 0:
            # connect head with new node
            if not isinstance(self._nodes[-1], SinkFilter) and not isinstance(node, SourceFilter):
                self._nodes[-1].add_output(node)
                node.add_input(self._nodes[-1])
        self._nodes.append(node)
        return self  # fluent
