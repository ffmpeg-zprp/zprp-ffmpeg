from abc import ABC
from abc import abstractmethod
from typing import Tuple

from .filter_graph import Stream


class BaseConnector(ABC):
    """
    Abstract class for talking with ffmpeg.

    A connector always needs a `run` method to execute something user wants.
    """

    # this is basically a named constructor
    @classmethod
    @abstractmethod
    def run(cls, graph: Stream) -> "BaseConnector":
        """Executes given fliter graph
        :param graph: filter graph describing user operations.

        :return: a handle to read output from ffmpeg, for example stdout from process."""

    @abstractmethod
    def communicate(self) -> Tuple[bytes, bytes]:
        """Reads stdout and stderr from process"""
