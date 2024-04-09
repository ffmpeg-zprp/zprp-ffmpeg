from abc import ABC
from abc import abstractmethod

from .BaseCommunicator import BaseCommunicator
from .FilterGraph import Stream


class BaseConnector(ABC):
    """
    Abstract class for talking with ffmpeg.

    A connector always needs a `run` method to execute something user wants.
    """

    @abstractmethod
    def run(graph: Stream) -> BaseCommunicator:
        """Executes given fliter graph
        :param graph: filter graph describing user operations.

        :return: a handle to read output from ffmpeg, for example stdout from process."""
