from abc import ABC
from abc import abstractmethod


class BaseCommunicator(ABC):
    """
    Object returned by :meth:`zprp-BaseConnector.BaseConnector.run`.

    When FFmpeg is working, a bunch of information statistical information is being returned (at least when ffmpeg is a process).
    This object should expose this interface
    """

    @abstractmethod
    def communicate():
        pass
