__version__ = "0.0.0"
from zprp_ffmpeg import ProcessConnector

from ._zprp_ffmpeg import longest

# This is for `from xyz import *`, but also to make linter shut up
__all__ = ["longest", "ProcessConnector"]
