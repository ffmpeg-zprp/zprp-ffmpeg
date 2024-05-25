__version__ = "1.1.0"
# from . import longest
from inspect import getmembers
from inspect import isfunction

from zprp_ffmpeg import _api_compat
from zprp_ffmpeg import generated_filters

from ._api_compat import input
from ._api_compat import output
from ._api_compat import run
from ._api_compat import run_async
from .FilterGraph import Stream
from .filters import hflip
from .generated_filters import *  # noqa: F403 this is impossible to avoid
from .probe import probe

# This is for `from xyz import *`, but also to make linter shut up
__all__ = ["input", "output", "run", "run_async", "hflip", "probe"]

stream_modules = [generated_filters, _api_compat]

for module in stream_modules:
    for name, func in getmembers(module, isfunction):
        setattr(Stream, name, func)
