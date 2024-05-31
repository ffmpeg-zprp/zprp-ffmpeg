__version__ = "1.2.0"
# from . import longest
from inspect import getmembers
from inspect import isfunction

from zprp_ffmpeg import _api_compat
from zprp_ffmpeg import generated_filters

from ._api_compat import input  # noqa: F401
from ._api_compat import output  # noqa: F401
from ._api_compat import run  # noqa: F401
from ._api_compat import run_async  # noqa: F401
from .FilterGraph import Stream
from .filters import hflip  # noqa: F401
from .generated_filters import *  # noqa: F403 this is impossible to avoid
from .probe import probe  # noqa: F401
from .view import view  # noqa: F401

# This is for `from xyz import *`, but also to make linter shut up
# __all__ = ["input", "output", "run", "run_async", "hflip", "probe", "view"]

stream_modules = [generated_filters, _api_compat]

for module in stream_modules:
    for name, func in getmembers(module, isfunction):
        setattr(Stream, name, func)
