__version__ = "2.1.0"
# from . import longest
from inspect import getmembers
from inspect import isfunction
from typing import Any

from ._api_compat import input
from ._api_compat import filter
from ._api_compat import output
from ._api_compat import global_args
from ._api_compat import get_args
from ._api_compat import compile
from ._api_compat import run
from ._api_compat import run_async
from .FilterGraph import Stream
from .filters import *  # noqa: F403 this is impossible to avoid
from .generated_filters import *  # noqa: F403 this is impossible to avoid
from .probe import probe
from .view import view

generated_filters: Any
_api_compat: Any

# This is for `from xyz import *`, but also to make linter shut up

__all__ = ["input", "filter", "output", "global_args", "get_args", "compile", "run", "run_async", "probe", "view", "generated_filters", "filters"]  # noqa: F405

stream_modules = [generated_filters, _api_compat]  # noqa: F821

for module in stream_modules:
    for name, func in getmembers(module, isfunction):
        setattr(Stream, name, func)
