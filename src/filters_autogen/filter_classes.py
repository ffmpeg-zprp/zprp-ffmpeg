from dataclasses import dataclass
from typing import Optional

from zprp_ffmpeg.FilterGraph import FilterType


@dataclass
class FilterOption:
    name: str
    type: str  # the C type from ffmpeg as string
    description: Optional[str] = ""


@dataclass
class Filter:
    name: str
    description: str
    type: FilterType
    options: list[FilterOption]
