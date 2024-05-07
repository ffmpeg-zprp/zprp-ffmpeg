from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterOption:
    name: str
    type: str  # the C type from ffmpeg as string
    description: Optional[str] = ""


@dataclass
class Filter:
    name: str
    description: str
    options: list[FilterOption]
