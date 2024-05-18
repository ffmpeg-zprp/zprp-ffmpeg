from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional


@dataclass
class FilterOption:
    name: str
    type: str  # the C type from ffmpeg as string
    unit: str  # "family" of parameters, used with named constants
    available_values: Dict[str, int]
    description: Optional[str] = ""


@dataclass
class Filter:
    name: str
    description: str
    type: str
    options: List[FilterOption]
