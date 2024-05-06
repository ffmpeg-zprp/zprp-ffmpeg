from dataclasses import dataclass


@dataclass
class FilterOption:
    name: str
    type: str
    description: str


@dataclass
class Filter:
    name: str
    description: str
    options: list[FilterOption]
