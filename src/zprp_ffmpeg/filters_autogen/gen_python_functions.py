import keyword
import pickle
from pathlib import Path
from string import Template

from filter_classes import Filter
from filter_classes import ffmpeg_type_to_python

filter_template = Template(
    """
def $function_name($options):
    \"\"\"$description\"\"\"
    return Filter(command=$function_name,params=$params)
"""
)


def sanitize(name: str):
    """
    Escapes python keyword like "as", "with". Some filter options collide with them.
    """
    if name in keyword.kwlist or name[0].isdigit():
        name = "_" + name
    if "-" in name:
        name = name.replace("-", "")
    return name


def fill_template(filter_template, filter: Filter):
    """constructs python method based on filter dataclass."""
    options = []  # for function declaration
    filter_params = []  # for Filter initialisation
    for option in filter.options:
        sanitized = sanitize(option.name)
        option.type = ffmpeg_type_to_python[option.type].__qualname__
        options.append(f"{sanitized}: {option.type} = None")
        filter_params.append(f'FilterOption(name="{option.name}",type={option.type})')
        if option.description:
            filter.description += f"\n\t:param {option.type} {option.name}: {option.description}"
    options_str = ", ".join(options)
    filter_params_str = ", ".join(filter_params)

    return filter_template.safe_substitute(
        function_name=filter.name, options=options_str, description=filter.description, params=f"[{filter_params_str}]"
    )


with Path("FFmpeg/all_filters.pickle").open("rb") as f:
    all_filters = pickle.load(f)  # noqa: S301

out = ""

for filter in all_filters:
    out += fill_template(filter_template, filter)

with Path("out.py").open("w+") as f:
    f.write("from filter_classes import Filter,FilterOption\n")
    f.write(out)
