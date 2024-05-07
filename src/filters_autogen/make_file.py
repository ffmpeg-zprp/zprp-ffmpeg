import keyword
import pickle
from pathlib import Path
from string import Template

from zprp_ffmpeg.filter_classes import Filter

# @TODO: add custom types for where it makes sense
ffmpeg_type_to_python = {
    "AV_OPT_TYPE_FLAGS": str,  # this technically is a number, but TYPE_CONST allows for named constants with strings...
    "AV_OPT_TYPE_INT": int,
    "AV_OPT_TYPE_INT64": int,
    "AV_OPT_TYPE_DOUBLE": float,
    "AV_OPT_TYPE_FLOAT": float,
    "AV_OPT_TYPE_STRING": str,
    "AV_OPT_TYPE_RATIONAL": int,
    "AV_OPT_TYPE_BINARY": str,  # raw string, maybe should be bytes. Used only in libplacebo filter
    "AV_OPT_TYPE_DICT": str,
    "AV_OPT_TYPE_UINT64": int,
    "AV_OPT_TYPE_CONST": int,  # used as a named constant for "flags".
    "AV_OPT_TYPE_IMAGE_SIZE": int,
    "AV_OPT_TYPE_PIXEL_FMT": str,
    "AV_OPT_TYPE_SAMPLE_FMT": str,
    "AV_OPT_TYPE_VIDEO_RATE": str,
    "AV_OPT_TYPE_DURATION": int,
    "AV_OPT_TYPE_COLOR": str,
    "AV_OPT_TYPE_BOOL": bool,
    "AV_OPT_TYPE_CHLAYOUT": str,
    "AV_OPT_TYPE_FLAG_ARRAY": str,
}

filter_template = Template(
    """
def $function_name(graph: Stream, $options):
    \"\"\"$description\"\"\"
    filter=Filter(command="$function_name",params=$params)
    graph.append(filter)
    return graph
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
        options.append(f"{sanitized}: {option.type}")
        filter_params.append(f'FilterOption(name="{option.name}",type={option.type})')
        if option.description:
            filter.description += f"\n\t:param {option.type} {option.name}: {option.description}"
    options_str = ", ".join(options)
    filter_params_str = ", ".join(filter_params)

    return filter_template.safe_substitute(
        function_name=filter.name, options=options_str, description=filter.description, params=f"[{filter_params_str}]"
    )


def construct_file(filters: list[Filter]) -> str:
    out = ""

    for filter in filters:
        out += fill_template(filter_template, filter)
    return out


if __name__ == "__main__":
    with Path("FFmpeg/all_filters.pickle").open("rb") as f:
        filters = pickle.load(f)  # noqa: S301

    file_str = construct_file(filters)

    with Path("out.py").open("w+") as f:
        f.write("from filter_classes import Filter,FilterOption\n")
        f.write(file_str)
