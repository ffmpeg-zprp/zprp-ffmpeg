from dataclasses import dataclass

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
