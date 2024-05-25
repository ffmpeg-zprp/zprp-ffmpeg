from pathlib import Path

from filters_autogen.make_file import construct_file
from filters_autogen.parse_ffmpeg import parse_source_code

default_imports = """from typing import Optional

from zprp_ffmpeg.FilterGraph import Filter
from zprp_ffmpeg.FilterGraph import FilterOption
from zprp_ffmpeg.FilterGraph import Stream\n\n"""


def generate_filters(ffmpeg_dir: Path, outpath: Path = Path(__file__).parent / "zprp_ffmpeg" / "generated_filters.py"):
    filters = parse_source_code(
        ffmpeg_dir,
        save_pickle=True,
        debug=True,
    )
    file_str = construct_file(filters)

    with outpath.open("w+") as f:
        f.write(default_imports)
        f.write(file_str)


if __name__ == "__main__":
    ffmpeg_clone_dir = Path(__file__).parent / "filters_autogen" / "FFmpeg"
    generate_filters(ffmpeg_clone_dir)
