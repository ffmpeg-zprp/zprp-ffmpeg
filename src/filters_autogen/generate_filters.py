from pathlib import Path

from make_file import construct_file
from parse_ffmpeg import parse_source_code

filters = parse_source_code()
file_str = construct_file(filters)

outpath = Path(__file__).parent.parent / "zprp_ffmpeg" / "generated_filters.py"

with outpath.open("w+") as f:
    f.write("from filter_classes import Filter,FilterOption\n")
    f.write(file_str)
