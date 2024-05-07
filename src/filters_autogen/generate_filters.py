from pathlib import Path

from make_file import construct_file
from parse_ffmpeg import parse_source_code

filters = parse_source_code(save_pickle=True)
file_str = construct_file(filters)

outpath = Path(__file__).parent.parent / "zprp_ffmpeg" / "generated_filters.py"

imports = """from zprp_ffmpeg.FilterGraph import Filter
from zprp_ffmpeg.FilterGraph import FilterOption
from zprp_ffmpeg.FilterGraph import Stream\n\n"""


with outpath.open("w+") as f:
    f.write(imports)
    f.write(file_str)
