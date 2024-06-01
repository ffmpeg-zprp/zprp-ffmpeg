from zprp_ffmpeg.FilterGraph import Filter
from zprp_ffmpeg.FilterGraph import FilterOption
from zprp_ffmpeg.FilterGraph import FilterType


def test_filter_build_command_no_opts():
    f = Filter("hflip", filter_type=FilterType.VIDEO.value)
    assert f.get_command() == ":v]hflip"


def test_filter_build_command_with_opts():
    f = Filter("scale", filter_type=FilterType.VIDEO.value, params=[FilterOption("w", "20"), FilterOption("h", "40")])
    assert f.get_command() == ":v]scale=w=20:h=40"  # order isn't important in ffmpeg i think
