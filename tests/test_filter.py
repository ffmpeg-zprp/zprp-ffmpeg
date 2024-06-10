import zprp_ffmpeg as ffmpeg
from zprp_ffmpeg.filter_graph import Filter
from zprp_ffmpeg.filter_graph import FilterOption
from zprp_ffmpeg.filter_graph import FilterType


def test_filter_build_command_no_opts():
    f = Filter("hflip", filter_type=FilterType.VIDEO.value)
    assert f.get_command() == ":v]hflip"


def test_filter_build_command_with_opts():
    f = Filter("scale", filter_type=FilterType.VIDEO.value, params=[FilterOption("w", "20"), FilterOption("h", "40")])
    assert f.get_command() == ":v]scale=w=20:h=40"  # order isn't important in ffmpeg i think

def test_concat():
    g1 = ffmpeg.input("in1.mp4")
    g2 = ffmpeg.input("in2.mp4")
    result = ffmpeg.concat((g1,g2))
    assert len(result._nodes)==1
    assert result._nodes[0].command=="concat"
    assert len(result._nodes[0]._in)==2

    assert g1 in result._nodes[0]._in
    assert g2 in result._nodes[0]._in

def test_overlay():
    main_file = ffmpeg.input("in1.mp4")
    sub_file = ffmpeg.input("in2.mp4")
    result = ffmpeg.overlay(main=main_file,sub=sub_file)
    assert len(result._nodes)==1
    assert result._nodes[0].command=="overlay"
    assert len(result._nodes[0]._in)==2

    # order is super important
    assert result._nodes[0]._in[0]==main_file
    assert result._nodes[0]._in[1]==sub_file
