from pathlib import Path

import pytest

from filters_autogen import parse_ffmpeg

assets_dir = Path(__file__).parent / "assets"

@pytest.mark.requires_gcc
def test_parse_empty(tmp_path: Path):
    filter_source = tmp_path / "vf_something.c"
    filter_source.write_text("")
    assert parse_ffmpeg.parse_one_file(filter_source) == []

@pytest.mark.requires_gcc
def test_parse_simple():
    """video filter with no options"""
    target = assets_dir / "libavfilter" / "vf_simple.c"
    filters_list = parse_ffmpeg.parse_one_file(target)

    assert len(filters_list) == 1
    assert filters_list[0].name == "simple123"
    assert filters_list[0].description == "simple filter"
    assert filters_list[0].type == "AVMEDIA_TYPE_VIDEO"  # at this stage, type is string of identifier
    assert filters_list[0].options == []

@pytest.mark.requires_gcc
def test_parse_one_option():
    """video filter with one string option"""
    target = assets_dir / "libavfilter" / "vf_oneoption.c"
    filters_list = parse_ffmpeg.parse_one_file(target)

    assert len(filters_list) == 1
    assert filters_list[0].name == "oneoption"
    assert filters_list[0].description == "I have one option"
    assert filters_list[0].type == "AVMEDIA_TYPE_VIDEO"
    assert len(filters_list[0].options) == 1
    assert filters_list[0].options[0].name == "oname"
    assert filters_list[0].options[0].description == "odesc"
    assert filters_list[0].options[0].type == "AV_OPT_TYPE_STRING"  # this is a string as well
    assert filters_list[0].options[0].available_values == {}  # this option has no named consts

@pytest.mark.requires_gcc
def test_parse_named_constants():
    """video filter with option that has possible choices"""
    target = assets_dir / "libavfilter" / "vf_named_constants.c"
    filters_list = parse_ffmpeg.parse_one_file(target)

    assert len(filters_list) == 1
    assert filters_list[0].name == "myfilter"
    assert filters_list[0].type == "AVMEDIA_TYPE_VIDEO"
    assert len(filters_list[0].options) == 1
    assert filters_list[0].options[0].name == "color"
    assert filters_list[0].options[0].description == "select color"
    assert filters_list[0].options[0].type == "AV_OPT_TYPE_STRING"
    assert filters_list[0].options[0].available_values == {"red":1,"green":2,"blue":3}

@pytest.mark.requires_gcc
def test_parse_allfilters():
    """ this isn't super important but why not"""
    filters = parse_ffmpeg.parse_allfilters(assets_dir / "allfilters.c")

    assert len(filters)==3
    assert "ff_filter2" in filters
