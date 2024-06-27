# test if the package structure matches
import pytest

import zprp_ffmpeg as ffmpeg


def test_input():
    # example from readme
    stream1 = ffmpeg.input("input.mp4")
    stream2 = ffmpeg.hflip(stream1)
    stream3 = ffmpeg.output(ffmpeg.concat((stream1, stream2)), stream2, "out.mp4")
    ffmpeg.run(stream3)

    assert stream1 != stream2 != stream3


def test_filter_custom():
    """https://github.com/kkroening/ffmpeg-python/blob/df129c7ba30aaa9ffffb81a48f53aa7253b0b4e6/ffmpeg/tests/test_ffmpeg.py#L569"""
    stream = ffmpeg.input("dummy.mp4")
    stream = ffmpeg.filter(stream, "custom_filter", "a", "b", kwarg1="c")
    stream = ffmpeg.output(stream, "dummy2.mp4")
    assert stream.get_args() == [
        "-i",
        "dummy.mp4",
        "-filter_complex",
        '"[0:v]custom_filter=kwarg1=c[v0]"',
        "-map",
        "[v0]",
        "dummy2.mp4",
    ]


def test_filter_custom_fluent():
    """https://github.com/kkroening/ffmpeg-python/blob/df129c7ba30aaa9ffffb81a48f53aa7253b0b4e6/ffmpeg/tests/test_ffmpeg.py#L584"""
    stream = (
        ffmpeg.input("dummy.mp4")
        .filter("custom_filter", "a", "b", kwarg1="c")
        .output("dummy2.mp4")
    )
    assert stream.get_args() == [
        "-i",
        "dummy.mp4",
        "-filter_complex",
        '"[0:v]custom_filter=kwarg1=c[v0]"',
        "-map",
        "[v0]",
        "dummy2.mp4",
    ]


def test_compile_no_filters():
    out_file = ffmpeg.input("dummy.mp4").output("dummy2.mp4")
    with pytest.raises(ValueError, match="No filters selected"):
        out_file.compile()


def test_compile():
    """https://github.com/kkroening/ffmpeg-python/blob/df129c7ba30aaa9ffffb81a48f53aa7253b0b4e6/ffmpeg/tests/test_ffmpeg.py#L456"""
    stream = ffmpeg.input("dummy.mp4").hflip().output("dummy2.mp4")
    assert stream.compile() == [
        "ffmpeg",
        "-i",
        "dummy.mp4",
        "-filter_complex",
        '"[0:v]hflip[v0]"',
        "-map",
        "[v0]",
        "dummy2.mp4",
    ]
    assert stream.compile(cmd="ffmpeg.old") == [
        "ffmpeg.old",
        "-i",
        "dummy.mp4",
        "-filter_complex",
        '"[0:v]hflip[v0]"',
        "-map",
        "[v0]",
        "dummy2.mp4",
    ]


def test_get_args_simple():
    """https://github.com/kkroening/ffmpeg-python/blob/df129c7ba30aaa9ffffb81a48f53aa7253b0b4e6/ffmpeg/tests/test_ffmpeg.py#L135"""
    out_file = ffmpeg.input("dummy.mp4").hflip().output("dummy2.mp4")
    assert out_file.get_args() == ["-i", "dummy.mp4", "-filter_complex", '"[0:v]hflip[v0]"', "-map", "[v0]", "dummy2"
                                                                                                             ".mp4"]


def test_get_args_overwrite():
    out_file = ffmpeg.input("dummy.mp4").hflip().output("dummy2.mp4")
    assert out_file.get_args(True) == ["-i", "dummy.mp4", "-filter_complex", '"[0:v]hflip[v0]"', "-map", "[v0]",
                                       "dummy2.mp4", "-y"]


def test_global_args():
    out_file = (
        ffmpeg.input("dummy.mp4")
        .hflip()
        .output("dummy2.mp4")
        .global_args("-progress", "someurl")
    )
    assert out_file.get_args() == [
        "-i",
        "dummy.mp4",
        "-filter_complex",
        '"[0:v]hflip[v0]"',
        "-map",
        "[v0]",
        "dummy2.mp4",
        "-progress",
        "someurl",
    ]
