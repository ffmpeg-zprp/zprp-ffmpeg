# test if the package structure matches
import zprp_ffmpeg as ffmpeg


def test_input():
    # example from readme
    stream = ffmpeg.input("input.mp4")
    stream = ffmpeg.hflip(stream)
    stream = ffmpeg.output(stream, "out.mp4")
    ffmpeg.run(stream)
