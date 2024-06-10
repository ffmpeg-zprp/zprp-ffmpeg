import zprp_ffmpeg

stream = zprp_ffmpeg.input("input.mp4")
stream = zprp_ffmpeg.hflip(stream)
stream = zprp_ffmpeg.output(stream, "output.mp4")
zprp_ffmpeg.run(stream)
