from pathlib import Path

import zprp_ffmpeg
import zprp_ffmpeg.filter_graph

work_dir = Path(__file__).parent


stream = zprp_ffmpeg.input(str(work_dir / "in.mp4"))
image = zprp_ffmpeg.input(str(work_dir / "ffmpeg_logo.jpg"))
stream = zprp_ffmpeg.overlay(stream, image, x="20", y="20")
stream = zprp_ffmpeg.output(stream, str(work_dir / "output.mp4"))
zprp_ffmpeg.run(stream, extra_options=" -y")
