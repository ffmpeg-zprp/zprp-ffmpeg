from pathlib import Path

import zprp_ffmpeg
import zprp_ffmpeg.FilterGraph

work_dir = Path(__file__).parent


stream = zprp_ffmpeg.input(str(work_dir / "input.mp4"))
stream = zprp_ffmpeg.scale(stream, w="20", h="20")
stream = zprp_ffmpeg.output(stream, str(work_dir / "output.mp4"))

zprp_ffmpeg.view(stream, filename="a.png")
