from pathlib import Path

import zprp_ffmpeg
import zprp_ffmpeg.filter_graph

work_dir = Path(__file__).parent


stream = zprp_ffmpeg.input(str(work_dir / "in.mp4"))
stream = zprp_ffmpeg.scale(stream, w="20", h="20")
stream = zprp_ffmpeg.output(stream, str(work_dir / "output.mp4"))
stdout, stderr = zprp_ffmpeg.run(stream, extra_options=" -y")
print(stderr.decode("utf-8"))


# (zprp_ffmpeg.input(str(work_dir / "input.mp4")).scale(w="20", h="20").output(str(work_dir / "output.mp4")).run(extra_options=" -y"))
