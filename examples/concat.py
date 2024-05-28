from pathlib import Path

import zprp_ffmpeg
import zprp_ffmpeg.FilterGraph

work_dir = Path(__file__).parent


stream = zprp_ffmpeg.input(str(work_dir / "input.mp4"))
stream = zprp_ffmpeg.fade(stream, type="in", start_frame=0, nb_frames=30)
stream = zprp_ffmpeg.concat((stream, stream))
stream = zprp_ffmpeg.output(stream, str(work_dir / "output.mp4"))
zprp_ffmpeg.run(stream, extra_options=" -y")
