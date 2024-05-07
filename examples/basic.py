from pathlib import Path

import zprp_ffmpeg
import zprp_ffmpeg.FilterGraph

work_dir = Path(__file__).parent


stream = zprp_ffmpeg.input(str(work_dir / "input.mp4"))
# illegal use of filters, but this copies the stream raw.
# stream.append(zprp_ffmpeg.FilterGraph.Filter("copy",filter_prefix="-c"))
stream = zprp_ffmpeg.hflip(stream)
stream = zprp_ffmpeg.output(stream, str(work_dir / "output.mp4"))
zprp_ffmpeg.run(stream, extra_options=" -y")
