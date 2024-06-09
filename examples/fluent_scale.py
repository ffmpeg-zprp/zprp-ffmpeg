from pathlib import Path

import zprp_ffmpeg
import zprp_ffmpeg.filter_graph

work_dir = Path(__file__).parent


(zprp_ffmpeg.input(str(work_dir / "input.mp4")).scale(w="20", h="20").output(str(work_dir / "output.mp4")).run(extra_options=" -y"))
