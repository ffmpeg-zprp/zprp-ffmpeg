import pytest

from zprp_ffmpeg.FilterGraph import Stream
from zprp_ffmpeg.ProcessConnector import ProcessConnector


@pytest.mark.starts_process
def test_ProcessConnector_empty_graph():
    fg = Stream()
    proc = ProcessConnector.run(fg)
    out, err = proc.communicate()
    print(out)
