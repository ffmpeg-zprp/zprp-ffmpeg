import pytest

from zprp_ffmpeg.FilterGraph import Stream, Filter, FilterType
from zprp_ffmpeg.ProcessConnector import ProcessConnector


@pytest.mark.starts_process
def test_ProcessConnector_empty_graph():
    fg = Stream()
    with pytest.raises(ValueError):
        proc = ProcessConnector.run(fg)


@pytest.mark.starts_process
def test_ProcessConnector_any_graph():
    fg = Stream().append(Filter("hflip", filter_type=FilterType.VIDEO.value))
    proc = ProcessConnector.run(fg)
    out, err = proc.communicate()
    print(out)
