import pytest

from zprp_ffmpeg.filter_graph import Filter
from zprp_ffmpeg.filter_graph import FilterType
from zprp_ffmpeg.filter_graph import Stream
from zprp_ffmpeg.process_connector import ProcessConnector


@pytest.mark.starts_process
def test_ProcessConnector_empty_graph():
    fg = Stream()
    with pytest.raises(ValueError, match="No filters selected"):
        _ = ProcessConnector.run(fg)


@pytest.mark.starts_process
def test_ProcessConnector_any_graph():
    fg = Stream().append(Filter("hflip", filter_type=FilterType.VIDEO.value))
    proc = ProcessConnector.run(fg)
    out, err = proc.communicate()
    print(out)


@pytest.mark.starts_process
def test_communicate():
    fg = Stream().append(Filter("hflip", filter_type=FilterType.VIDEO.value))
    stdout,stderr = ProcessConnector.run(fg,extra_options="-h").communicate()

    assert stdout is not None # exact output is not important, but communication should be possible
    assert stderr is not None
