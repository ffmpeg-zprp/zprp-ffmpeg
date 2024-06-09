from typing import Iterable
from typing import Optional

from .filter_graph import Filter
from .filter_graph import FilterOption
from .filter_graph import Stream


def concat(graphs: Iterable[Stream], n: Optional[int] = None, v: Optional[int] = None, a: Optional[int] = None) -> Stream:
    """Concatenate audio and video streams, joining them together one after the other.

    :param int n: set the number of segments. Default is 2.
    :param int v: set the number of output video streams, that is also the number of video streams in each segment. Default is 1.
    :param int a: set the number of output audio streams, that is also the number of audio streams in each segment. Default is 0.
    """
    filter_ = Filter(
        "concat",
        params=[
            FilterOption(name="n", value=n),
            FilterOption(name="v", value=v),
            FilterOption(name="a", value=a),
        ],
    )
    for graph in graphs:
        filter_.add_input(graph)
    return Stream().append(filter_)


def overlay(
    main: Stream,
    sub: Stream,
    x: Optional[str] = None,
    y: Optional[str] = None,
    eof_action: Optional[str] = None,
    eval: Optional[str] = None,
    shortest: Optional[bool] = None,
    format: Optional[str] = None,
    repeatlast: Optional[bool] = None,
    alpha: Optional[str] = None,
) -> Stream:
    """Overlay a video source on top of the input.

    :param str x: set the x expression
    :param str y: set the y expression
    :param str eof_action: Action to take when encountering EOF from secondary input
            possible values: repeat, endall, pass
    :param str eval: specify when to evaluate expressions
            possible values: init, frame
    :param bool shortest: force termination when the shortest input terminates
    :param str format: set output format
            possible values: yuv420, yuv420p10, yuv422, yuv422p10, yuv444, yuv444p10, rgb, gbrp, auto
    :param bool repeatlast: repeat overlay of the last overlay frame
    :param str alpha: alpha format
            possible values: straight, premultiplied
    """
    filter_ = Filter(
        command="overlay",
        filter_type="AVMEDIA_TYPE_VIDEO",
        params=[
            FilterOption(name="x", value=x),
            FilterOption(name="y", value=y),
            FilterOption(name="eof_action", value=eof_action),
            FilterOption(name="eval", value=eval),
            FilterOption(name="shortest", value=shortest),
            FilterOption(name="format", value=format),
            FilterOption(name="repeatlast", value=repeatlast),
            FilterOption(name="alpha", value=alpha),
        ],
    )
    filter_.add_input(main)
    filter_.add_input(sub)
    return Stream().append(filter_)
