from typing import Optional

from zprp_ffmpeg.FilterGraph import Filter
from zprp_ffmpeg.FilterGraph import FilterOption
from zprp_ffmpeg.FilterGraph import Stream


def scale(
    graph: Stream,
    w: Optional[str] = None,
    h: Optional[str] = None,
    flags: Optional[str] = None,
    interl: Optional[bool] = None,
    size: Optional[str] = None,
    in_color_matrix: Optional[int] = None,
    out_color_matrix: Optional[int] = None,
    in_range: Optional[int] = None,
    out_range: Optional[int] = None,
    in_v_chr_pos: Optional[int] = None,
    in_h_chr_pos: Optional[int] = None,
    out_v_chr_pos: Optional[int] = None,
    out_h_chr_pos: Optional[int] = None,
    force_original_aspect_ratio: Optional[int] = None,
    force_divisible_by: Optional[int] = None,
    param0: Optional[float] = None,
    param1: Optional[float] = None,
    eval: Optional[int] = None,
):
    """Scale the input video size and/or convert the image format.
    :param str w: Output video width
    :param str h: Output video height
    :param str flags: Flags to pass to libswscale
    :param bool interl: set interlacing
    :param str size: set video size
    :param int in_color_matrix: set input YCbCr type
    :param int out_color_matrix: set output YCbCr type
    :param int in_range: set input color range
    :param int out_range: set output color range
    :param int in_v_chr_pos: input vertical chroma position in luma grid/256
    :param int in_h_chr_pos: input horizontal chroma position in luma grid/256
    :param int out_v_chr_pos: output vertical chroma position in luma grid/256
    :param int out_h_chr_pos: output horizontal chroma position in luma grid/256
    :param int force_original_aspect_ratio: decrease or increase w/h if necessary to keep the original AR
    :param int force_divisible_by: enforce that the output resolution is divisible by a defined integer when force_original_aspect_ratio is used
    :param float param0: Scaler param 0
    :param float param1: Scaler param 1
    :param int eval: specify when to evaluate expressions"""
    graph.append(
        Filter(
            command="scale",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="flags", value=flags),
                FilterOption(name="interl", value=interl),
                FilterOption(name="size", value=size),
                FilterOption(name="in_color_matrix", value=in_color_matrix),
                FilterOption(name="out_color_matrix", value=out_color_matrix),
                FilterOption(name="in_range", value=in_range),
                FilterOption(name="out_range", value=out_range),
                FilterOption(name="in_v_chr_pos", value=in_v_chr_pos),
                FilterOption(name="in_h_chr_pos", value=in_h_chr_pos),
                FilterOption(name="out_v_chr_pos", value=out_v_chr_pos),
                FilterOption(name="out_h_chr_pos", value=out_h_chr_pos),
                FilterOption(name="force_original_aspect_ratio", value=force_original_aspect_ratio),
                FilterOption(name="force_divisible_by", value=force_divisible_by),
                FilterOption(name="param0", value=param0),
                FilterOption(name="param1", value=param1),
                FilterOption(name="eval", value=eval),
            ],
        )
    )
    return graph


def scale2ref(
    graph: Stream,
):
    """Scale the input video size and/or convert the image format to the given reference."""
    graph.append(Filter(command="scale2ref", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph
