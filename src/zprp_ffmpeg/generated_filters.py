from typing import Optional

from zprp_ffmpeg.FilterGraph import Filter
from zprp_ffmpeg.FilterGraph import FilterOption
from zprp_ffmpeg.FilterGraph import Stream


def hwmap(graph: Stream, mode: Optional[str] = None, derive_device: Optional[str] = None, reverse: Optional[int] = None):
    """Map hardware frames

    :param str mode: Frame mapping mode
            possible values: read, write, overwrite, direct
    :param str derive_device: Derive a new device of this type
    :param int reverse: Map in reverse (create and allocate in the sink)
    """
    graph.append(
        Filter(
            command="hwmap",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="derive_device", value=derive_device),
                FilterOption(name="reverse", value=reverse),
            ],
        )
    )
    return graph


def vibrato(graph: Stream, f: Optional[float] = None, d: Optional[float] = None):
    """Apply vibrato effect.

    :param float f: set frequency in hertz
    :param float d: set depth as percentage
    """
    graph.append(
        Filter(
            command="vibrato", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="f", value=f), FilterOption(name="d", value=d)]
        )
    )
    return graph


def sidechaincompress(
    graph: Stream,
):
    """Sidechain compressor."""
    graph.append(Filter(command="sidechaincompress", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def acompressor(
    graph: Stream,
):
    """Audio compressor."""
    graph.append(Filter(command="acompressor", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def colormap(
    graph: Stream,
    patch_size: Optional[int] = None,
    nb_patches: Optional[int] = None,
    type: Optional[str] = None,
    kernel: Optional[str] = None,
):
    """Apply custom Color Maps to video stream.

    :param int patch_size: set patch size
    :param int nb_patches: set number of patches
    :param str type: set the target type used
            possible values: relative, absolute
    :param str kernel: set the kernel used for measuring color difference
            possible values: euclidean, weuclidean
    """
    graph.append(
        Filter(
            command="colormap",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="patch_size", value=patch_size),
                FilterOption(name="nb_patches", value=nb_patches),
                FilterOption(name="type", value=type),
                FilterOption(name="kernel", value=kernel),
            ],
        )
    )
    return graph


def separatefields(
    graph: Stream,
):
    """Split input video frames into fields."""
    graph.append(Filter(command="separatefields", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def tiltandshift(
    graph: Stream,
    tilt: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    hold: Optional[int] = None,
    pad: Optional[int] = None,
):
    """Generate a tilt-and-shift'd video.

    :param int tilt: Tilt the video horizontally while shifting
    :param str start: Action at the start of input
            possible values: none, frame, black
    :param str end: Action at the end of input
            possible values: none, frame, black
    :param int hold: Number of columns to hold at the start of the video
    :param int pad: Number of columns to pad at the end of the video
    """
    graph.append(
        Filter(
            command="tiltandshift",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="tilt", value=tilt),
                FilterOption(name="start", value=start),
                FilterOption(name="end", value=end),
                FilterOption(name="hold", value=hold),
                FilterOption(name="pad", value=pad),
            ],
        )
    )
    return graph


def w3fdif(
    graph: Stream, filter: Optional[str] = None, mode: Optional[str] = None, parity: Optional[str] = None, deint: Optional[str] = None
):
    """Apply Martin Weston three field deinterlace.

    :param str filter: specify the filter
            possible values: simple, complex
    :param str mode: specify the interlacing mode
            possible values: frame, field
    :param str parity: specify the assumed picture field parity
            possible values: tff, bff, auto
    :param str deint: specify which frames to deinterlace
            possible values: all, interlaced
    """
    graph.append(
        Filter(
            command="w3fdif",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="filter", value=filter),
                FilterOption(name="mode", value=mode),
                FilterOption(name="parity", value=parity),
                FilterOption(name="deint", value=deint),
            ],
        )
    )
    return graph


def aap(
    graph: Stream,
    order: Optional[int] = None,
    projection: Optional[int] = None,
    mu: Optional[float] = None,
    delta: Optional[float] = None,
    out_mode: Optional[str] = None,
    precision: Optional[str] = None,
):
    """Apply Affine Projection algorithm to first audio stream.

    :param int order: set the filter order
    :param int projection: set the filter projection
    :param float mu: set the filter mu
    :param float delta: set the filter delta
    :param str out_mode: set output mode
            possible values: i, d, o, n, e
    :param str precision: set processing precision
            possible values: auto, float, double
    """
    graph.append(
        Filter(
            command="aap",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="order", value=order),
                FilterOption(name="projection", value=projection),
                FilterOption(name="mu", value=mu),
                FilterOption(name="delta", value=delta),
                FilterOption(name="out_mode", value=out_mode),
                FilterOption(name="precision", value=precision),
            ],
        )
    )
    return graph


def drawbox(
    graph: Stream,
    x: Optional[str] = None,
    y: Optional[str] = None,
    width: Optional[str] = None,
    height: Optional[str] = None,
    color: Optional[str] = None,
    thickness: Optional[str] = None,
    replace: Optional[bool] = None,
    box_source: Optional[str] = None,
):
    """Draw a colored box on the input video.

    :param str x: set horizontal position of the left box edge
    :param str y: set vertical position of the top box edge
    :param str width: set width of the box
    :param str height: set height of the box
    :param str color: set color of the box
    :param str thickness: set the box thickness
    :param bool replace: replace color & alpha
    :param str box_source: use datas from bounding box in side data
    """
    graph.append(
        Filter(
            command="drawbox",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="width", value=width),
                FilterOption(name="height", value=height),
                FilterOption(name="color", value=color),
                FilterOption(name="thickness", value=thickness),
                FilterOption(name="replace", value=replace),
                FilterOption(name="box_source", value=box_source),
            ],
        )
    )
    return graph


def drawgrid(
    graph: Stream,
    x: Optional[str] = None,
    y: Optional[str] = None,
    width: Optional[str] = None,
    height: Optional[str] = None,
    color: Optional[str] = None,
    thickness: Optional[str] = None,
    replace: Optional[bool] = None,
):
    """Draw a colored grid on the input video.

    :param str x: set horizontal offset
    :param str y: set vertical offset
    :param str width: set width of grid cell
    :param str height: set height of grid cell
    :param str color: set color of the grid
    :param str thickness: set grid line thickness
    :param bool replace: replace color & alpha
    """
    graph.append(
        Filter(
            command="drawgrid",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="width", value=width),
                FilterOption(name="height", value=height),
                FilterOption(name="color", value=color),
                FilterOption(name="thickness", value=thickness),
                FilterOption(name="replace", value=replace),
            ],
        )
    )
    return graph


def earwax(
    graph: Stream,
):
    """Widen the stereo image."""
    graph.append(Filter(command="earwax", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def bbox(graph: Stream, min_val: Optional[int] = None):
    """Compute bounding box for each frame.

    :param int min_val: set minimum luminance value for bounding box
    """
    graph.append(Filter(command="bbox", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="min_val", value=min_val)]))
    return graph


def mandelbrot(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    maxiter: Optional[int] = None,
    start_x: Optional[float] = None,
    start_y: Optional[float] = None,
    start_scale: Optional[float] = None,
    end_scale: Optional[float] = None,
    end_pts: Optional[float] = None,
    bailout: Optional[float] = None,
    morphxf: Optional[float] = None,
    morphyf: Optional[float] = None,
    morphamp: Optional[float] = None,
    outer: Optional[str] = None,
    inner: Optional[str] = None,
):
    """Render a Mandelbrot fractal.

    :param int size: set frame size
    :param str rate: set frame rate
    :param int maxiter: set max iterations number
    :param float start_x: set the initial x position
    :param float start_y: set the initial y position
    :param float start_scale: set the initial scale value
    :param float end_scale: set the terminal scale value
    :param float end_pts: set the terminal pts value
    :param float bailout: set the bailout value
    :param float morphxf: set morph x frequency
    :param float morphyf: set morph y frequency
    :param float morphamp: set morph amplitude
    :param str outer: set outer coloring mode
            possible values: iteration_count, normalized_iteration_count, white, outz
    :param str inner: set inner coloring mode
            possible values: black, period, convergence, mincol
    """
    graph.append(
        Filter(
            command="mandelbrot",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="maxiter", value=maxiter),
                FilterOption(name="start_x", value=start_x),
                FilterOption(name="start_y", value=start_y),
                FilterOption(name="start_scale", value=start_scale),
                FilterOption(name="end_scale", value=end_scale),
                FilterOption(name="end_pts", value=end_pts),
                FilterOption(name="bailout", value=bailout),
                FilterOption(name="morphxf", value=morphxf),
                FilterOption(name="morphyf", value=morphyf),
                FilterOption(name="morphamp", value=morphamp),
                FilterOption(name="outer", value=outer),
                FilterOption(name="inner", value=inner),
            ],
        )
    )
    return graph


def haas(
    graph: Stream,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    side_gain: Optional[float] = None,
    middle_source: Optional[str] = None,
    middle_phase: Optional[bool] = None,
    left_delay: Optional[float] = None,
    left_balance: Optional[float] = None,
    left_gain: Optional[float] = None,
    left_phase: Optional[bool] = None,
    right_delay: Optional[float] = None,
    right_balance: Optional[float] = None,
    right_gain: Optional[float] = None,
    right_phase: Optional[bool] = None,
):
    """Apply Haas Stereo Enhancer.

    :param float level_in: set level in
    :param float level_out: set level out
    :param float side_gain: set side gain
    :param str middle_source: set middle source
            possible values: left, right, mid, side
    :param bool middle_phase: set middle phase
    :param float left_delay: set left delay
    :param float left_balance: set left balance
    :param float left_gain: set left gain
    :param bool left_phase: set left phase
    :param float right_delay: set right delay
    :param float right_balance: set right balance
    :param float right_gain: set right gain
    :param bool right_phase: set right phase
    """
    graph.append(
        Filter(
            command="haas",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="side_gain", value=side_gain),
                FilterOption(name="middle_source", value=middle_source),
                FilterOption(name="middle_phase", value=middle_phase),
                FilterOption(name="left_delay", value=left_delay),
                FilterOption(name="left_balance", value=left_balance),
                FilterOption(name="left_gain", value=left_gain),
                FilterOption(name="left_phase", value=left_phase),
                FilterOption(name="right_delay", value=right_delay),
                FilterOption(name="right_balance", value=right_balance),
                FilterOption(name="right_gain", value=right_gain),
                FilterOption(name="right_phase", value=right_phase),
            ],
        )
    )
    return graph


def random(graph: Stream, frames: Optional[int] = None, seed: Optional[int] = None):
    """Return random frames.

    :param int frames: set number of frames in cache
    :param int seed: set the seed
    """
    graph.append(
        Filter(
            command="random",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="frames", value=frames), FilterOption(name="seed", value=seed)],
        )
    )
    return graph


def aphaser(
    graph: Stream,
    in_gain: Optional[float] = None,
    out_gain: Optional[float] = None,
    delay: Optional[float] = None,
    decay: Optional[float] = None,
    speed: Optional[float] = None,
    type: Optional[str] = None,
):
    """Add a phasing effect to the audio.

    :param float in_gain: set input gain
    :param float out_gain: set output gain
    :param float delay: set delay in milliseconds
    :param float decay: set decay
    :param float speed: set modulation speed
    :param str type: set modulation type
            possible values: triangular, t, sinusoidal, s
    """
    graph.append(
        Filter(
            command="aphaser",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="in_gain", value=in_gain),
                FilterOption(name="out_gain", value=out_gain),
                FilterOption(name="delay", value=delay),
                FilterOption(name="decay", value=decay),
                FilterOption(name="speed", value=speed),
                FilterOption(name="type", value=type),
            ],
        )
    )
    return graph


def feedback(graph: Stream, x: Optional[int] = None, y: Optional[int] = None, w: Optional[int] = None, h: Optional[int] = None):
    """Apply feedback video filter.

    :param int x: set top left crop position
    :param int y: set top left crop position
    :param int w: set crop size
    :param int h: set crop size
    """
    graph.append(
        Filter(
            command="feedback",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
            ],
        )
    )
    return graph


def aexciter(
    graph: Stream,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    amount: Optional[float] = None,
    drive: Optional[float] = None,
    blend: Optional[float] = None,
    freq: Optional[float] = None,
    ceil: Optional[float] = None,
    listen: Optional[bool] = None,
):
    """Enhance high frequency part of audio.

    :param float level_in: set level in
    :param float level_out: set level out
    :param float amount: set amount
    :param float drive: set harmonics
    :param float blend: set blend harmonics
    :param float freq: set scope
    :param float ceil: set ceiling
    :param bool listen: enable listen mode
    """
    graph.append(
        Filter(
            command="aexciter",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="amount", value=amount),
                FilterOption(name="drive", value=drive),
                FilterOption(name="blend", value=blend),
                FilterOption(name="freq", value=freq),
                FilterOption(name="ceil", value=ceil),
                FilterOption(name="listen", value=listen),
            ],
        )
    )
    return graph


def aderivative(
    graph: Stream,
):
    """Compute derivative of input audio."""
    graph.append(Filter(command="aderivative", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def aintegral(
    graph: Stream,
):
    """Compute integral of input audio."""
    graph.append(Filter(command="aintegral", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def palettegen(
    graph: Stream,
    max_colors: Optional[int] = None,
    reserve_transparent: Optional[bool] = None,
    transparency_color: Optional[str] = None,
    stats_mode: Optional[str] = None,
):
    """Find the optimal palette for a given stream.

    :param int max_colors: set the maximum number of colors to use in the palette
    :param bool reserve_transparent: reserve a palette entry for transparency
    :param str transparency_color: set a background color for transparency
    :param str stats_mode: set statistics mode
            possible values: full, diff, single
    """
    graph.append(
        Filter(
            command="palettegen",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="max_colors", value=max_colors),
                FilterOption(name="reserve_transparent", value=reserve_transparent),
                FilterOption(name="transparency_color", value=transparency_color),
                FilterOption(name="stats_mode", value=stats_mode),
            ],
        )
    )
    return graph


def adrc(
    graph: Stream,
    transfer: Optional[str] = None,
    attack: Optional[float] = None,
    release: Optional[float] = None,
    channels: Optional[str] = None,
):
    """Audio Spectral Dynamic Range Controller.

    :param str transfer: set the transfer expression
    :param float attack: set the attack
    :param float release: set the release
    :param str channels: set channels to filter
    """
    graph.append(
        Filter(
            command="adrc",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="transfer", value=transfer),
                FilterOption(name="attack", value=attack),
                FilterOption(name="release", value=release),
                FilterOption(name="channels", value=channels),
            ],
        )
    )
    return graph


def loudnorm(
    graph: Stream,
    I: Optional[float] = None,
    LRA: Optional[float] = None,
    TP: Optional[float] = None,
    measured_I: Optional[float] = None,
    measured_LRA: Optional[float] = None,
    measured_TP: Optional[float] = None,
    measured_thresh: Optional[float] = None,
    offset: Optional[float] = None,
    linear: Optional[bool] = None,
    dual_mono: Optional[bool] = None,
    print_format: Optional[str] = None,
):
    """EBU R128 loudness normalization

    :param float I: set integrated loudness target
    :param float LRA: set loudness range target
    :param float TP: set maximum true peak
    :param float measured_I: measured IL of input file
    :param float measured_LRA: measured LRA of input file
    :param float measured_TP: measured true peak of input file
    :param float measured_thresh: measured threshold of input file
    :param float offset: set offset gain
    :param bool linear: normalize linearly if possible
    :param bool dual_mono: treat mono input as dual-mono
    :param str print_format: set print format for stats
            possible values: none, json, summary
    """
    graph.append(
        Filter(
            command="loudnorm",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="I", value=I),
                FilterOption(name="LRA", value=LRA),
                FilterOption(name="TP", value=TP),
                FilterOption(name="measured_I", value=measured_I),
                FilterOption(name="measured_LRA", value=measured_LRA),
                FilterOption(name="measured_TP", value=measured_TP),
                FilterOption(name="measured_thresh", value=measured_thresh),
                FilterOption(name="offset", value=offset),
                FilterOption(name="linear", value=linear),
                FilterOption(name="dual_mono", value=dual_mono),
                FilterOption(name="print_format", value=print_format),
            ],
        )
    )
    return graph


def colorcontrast(
    graph: Stream,
    rc: Optional[float] = None,
    gm: Optional[float] = None,
    by: Optional[float] = None,
    rcw: Optional[float] = None,
    gmw: Optional[float] = None,
    byw: Optional[float] = None,
    pl: Optional[float] = None,
):
    """Adjust color contrast between RGB components.

    :param float rc: set the red-cyan contrast
    :param float gm: set the green-magenta contrast
    :param float by: set the blue-yellow contrast
    :param float rcw: set the red-cyan weight
    :param float gmw: set the green-magenta weight
    :param float byw: set the blue-yellow weight
    :param float pl: set the amount of preserving lightness
    """
    graph.append(
        Filter(
            command="colorcontrast",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rc", value=rc),
                FilterOption(name="gm", value=gm),
                FilterOption(name="by", value=by),
                FilterOption(name="rcw", value=rcw),
                FilterOption(name="gmw", value=gmw),
                FilterOption(name="byw", value=byw),
                FilterOption(name="pl", value=pl),
            ],
        )
    )
    return graph


def scroll(
    graph: Stream,
    horizontal: Optional[float] = None,
    vertical: Optional[float] = None,
    hpos: Optional[float] = None,
    vpos: Optional[float] = None,
):
    """Scroll input video.

    :param float horizontal: set the horizontal scrolling speed
    :param float vertical: set the vertical scrolling speed
    :param float hpos: set initial horizontal position
    :param float vpos: set initial vertical position
    """
    graph.append(
        Filter(
            command="scroll",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="horizontal", value=horizontal),
                FilterOption(name="vertical", value=vertical),
                FilterOption(name="hpos", value=hpos),
                FilterOption(name="vpos", value=vpos),
            ],
        )
    )
    return graph


def shufflepixels(
    graph: Stream,
    direction: Optional[str] = None,
    mode: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Shuffle video pixels.

    :param str direction: set shuffle direction
            possible values: forward, inverse
    :param str mode: set shuffle mode
            possible values: horizontal, vertical, block
    :param int width: set block width
    :param int height: set block height
    :param int seed: set random seed
    """
    graph.append(
        Filter(
            command="shufflepixels",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="direction", value=direction),
                FilterOption(name="mode", value=mode),
                FilterOption(name="width", value=width),
                FilterOption(name="height", value=height),
                FilterOption(name="seed", value=seed),
            ],
        )
    )
    return graph


def sinc(
    graph: Stream,
    sample_rate: Optional[int] = None,
    nb_samples: Optional[int] = None,
    hp: Optional[float] = None,
    lp: Optional[float] = None,
    phase: Optional[float] = None,
    beta: Optional[float] = None,
    att: Optional[float] = None,
    round: Optional[bool] = None,
    hptaps: Optional[int] = None,
    lptaps: Optional[int] = None,
):
    """Generate a sinc kaiser-windowed low-pass, high-pass, band-pass, or band-reject FIR coefficients.

    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param float hp: set high-pass filter frequency
    :param float lp: set low-pass filter frequency
    :param float phase: set filter phase response
    :param float beta: set kaiser window beta
    :param float att: set stop-band attenuation
    :param bool round: enable rounding
    :param int hptaps: set number of taps for high-pass filter
    :param int lptaps: set number of taps for low-pass filter
    """
    graph.append(
        Filter(
            command="sinc",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="hp", value=hp),
                FilterOption(name="lp", value=lp),
                FilterOption(name="phase", value=phase),
                FilterOption(name="beta", value=beta),
                FilterOption(name="att", value=att),
                FilterOption(name="round", value=round),
                FilterOption(name="hptaps", value=hptaps),
                FilterOption(name="lptaps", value=lptaps),
            ],
        )
    )
    return graph


def hilbert(
    graph: Stream,
    sample_rate: Optional[int] = None,
    taps: Optional[int] = None,
    nb_samples: Optional[int] = None,
    win_func: Optional[str] = None,
):
    """Generate a Hilbert transform FIR coefficients.

    :param int sample_rate: set sample rate
    :param int taps: set number of taps
    :param int nb_samples: set the number of samples per requested frame
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    """
    graph.append(
        Filter(
            command="hilbert",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="taps", value=taps),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="win_func", value=win_func),
            ],
        )
    )
    return graph


def exposure(graph: Stream, exposure: Optional[float] = None, black: Optional[float] = None):
    """Adjust exposure of the video stream.

    :param float exposure: set the exposure correction
    :param float black: set the black level correction
    """
    graph.append(
        Filter(
            command="exposure",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="exposure", value=exposure), FilterOption(name="black", value=black)],
        )
    )
    return graph


def boxblur(
    graph: Stream,
    luma_radius: Optional[str] = None,
    lr: Optional[str] = None,
    luma_power: Optional[int] = None,
    lp: Optional[int] = None,
    chroma_radius: Optional[str] = None,
    cr: Optional[str] = None,
    chroma_power: Optional[int] = None,
    cp: Optional[int] = None,
    alpha_radius: Optional[str] = None,
    ar: Optional[str] = None,
    alpha_power: Optional[int] = None,
    ap: Optional[int] = None,
):
    """Blur the input.

    :param str luma_radius: Radius of the luma blurring box
    :param str lr: Radius of the luma blurring box
    :param int luma_power: How many times should the boxblur be applied to luma
    :param int lp: How many times should the boxblur be applied to luma
    :param str chroma_radius: Radius of the chroma blurring box
    :param str cr: Radius of the chroma blurring box
    :param int chroma_power: How many times should the boxblur be applied to chroma
    :param int cp: How many times should the boxblur be applied to chroma
    :param str alpha_radius: Radius of the alpha blurring box
    :param str ar: Radius of the alpha blurring box
    :param int alpha_power: How many times should the boxblur be applied to alpha
    :param int ap: How many times should the boxblur be applied to alpha
    """
    graph.append(
        Filter(
            command="boxblur",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="luma_radius", value=luma_radius),
                FilterOption(name="lr", value=lr),
                FilterOption(name="luma_power", value=luma_power),
                FilterOption(name="lp", value=lp),
                FilterOption(name="chroma_radius", value=chroma_radius),
                FilterOption(name="cr", value=cr),
                FilterOption(name="chroma_power", value=chroma_power),
                FilterOption(name="cp", value=cp),
                FilterOption(name="alpha_radius", value=alpha_radius),
                FilterOption(name="ar", value=ar),
                FilterOption(name="alpha_power", value=alpha_power),
                FilterOption(name="ap", value=ap),
            ],
        )
    )
    return graph


def copy(
    graph: Stream,
):
    """Copy the input video unchanged to the output."""
    graph.append(Filter(command="copy", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def addroi(
    graph: Stream,
    x: Optional[str] = None,
    y: Optional[str] = None,
    w: Optional[str] = None,
    h: Optional[str] = None,
    qoffset: Optional[int] = None,
    clear: Optional[bool] = None,
):
    """Add region of interest to frame.

    :param str x: Region distance from left edge of frame.
    :param str y: Region distance from top edge of frame.
    :param str w: Region width.
    :param str h: Region height.
    :param int qoffset: Quantisation offset to apply in the region.
    :param bool clear: Remove any existing regions of interest before adding the new one.
    """
    graph.append(
        Filter(
            command="addroi",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="qoffset", value=qoffset),
                FilterOption(name="clear", value=clear),
            ],
        )
    )
    return graph


def afftfilt(
    graph: Stream,
    real: Optional[str] = None,
    imag: Optional[str] = None,
    win_size: Optional[int] = None,
    win_func: Optional[str] = None,
    overlap: Optional[float] = None,
):
    """Apply arbitrary expressions to samples in frequency domain.

    :param str real: set channels real expressions
    :param str imag: set channels imaginary expressions
    :param int win_size: set window size
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    :param float overlap: set window overlap
    """
    graph.append(
        Filter(
            command="afftfilt",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="real", value=real),
                FilterOption(name="imag", value=imag),
                FilterOption(name="win_size", value=win_size),
                FilterOption(name="win_func", value=win_func),
                FilterOption(name="overlap", value=overlap),
            ],
        )
    )
    return graph


def colortemperature(graph: Stream, temperature: Optional[float] = None, mix: Optional[float] = None, pl: Optional[float] = None):
    """Adjust color temperature of video.

    :param float temperature: set the temperature in Kelvin
    :param float mix: set the mix with filtered output
    :param float pl: set the amount of preserving lightness
    """
    graph.append(
        Filter(
            command="colortemperature",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="temperature", value=temperature),
                FilterOption(name="mix", value=mix),
                FilterOption(name="pl", value=pl),
            ],
        )
    )
    return graph


def colorcorrect(
    graph: Stream,
    rl: Optional[float] = None,
    bl: Optional[float] = None,
    rh: Optional[float] = None,
    bh: Optional[float] = None,
    saturation: Optional[float] = None,
    analyze: Optional[str] = None,
):
    """Adjust color white balance selectively for blacks and whites.

    :param float rl: set the red shadow spot
    :param float bl: set the blue shadow spot
    :param float rh: set the red highlight spot
    :param float bh: set the blue highlight spot
    :param float saturation: set the amount of saturation
    :param str analyze: set the analyze mode
            possible values: manual, average, minmax, median
    """
    graph.append(
        Filter(
            command="colorcorrect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rl", value=rl),
                FilterOption(name="bl", value=bl),
                FilterOption(name="rh", value=rh),
                FilterOption(name="bh", value=bh),
                FilterOption(name="saturation", value=saturation),
                FilterOption(name="analyze", value=analyze),
            ],
        )
    )
    return graph


def maskfun(
    graph: Stream,
    low: Optional[int] = None,
    high: Optional[int] = None,
    planes: Optional[int] = None,
    fill: Optional[int] = None,
    sum: Optional[int] = None,
):
    """Create Mask.

    :param int low: set low threshold
    :param int high: set high threshold
    :param int planes: set planes
    :param int fill: set fill value
    :param int sum: set sum value
    """
    graph.append(
        Filter(
            command="maskfun",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="low", value=low),
                FilterOption(name="high", value=high),
                FilterOption(name="planes", value=planes),
                FilterOption(name="fill", value=fill),
                FilterOption(name="sum", value=sum),
            ],
        )
    )
    return graph


def scale_vaapi(
    graph: Stream,
    w: Optional[str] = None,
    h: Optional[str] = None,
    format: Optional[str] = None,
    mode: Optional[str] = None,
    out_color_matrix: Optional[str] = None,
    out_range: Optional[str] = None,
    out_color_primaries: Optional[str] = None,
    out_color_transfer: Optional[str] = None,
    out_chroma_location: Optional[str] = None,
    force_original_aspect_ratio: Optional[str] = None,
    force_divisible_by: Optional[int] = None,
):
    """Scale to/from VAAPI surfaces.

    :param str w: Output video width
    :param str h: Output video height
    :param str format: Output video format (software format of hardware frames)
    :param str mode: Scaling mode
            possible values: default, fast, hq, nl_anamorphic
    :param str out_color_matrix: Output colour matrix coefficient set
    :param str out_range: Output colour range
            possible values: full, limited, jpeg, mpeg, tv, pc
    :param str out_color_primaries: Output colour primaries
    :param str out_color_transfer: Output colour transfer characteristics
    :param str out_chroma_location: Output chroma sample location
    :param str force_original_aspect_ratio: decrease or increase w/h if necessary to keep the original AR
            possible values: disable, decrease, increase
    :param int force_divisible_by: enforce that the output resolution is divisible by a defined integer when force_original_aspect_ratio is used
    """
    graph.append(
        Filter(
            command="scale_vaapi",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="format", value=format),
                FilterOption(name="mode", value=mode),
                FilterOption(name="out_color_matrix", value=out_color_matrix),
                FilterOption(name="out_range", value=out_range),
                FilterOption(name="out_color_primaries", value=out_color_primaries),
                FilterOption(name="out_color_transfer", value=out_color_transfer),
                FilterOption(name="out_chroma_location", value=out_chroma_location),
                FilterOption(name="force_original_aspect_ratio", value=force_original_aspect_ratio),
                FilterOption(name="force_divisible_by", value=force_divisible_by),
            ],
        )
    )
    return graph


def afirsrc(
    graph: Stream,
    taps: Optional[int] = None,
    frequency: Optional[str] = None,
    magnitude: Optional[str] = None,
    phase: Optional[str] = None,
    sample_rate: Optional[int] = None,
    nb_samples: Optional[int] = None,
    win_func: Optional[str] = None,
):
    """Generate a FIR coefficients audio stream.

    :param int taps: set number of taps
    :param str frequency: set frequency points
    :param str magnitude: set magnitude values
    :param str phase: set phase values
    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    """
    graph.append(
        Filter(
            command="afirsrc",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="taps", value=taps),
                FilterOption(name="frequency", value=frequency),
                FilterOption(name="magnitude", value=magnitude),
                FilterOption(name="phase", value=phase),
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="win_func", value=win_func),
            ],
        )
    )
    return graph


def afireqsrc(
    graph: Stream,
    preset: Optional[str] = None,
    gains: Optional[str] = None,
    bands: Optional[str] = None,
    taps: Optional[int] = None,
    sample_rate: Optional[int] = None,
    nb_samples: Optional[int] = None,
    interp: Optional[str] = None,
    phase: Optional[str] = None,
):
    """Generate a FIR equalizer coefficients audio stream.

    :param str preset: set equalizer preset
            possible values: custom
    :param str gains: set gain values per band
    :param str bands: set central frequency values per band
    :param int taps: set number of taps
    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param str interp: set the interpolation
            possible values: linear, cubic
    :param str phase: set the phase
            possible values: linear, min
    """
    graph.append(
        Filter(
            command="afireqsrc",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="preset", value=preset),
                FilterOption(name="gains", value=gains),
                FilterOption(name="bands", value=bands),
                FilterOption(name="taps", value=taps),
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="interp", value=interp),
                FilterOption(name="phase", value=phase),
            ],
        )
    )
    return graph


def crop(
    graph: Stream,
    out_w: Optional[str] = None,
    out_h: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    keep_aspect: Optional[bool] = None,
    exact: Optional[bool] = None,
):
    """Crop the input video.

    :param str out_w: set the width crop area expression
    :param str out_h: set the height crop area expression
    :param str x: set the x crop area expression
    :param str y: set the y crop area expression
    :param bool keep_aspect: keep aspect ratio
    :param bool exact: do exact cropping
    """
    graph.append(
        Filter(
            command="crop",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="out_w", value=out_w),
                FilterOption(name="out_h", value=out_h),
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="keep_aspect", value=keep_aspect),
                FilterOption(name="exact", value=exact),
            ],
        )
    )
    return graph


def acopy(
    graph: Stream,
):
    """Copy the input audio unchanged to the output."""
    graph.append(Filter(command="acopy", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def readeia608(
    graph: Stream,
    scan_min: Optional[int] = None,
    scan_max: Optional[int] = None,
    spw: Optional[float] = None,
    chp: Optional[bool] = None,
    lp: Optional[bool] = None,
):
    """Read EIA-608 Closed Caption codes from input video and write them to frame metadata.

    :param int scan_min: set from which line to scan for codes
    :param int scan_max: set to which line to scan for codes
    :param float spw: set ratio of width reserved for sync code detection
    :param bool chp: check and apply parity bit
    :param bool lp: lowpass line prior to processing
    """
    graph.append(
        Filter(
            command="readeia608",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="scan_min", value=scan_min),
                FilterOption(name="scan_max", value=scan_max),
                FilterOption(name="spw", value=spw),
                FilterOption(name="chp", value=chp),
                FilterOption(name="lp", value=lp),
            ],
        )
    )
    return graph


def zoompan(
    graph: Stream,
    zoom: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    d: Optional[str] = None,
    s: Optional[int] = None,
    fps: Optional[str] = None,
):
    """Apply Zoom & Pan effect.

    :param str zoom: set the zoom expression
    :param str x: set the x expression
    :param str y: set the y expression
    :param str d: set the duration expression
    :param int s: set the output image size
    :param str fps: set the output framerate
    """
    graph.append(
        Filter(
            command="zoompan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="zoom", value=zoom),
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="d", value=d),
                FilterOption(name="s", value=s),
                FilterOption(name="fps", value=fps),
            ],
        )
    )
    return graph


def colorbalance(
    graph: Stream,
    rs: Optional[float] = None,
    gs: Optional[float] = None,
    bs: Optional[float] = None,
    rm: Optional[float] = None,
    gm: Optional[float] = None,
    bm: Optional[float] = None,
    rh: Optional[float] = None,
    gh: Optional[float] = None,
    bh: Optional[float] = None,
    pl: Optional[bool] = None,
):
    """Adjust the color balance.

    :param float rs: set red shadows
    :param float gs: set green shadows
    :param float bs: set blue shadows
    :param float rm: set red midtones
    :param float gm: set green midtones
    :param float bm: set blue midtones
    :param float rh: set red highlights
    :param float gh: set green highlights
    :param float bh: set blue highlights
    :param bool pl: preserve lightness
    """
    graph.append(
        Filter(
            command="colorbalance",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rs", value=rs),
                FilterOption(name="gs", value=gs),
                FilterOption(name="bs", value=bs),
                FilterOption(name="rm", value=rm),
                FilterOption(name="gm", value=gm),
                FilterOption(name="bm", value=bm),
                FilterOption(name="rh", value=rh),
                FilterOption(name="gh", value=gh),
                FilterOption(name="bh", value=bh),
                FilterOption(name="pl", value=pl),
            ],
        )
    )
    return graph


def premultiply(
    graph: Stream,
):
    """PreMultiply first stream with first plane of second stream."""
    graph.append(Filter(command="premultiply", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def unpremultiply(
    graph: Stream,
):
    """UnPreMultiply first stream with first plane of second stream."""
    graph.append(Filter(command="unpremultiply", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def graphmonitor(
    graph: Stream,
    size: Optional[int] = None,
    opacity: Optional[float] = None,
    mode: Optional[str] = None,
    flags: Optional[str] = None,
    rate: Optional[str] = None,
):
    """Show various filtergraph stats.

    :param int size: set monitor size
    :param float opacity: set video opacity
    :param str mode: set mode
            possible values: full, compact, nozero, noeof, nodisabled
    :param str flags: set flags
            possible values: none, all, queue, frame_count_in, frame_count_out, frame_count_delta, pts, pts_delta, time, time_delta, timebase, format, size, rate, eof, sample_count_in, sample_count_out, sample_count_delta, disabled
    :param str rate: set video rate
    """
    graph.append(
        Filter(
            command="graphmonitor",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="opacity", value=opacity),
                FilterOption(name="mode", value=mode),
                FilterOption(name="flags", value=flags),
                FilterOption(name="rate", value=rate),
            ],
        )
    )
    return graph


def agraphmonitor(
    graph: Stream,
):
    """Show various filtergraph stats."""
    graph.append(Filter(command="agraphmonitor", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def untile(graph: Stream, layout: Optional[int] = None):
    """Untile a frame into a sequence of frames.

    :param int layout: set grid size
    """
    graph.append(Filter(command="untile", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="layout", value=layout)]))
    return graph


def vmafmotion(graph: Stream, stats_file: Optional[str] = None):
    """Calculate the VMAF Motion score.

    :param str stats_file: Set file where to store per-frame difference information
    """
    graph.append(Filter(command="vmafmotion", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="stats_file", value=stats_file)]))
    return graph


def blockdetect(graph: Stream, period_min: Optional[int] = None, period_max: Optional[int] = None, planes: Optional[int] = None):
    """Blockdetect filter.

    :param int period_min: Minimum period to search for
    :param int period_max: Maximum period to search for
    :param int planes: set planes to filter
    """
    graph.append(
        Filter(
            command="blockdetect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="period_min", value=period_min),
                FilterOption(name="period_max", value=period_max),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def tmidequalizer(graph: Stream, radius: Optional[int] = None, sigma: Optional[float] = None, planes: Optional[int] = None):
    """Apply Temporal Midway Equalization.

    :param int radius: set radius
    :param float sigma: set sigma
    :param int planes: set planes
    """
    graph.append(
        Filter(
            command="tmidequalizer",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="radius", value=radius),
                FilterOption(name="sigma", value=sigma),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def overlay_vulkan(graph: Stream, x: Optional[int] = None, y: Optional[int] = None):
    """Overlay a source on top of another

    :param int x: Set horizontal offset
    :param int y: Set vertical offset
    """
    graph.append(
        Filter(
            command="overlay_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="x", value=x), FilterOption(name="y", value=y)],
        )
    )
    return graph


def despill(
    graph: Stream,
    type: Optional[str] = None,
    mix: Optional[float] = None,
    expand: Optional[float] = None,
    red: Optional[float] = None,
    green: Optional[float] = None,
    blue: Optional[float] = None,
    brightness: Optional[float] = None,
    alpha: Optional[bool] = None,
):
    """Despill video.

    :param str type: set the screen type
            possible values: green, blue
    :param float mix: set the spillmap mix
    :param float expand: set the spillmap expand
    :param float red: set red scale
    :param float green: set green scale
    :param float blue: set blue scale
    :param float brightness: set brightness
    :param bool alpha: change alpha component
    """
    graph.append(
        Filter(
            command="despill",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="type", value=type),
                FilterOption(name="mix", value=mix),
                FilterOption(name="expand", value=expand),
                FilterOption(name="red", value=red),
                FilterOption(name="green", value=green),
                FilterOption(name="blue", value=blue),
                FilterOption(name="brightness", value=brightness),
                FilterOption(name="alpha", value=alpha),
            ],
        )
    )
    return graph


def volume(
    graph: Stream,
    volume: Optional[str] = None,
    precision: Optional[str] = None,
    eval: Optional[str] = None,
    replaygain: Optional[str] = None,
    replaygain_preamp: Optional[float] = None,
    replaygain_noclip: Optional[bool] = None,
):
    """Change input volume.

    :param str volume: set volume adjustment expression
    :param str precision: select mathematical precision
            possible values: fixed, float, double
    :param str eval: specify when to evaluate expressions
            possible values: once, frame
    :param str replaygain: Apply replaygain side data when present
            possible values: drop, ignore, track, album
    :param float replaygain_preamp: Apply replaygain pre-amplification
    :param bool replaygain_noclip: Apply replaygain clipping prevention
    """
    graph.append(
        Filter(
            command="volume",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="volume", value=volume),
                FilterOption(name="precision", value=precision),
                FilterOption(name="eval", value=eval),
                FilterOption(name="replaygain", value=replaygain),
                FilterOption(name="replaygain_preamp", value=replaygain_preamp),
                FilterOption(name="replaygain_noclip", value=replaygain_noclip),
            ],
        )
    )
    return graph


def unsharp(
    graph: Stream,
    luma_msize_x: Optional[int] = None,
    luma_msize_y: Optional[int] = None,
    luma_amount: Optional[float] = None,
    chroma_msize_x: Optional[int] = None,
    chroma_msize_y: Optional[int] = None,
    chroma_amount: Optional[float] = None,
    alpha_msize_x: Optional[int] = None,
    alpha_msize_y: Optional[int] = None,
    alpha_amount: Optional[float] = None,
):
    """Sharpen or blur the input video.

    :param int luma_msize_x: set luma matrix horizontal size
    :param int luma_msize_y: set luma matrix vertical size
    :param float luma_amount: set luma effect strength
    :param int chroma_msize_x: set chroma matrix horizontal size
    :param int chroma_msize_y: set chroma matrix vertical size
    :param float chroma_amount: set chroma effect strength
    :param int alpha_msize_x: set alpha matrix horizontal size
    :param int alpha_msize_y: set alpha matrix vertical size
    :param float alpha_amount: set alpha effect strength
    """
    graph.append(
        Filter(
            command="unsharp",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="luma_msize_x", value=luma_msize_x),
                FilterOption(name="luma_msize_y", value=luma_msize_y),
                FilterOption(name="luma_amount", value=luma_amount),
                FilterOption(name="chroma_msize_x", value=chroma_msize_x),
                FilterOption(name="chroma_msize_y", value=chroma_msize_y),
                FilterOption(name="chroma_amount", value=chroma_amount),
                FilterOption(name="alpha_msize_x", value=alpha_msize_x),
                FilterOption(name="alpha_msize_y", value=alpha_msize_y),
                FilterOption(name="alpha_amount", value=alpha_amount),
            ],
        )
    )
    return graph


def swapuv(
    graph: Stream,
):
    """Swap U and V components."""
    graph.append(Filter(command="swapuv", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def avgblur_vulkan(graph: Stream, sizeX: Optional[int] = None, sizeY: Optional[int] = None, planes: Optional[int] = None):
    """Apply avgblur mask to input video

    :param int sizeX: Set horizontal radius
    :param int sizeY: Set vertical radius
    :param int planes: Set planes to filter (bitmask)
    """
    graph.append(
        Filter(
            command="avgblur_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="sizeX", value=sizeX),
                FilterOption(name="sizeY", value=sizeY),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def pad_opencl(
    graph: Stream,
    width: Optional[str] = None,
    height: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    aspect: Optional[int] = None,
):
    """Pad the input video.

    :param str width: set the pad area width
    :param str height: set the pad area height
    :param str x: set the x offset for the input image position
    :param str y: set the y offset for the input image position
    :param str color: set the color of the padded area border
    :param int aspect: pad to fit an aspect instead of a resolution
    """
    graph.append(
        Filter(
            command="pad_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="width", value=width),
                FilterOption(name="height", value=height),
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="color", value=color),
                FilterOption(name="aspect", value=aspect),
            ],
        )
    )
    return graph


def limiter(graph: Stream, min: Optional[int] = None, max: Optional[int] = None, planes: Optional[int] = None):
    """Limit pixels components to the specified range.

    :param int min: set min value
    :param int max: set max value
    :param int planes: set planes
    """
    graph.append(
        Filter(
            command="limiter",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="min", value=min), FilterOption(name="max", value=max), FilterOption(name="planes", value=planes)],
        )
    )
    return graph


def apulsator(
    graph: Stream,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    mode: Optional[str] = None,
    amount: Optional[float] = None,
    offset_l: Optional[float] = None,
    offset_r: Optional[float] = None,
    width: Optional[float] = None,
    timing: Optional[str] = None,
    bpm: Optional[float] = None,
    ms: Optional[int] = None,
    hz: Optional[float] = None,
):
    """Audio pulsator.

    :param float level_in: set input gain
    :param float level_out: set output gain
    :param str mode: set mode
            possible values: sine, triangle, square, sawup, sawdown
    :param float amount: set modulation
    :param float offset_l: set offset L
    :param float offset_r: set offset R
    :param float width: set pulse width
    :param str timing: set timing
            possible values: bpm, ms, hz
    :param float bpm: set BPM
    :param int ms: set ms
    :param float hz: set frequency
    """
    graph.append(
        Filter(
            command="apulsator",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="mode", value=mode),
                FilterOption(name="amount", value=amount),
                FilterOption(name="offset_l", value=offset_l),
                FilterOption(name="offset_r", value=offset_r),
                FilterOption(name="width", value=width),
                FilterOption(name="timing", value=timing),
                FilterOption(name="bpm", value=bpm),
                FilterOption(name="ms", value=ms),
                FilterOption(name="hz", value=hz),
            ],
        )
    )
    return graph


def il(
    graph: Stream,
    luma_mode: Optional[str] = None,
    chroma_mode: Optional[str] = None,
    alpha_mode: Optional[str] = None,
    luma_swap: Optional[bool] = None,
    chroma_swap: Optional[bool] = None,
    alpha_swap: Optional[bool] = None,
):
    """Deinterleave or interleave fields.

    :param str luma_mode: select luma mode
            possible values: none, interleave, i, deinterleave, d
    :param str chroma_mode: select chroma mode
            possible values: none, interleave, i, deinterleave, d
    :param str alpha_mode: select alpha mode
            possible values: none, interleave, i, deinterleave, d
    :param bool luma_swap: swap luma fields
    :param bool chroma_swap: swap chroma fields
    :param bool alpha_swap: swap alpha fields
    """
    graph.append(
        Filter(
            command="il",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="luma_mode", value=luma_mode),
                FilterOption(name="chroma_mode", value=chroma_mode),
                FilterOption(name="alpha_mode", value=alpha_mode),
                FilterOption(name="luma_swap", value=luma_swap),
                FilterOption(name="chroma_swap", value=chroma_swap),
                FilterOption(name="alpha_swap", value=alpha_swap),
            ],
        )
    )
    return graph


def morpho(graph: Stream, mode: Optional[str] = None, planes: Optional[int] = None, structure: Optional[str] = None):
    """Apply Morphological filter.

    :param str mode: set morphological transform
            possible values: erode, dilate, open, close, gradient, tophat, blackhat
    :param int planes: set planes to filter
    :param str structure: when to process structures
            possible values: first, all
    """
    graph.append(
        Filter(
            command="morpho",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="planes", value=planes),
                FilterOption(name="structure", value=structure),
            ],
        )
    )
    return graph


def lut(
    graph: Stream,
):
    """Compute and apply a lookup table to the RGB/YUV input video."""
    graph.append(Filter(command="lut", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def lutyuv(
    graph: Stream,
):
    """Compute and apply a lookup table to the YUV input video."""
    graph.append(Filter(command="lutyuv", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def lutrgb(
    graph: Stream,
):
    """Compute and apply a lookup table to the RGB input video."""
    graph.append(Filter(command="lutrgb", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def negate(graph: Stream, components: Optional[str] = None, negate_alpha: Optional[bool] = None):
    """Negate input video.

    :param str components: set components to negate
            possible values: y, u, v, r, g, b, a
    """
    graph.append(
        Filter(
            command="negate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="components", value=components), FilterOption(name="negate_alpha", value=negate_alpha)],
        )
    )
    return graph


def hwupload(graph: Stream, derive_device: Optional[str] = None):
    """Upload a normal frame to a hardware frame

    :param str derive_device: Derive a new device of this type
    """
    graph.append(
        Filter(command="hwupload", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="derive_device", value=derive_device)])
    )
    return graph


def dejudder(graph: Stream, cycle: Optional[int] = None):
    """Remove judder produced by pullup.

    :param int cycle: set the length of the cycle to use for dejuddering
    """
    graph.append(Filter(command="dejudder", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="cycle", value=cycle)]))
    return graph


def fsync(graph: Stream, file: Optional[str] = None):
    """Synchronize video frames from external source.

    :param str file: set the file name to use for frame sync
    """
    graph.append(Filter(command="fsync", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="file", value=file)]))
    return graph


def tonemap(
    graph: Stream, tonemap: Optional[str] = None, param: Optional[float] = None, desat: Optional[float] = None, peak: Optional[float] = None
):
    """Conversion to/from different dynamic ranges.

    :param str tonemap: tonemap algorithm selection
            possible values: none, linear, gamma, clip, reinhard, hable, mobius
    :param float param: tonemap parameter
    :param float desat: desaturation strength
    :param float peak: signal peak override
    """
    graph.append(
        Filter(
            command="tonemap",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="tonemap", value=tonemap),
                FilterOption(name="param", value=param),
                FilterOption(name="desat", value=desat),
                FilterOption(name="peak", value=peak),
            ],
        )
    )
    return graph


def superequalizer(
    graph: Stream,
    _1b: Optional[float] = None,
    _2b: Optional[float] = None,
    _3b: Optional[float] = None,
    _4b: Optional[float] = None,
    _5b: Optional[float] = None,
    _6b: Optional[float] = None,
    _7b: Optional[float] = None,
    _8b: Optional[float] = None,
    _9b: Optional[float] = None,
    _10b: Optional[float] = None,
    _11b: Optional[float] = None,
    _12b: Optional[float] = None,
    _13b: Optional[float] = None,
    _14b: Optional[float] = None,
    _15b: Optional[float] = None,
    _16b: Optional[float] = None,
    _17b: Optional[float] = None,
    _18b: Optional[float] = None,
):
    """Apply 18 band equalization filter.

    :param float 1b: set 65Hz band gain
    :param float 2b: set 92Hz band gain
    :param float 3b: set 131Hz band gain
    :param float 4b: set 185Hz band gain
    :param float 5b: set 262Hz band gain
    :param float 6b: set 370Hz band gain
    :param float 7b: set 523Hz band gain
    :param float 8b: set 740Hz band gain
    :param float 9b: set 1047Hz band gain
    :param float 10b: set 1480Hz band gain
    :param float 11b: set 2093Hz band gain
    :param float 12b: set 2960Hz band gain
    :param float 13b: set 4186Hz band gain
    :param float 14b: set 5920Hz band gain
    :param float 15b: set 8372Hz band gain
    :param float 16b: set 11840Hz band gain
    :param float 17b: set 16744Hz band gain
    :param float 18b: set 20000Hz band gain
    """
    graph.append(
        Filter(
            command="superequalizer",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="1b", value=_1b),
                FilterOption(name="2b", value=_2b),
                FilterOption(name="3b", value=_3b),
                FilterOption(name="4b", value=_4b),
                FilterOption(name="5b", value=_5b),
                FilterOption(name="6b", value=_6b),
                FilterOption(name="7b", value=_7b),
                FilterOption(name="8b", value=_8b),
                FilterOption(name="9b", value=_9b),
                FilterOption(name="10b", value=_10b),
                FilterOption(name="11b", value=_11b),
                FilterOption(name="12b", value=_12b),
                FilterOption(name="13b", value=_13b),
                FilterOption(name="14b", value=_14b),
                FilterOption(name="15b", value=_15b),
                FilterOption(name="16b", value=_16b),
                FilterOption(name="17b", value=_17b),
                FilterOption(name="18b", value=_18b),
            ],
        )
    )
    return graph


def asubboost(
    graph: Stream,
    dry: Optional[float] = None,
    wet: Optional[float] = None,
    boost: Optional[float] = None,
    decay: Optional[float] = None,
    feedback: Optional[float] = None,
    cutoff: Optional[float] = None,
    slope: Optional[float] = None,
    delay: Optional[float] = None,
    channels: Optional[str] = None,
):
    """Boost subwoofer frequencies.

    :param float dry: set dry gain
    :param float wet: set wet gain
    :param float boost: set max boost
    :param float decay: set decay
    :param float feedback: set feedback
    :param float cutoff: set cutoff
    :param float slope: set slope
    :param float delay: set delay
    :param str channels: set channels to filter
    """
    graph.append(
        Filter(
            command="asubboost",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="dry", value=dry),
                FilterOption(name="wet", value=wet),
                FilterOption(name="boost", value=boost),
                FilterOption(name="decay", value=decay),
                FilterOption(name="feedback", value=feedback),
                FilterOption(name="cutoff", value=cutoff),
                FilterOption(name="slope", value=slope),
                FilterOption(name="delay", value=delay),
                FilterOption(name="channels", value=channels),
            ],
        )
    )
    return graph


def afade(
    graph: Stream,
    type: Optional[str] = None,
    start_sample: Optional[int] = None,
    nb_samples: Optional[int] = None,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
    curve: Optional[str] = None,
    silence: Optional[float] = None,
    unity: Optional[float] = None,
):
    """Fade in/out input audio.

    :param str type: set the fade direction
            possible values: in, out
    :param int start_sample: set number of first sample to start fading
    :param int nb_samples: set number of samples for fade duration
    :param int start_time: set time to start fading
    :param int duration: set fade duration
    :param str curve: set fade curve type
            possible values: nofade, tri, qsin, esin, hsin, log, ipar, qua, cub, squ, cbr, par, exp, iqsin, ihsin, dese, desi, losi, sinc, isinc, quat, quatr, qsin2, hsin2
    :param float silence: set the silence gain
    :param float unity: set the unity gain
    """
    graph.append(
        Filter(
            command="afade",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="type", value=type),
                FilterOption(name="start_sample", value=start_sample),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="start_time", value=start_time),
                FilterOption(name="duration", value=duration),
                FilterOption(name="curve", value=curve),
                FilterOption(name="silence", value=silence),
                FilterOption(name="unity", value=unity),
            ],
        )
    )
    return graph


def acrossfade(
    graph: Stream,
    nb_samples: Optional[int] = None,
    duration: Optional[int] = None,
    overlap: Optional[bool] = None,
    curve1: Optional[str] = None,
    curve2: Optional[int] = None,
):
    """Cross fade two input audio streams.

    :param int nb_samples: set number of samples for cross fade duration
    :param int duration: set cross fade duration
    :param bool overlap: overlap 1st stream end with 2nd stream start
    :param str curve1: set fade curve type for 1st stream
            possible values: nofade, tri, qsin, esin, hsin, log, ipar, qua, cub, squ, cbr, par, exp, iqsin, ihsin, dese, desi, losi, sinc, isinc, quat, quatr, qsin2, hsin2
    :param int curve2: set fade curve type for 2nd stream
    """
    graph.append(
        Filter(
            command="acrossfade",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="duration", value=duration),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="curve1", value=curve1),
                FilterOption(name="curve2", value=curve2),
            ],
        )
    )
    return graph


def pp(graph: Stream, subfilters: Optional[str] = None):
    """Filter video using libpostproc.

    :param str subfilters: set postprocess subfilters
    """
    graph.append(Filter(command="pp", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="subfilters", value=subfilters)]))
    return graph


def histeq(graph: Stream, strength: Optional[float] = None, intensity: Optional[float] = None, antibanding: Optional[str] = None):
    """Apply global color histogram equalization.

    :param float strength: set the strength
    :param float intensity: set the intensity
    :param str antibanding: set the antibanding level
            possible values: none, weak, strong
    """
    graph.append(
        Filter(
            command="histeq",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="strength", value=strength),
                FilterOption(name="intensity", value=intensity),
                FilterOption(name="antibanding", value=antibanding),
            ],
        )
    )
    return graph


def ashowinfo(
    graph: Stream,
):
    """Show textual information for each audio frame."""
    graph.append(Filter(command="ashowinfo", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def afftdn(
    graph: Stream,
    noise_reduction: Optional[float] = None,
    noise_floor: Optional[float] = None,
    noise_type: Optional[str] = None,
    band_noise: Optional[str] = None,
    residual_floor: Optional[float] = None,
    track_noise: Optional[bool] = None,
    track_residual: Optional[bool] = None,
    output_mode: Optional[str] = None,
    adaptivity: Optional[float] = None,
    floor_offset: Optional[float] = None,
    noise_link: Optional[str] = None,
    band_multiplier: Optional[float] = None,
    sample_noise: Optional[str] = None,
    gain_smooth: Optional[int] = None,
):
    """Denoise audio samples using FFT.

    :param float noise_reduction: set the noise reduction
    :param float noise_floor: set the noise floor
    :param str noise_type: set the noise type
            possible values: white, w, vinyl, v, shellac, s, custom, c
    :param str band_noise: set the custom bands noise
    :param float residual_floor: set the residual floor
    :param bool track_noise: track noise
    :param bool track_residual: track residual
    :param str output_mode: set output mode
            possible values: input, i, output, o, noise, n
    :param float adaptivity: set adaptivity factor
    :param float floor_offset: set noise floor offset factor
    :param str noise_link: set the noise floor link
            possible values: none, min, max, average
    :param float band_multiplier: set band multiplier
    :param str sample_noise: set sample noise mode
            possible values: none, start, begin, stop, end
    :param int gain_smooth: set gain smooth radius
    """
    graph.append(
        Filter(
            command="afftdn",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="noise_reduction", value=noise_reduction),
                FilterOption(name="noise_floor", value=noise_floor),
                FilterOption(name="noise_type", value=noise_type),
                FilterOption(name="band_noise", value=band_noise),
                FilterOption(name="residual_floor", value=residual_floor),
                FilterOption(name="track_noise", value=track_noise),
                FilterOption(name="track_residual", value=track_residual),
                FilterOption(name="output_mode", value=output_mode),
                FilterOption(name="adaptivity", value=adaptivity),
                FilterOption(name="floor_offset", value=floor_offset),
                FilterOption(name="noise_link", value=noise_link),
                FilterOption(name="band_multiplier", value=band_multiplier),
                FilterOption(name="sample_noise", value=sample_noise),
                FilterOption(name="gain_smooth", value=gain_smooth),
            ],
        )
    )
    return graph


def signature(
    graph: Stream,
    detectmode: Optional[str] = None,
    nb_inputs: Optional[int] = None,
    filename: Optional[str] = None,
    format: Optional[str] = None,
    th_d: Optional[int] = None,
    th_dc: Optional[int] = None,
    th_xh: Optional[int] = None,
    th_di: Optional[int] = None,
    th_it: Optional[float] = None,
):
    """Calculate the MPEG-7 video signature

    :param str detectmode: set the detectmode
            possible values: off, full, fast
    :param int nb_inputs: number of inputs
    :param str filename: filename for output files
    :param str format: set output format
            possible values: binary, xml
    :param int th_d: threshold to detect one word as similar
    :param int th_dc: threshold to detect all words as similar
    :param int th_xh: threshold to detect frames as similar
    :param int th_di: minimum length of matching sequence in frames
    :param float th_it: threshold for relation of good to all frames
    """
    graph.append(
        Filter(
            command="signature",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="detectmode", value=detectmode),
                FilterOption(name="nb_inputs", value=nb_inputs),
                FilterOption(name="filename", value=filename),
                FilterOption(name="format", value=format),
                FilterOption(name="th_d", value=th_d),
                FilterOption(name="th_dc", value=th_dc),
                FilterOption(name="th_xh", value=th_xh),
                FilterOption(name="th_di", value=th_di),
                FilterOption(name="th_it", value=th_it),
            ],
        )
    )
    return graph


def adynamicequalizer(
    graph: Stream,
    threshold: Optional[float] = None,
    dfrequency: Optional[float] = None,
    dqfactor: Optional[float] = None,
    tfrequency: Optional[float] = None,
    tqfactor: Optional[float] = None,
    attack: Optional[float] = None,
    release: Optional[float] = None,
    ratio: Optional[float] = None,
    makeup: Optional[float] = None,
    range: Optional[float] = None,
    mode: Optional[str] = None,
    dftype: Optional[str] = None,
    tftype: Optional[str] = None,
    auto: Optional[str] = None,
    precision: Optional[str] = None,
):
    """Apply Dynamic Equalization of input audio.

    :param float threshold: set detection threshold
    :param float dfrequency: set detection frequency
    :param float dqfactor: set detection Q factor
    :param float tfrequency: set target frequency
    :param float tqfactor: set target Q factor
    :param float attack: set detection attack duration
    :param float release: set detection release duration
    :param float ratio: set ratio factor
    :param float makeup: set makeup gain
    :param float range: set max gain
    :param str mode: set mode
            possible values: listen, cutbelow, cutabove, boostbelow, boostabove
    :param str dftype: set detection filter type
            possible values: bandpass, lowpass, highpass, peak
    :param str tftype: set target filter type
            possible values: bell, lowshelf, highshelf
    :param str auto: set auto threshold
            possible values: disabled, off, on, adaptive
    :param str precision: set processing precision
            possible values: auto, float, double
    """
    graph.append(
        Filter(
            command="adynamicequalizer",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="dfrequency", value=dfrequency),
                FilterOption(name="dqfactor", value=dqfactor),
                FilterOption(name="tfrequency", value=tfrequency),
                FilterOption(name="tqfactor", value=tqfactor),
                FilterOption(name="attack", value=attack),
                FilterOption(name="release", value=release),
                FilterOption(name="ratio", value=ratio),
                FilterOption(name="makeup", value=makeup),
                FilterOption(name="range", value=range),
                FilterOption(name="mode", value=mode),
                FilterOption(name="dftype", value=dftype),
                FilterOption(name="tftype", value=tftype),
                FilterOption(name="auto", value=auto),
                FilterOption(name="precision", value=precision),
            ],
        )
    )
    return graph


def sine(
    graph: Stream,
    frequency: Optional[float] = None,
    beep_factor: Optional[float] = None,
    sample_rate: Optional[int] = None,
    duration: Optional[int] = None,
    samples_per_frame: Optional[str] = None,
):
    """Generate sine wave audio signal.

    :param float frequency: set the sine frequency
    :param float beep_factor: set the beep frequency factor
    :param int sample_rate: set the sample rate
    :param int duration: set the audio duration
    :param str samples_per_frame: set the number of samples per frame
    """
    graph.append(
        Filter(
            command="sine",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="frequency", value=frequency),
                FilterOption(name="beep_factor", value=beep_factor),
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="samples_per_frame", value=samples_per_frame),
            ],
        )
    )
    return graph


def anlms(
    graph: Stream,
    order: Optional[int] = None,
    mu: Optional[float] = None,
    eps: Optional[float] = None,
    leakage: Optional[float] = None,
    out_mode: Optional[str] = None,
    precision: Optional[str] = None,
):
    """Apply Normalized Least-Mean-Squares algorithm to first audio stream.

    :param int order: set the filter order
    :param float mu: set the filter mu
    :param float eps: set the filter eps
    :param float leakage: set the filter leakage
    :param str out_mode: set output mode
            possible values: i, d, o, n, e
    :param str precision: set processing precision
            possible values: auto, float, double
    """
    graph.append(
        Filter(
            command="anlms",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="order", value=order),
                FilterOption(name="mu", value=mu),
                FilterOption(name="eps", value=eps),
                FilterOption(name="leakage", value=leakage),
                FilterOption(name="out_mode", value=out_mode),
                FilterOption(name="precision", value=precision),
            ],
        )
    )
    return graph


def anlmf(
    graph: Stream,
):
    """Apply Normalized Least-Mean-Fourth algorithm to first audio stream."""
    graph.append(Filter(command="anlmf", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def maskedthreshold(graph: Stream, threshold: Optional[int] = None, planes: Optional[int] = None, mode: Optional[str] = None):
    """Pick pixels comparing absolute difference of two streams with threshold.

    :param int threshold: set threshold
    :param int planes: set planes
    :param str mode: set mode
            possible values: abs, diff
    """
    graph.append(
        Filter(
            command="maskedthreshold",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="planes", value=planes),
                FilterOption(name="mode", value=mode),
            ],
        )
    )
    return graph


def alimiter(
    graph: Stream,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    limit: Optional[float] = None,
    attack: Optional[float] = None,
    release: Optional[float] = None,
    asc: Optional[bool] = None,
    asc_level: Optional[float] = None,
    level: Optional[bool] = None,
    latency: Optional[bool] = None,
):
    """Audio lookahead limiter.

    :param float level_in: set input level
    :param float level_out: set output level
    :param float limit: set limit
    :param float attack: set attack
    :param float release: set release
    :param bool asc: enable asc
    :param float asc_level: set asc level
    :param bool level: auto level
    :param bool latency: compensate delay
    """
    graph.append(
        Filter(
            command="alimiter",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="limit", value=limit),
                FilterOption(name="attack", value=attack),
                FilterOption(name="release", value=release),
                FilterOption(name="asc", value=asc),
                FilterOption(name="asc_level", value=asc_level),
                FilterOption(name="level", value=level),
                FilterOption(name="latency", value=latency),
            ],
        )
    )
    return graph


def asidedata(graph: Stream, mode: Optional[str] = None, type: Optional[str] = None):
    """Manipulate audio frame side data.

    :param str mode: set a mode of operation
            possible values: select, delete
    :param str type: set side data type
            possible values: PANSCAN, A53_CC, STEREO3D, MATRIXENCODING, DOWNMIX_INFO, REPLAYGAIN, DISPLAYMATRIX, AFD, MOTION_VECTORS, SKIP_SAMPLES, AUDIO_SERVICE_TYPE, MASTERING_DISPLAY_METADATA, GOP_TIMECODE, SPHERICAL, CONTENT_LIGHT_LEVEL, ICC_PROFILE, S12M_TIMECOD, DYNAMIC_HDR_PLUS, REGIONS_OF_INTEREST, VIDEO_ENC_PARAMS, SEI_UNREGISTERED, FILM_GRAIN_PARAMS, DETECTION_BOUNDING_BOXES, DETECTION_BBOXES, DOVI_RPU_BUFFER, DOVI_METADATA, DYNAMIC_HDR_VIVID, AMBIENT_VIEWING_ENVIRONMENT, VIDEO_HINT
    """
    graph.append(
        Filter(
            command="asidedata",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="mode", value=mode), FilterOption(name="type", value=type)],
        )
    )
    return graph


def sidedata(graph: Stream, mode: Optional[str] = None, type: Optional[str] = None):
    """Manipulate video frame side data.

    :param str mode: set a mode of operation
            possible values: select, delete
    :param str type: set side data type
            possible values: PANSCAN, A53_CC, STEREO3D, MATRIXENCODING, DOWNMIX_INFO, REPLAYGAIN, DISPLAYMATRIX, AFD, MOTION_VECTORS, SKIP_SAMPLES, AUDIO_SERVICE_TYPE, MASTERING_DISPLAY_METADATA, GOP_TIMECODE, SPHERICAL, CONTENT_LIGHT_LEVEL, ICC_PROFILE, S12M_TIMECOD, DYNAMIC_HDR_PLUS, REGIONS_OF_INTEREST, VIDEO_ENC_PARAMS, SEI_UNREGISTERED, FILM_GRAIN_PARAMS, DETECTION_BOUNDING_BOXES, DETECTION_BBOXES, DOVI_RPU_BUFFER, DOVI_METADATA, DYNAMIC_HDR_VIVID, AMBIENT_VIEWING_ENVIRONMENT, VIDEO_HINT
    """
    graph.append(
        Filter(
            command="sidedata",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="mode", value=mode), FilterOption(name="type", value=type)],
        )
    )
    return graph


def siti(graph: Stream, print_summary: Optional[bool] = None):
    """Calculate spatial information (SI) and temporal information (TI).

    :param bool print_summary: Print summary showing average values
    """
    graph.append(Filter(command="siti", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="print_summary", value=print_summary)]))
    return graph


def sab(
    graph: Stream,
    luma_radius: Optional[float] = None,
    lr: Optional[float] = None,
    luma_pre_filter_radius: Optional[float] = None,
    lpfr: Optional[float] = None,
    luma_strength: Optional[float] = None,
    ls: Optional[float] = None,
    chroma_radius: Optional[float] = None,
    cr: Optional[float] = None,
    chroma_pre_filter_radius: Optional[float] = None,
    cpfr: Optional[float] = None,
    chroma_strength: Optional[float] = None,
    cs: Optional[float] = None,
):
    """Apply shape adaptive blur.

    :param float luma_radius: set luma radius
    :param float lr: set luma radius
    :param float luma_pre_filter_radius: set luma pre-filter radius
    :param float lpfr: set luma pre-filter radius
    :param float luma_strength: set luma strength
    :param float ls: set luma strength
    :param float chroma_radius: set chroma radius
    :param float cr: set chroma radius
    :param float chroma_pre_filter_radius: set chroma pre-filter radius
    :param float cpfr: set chroma pre-filter radius
    :param float chroma_strength: set chroma strength
    :param float cs: set chroma strength
    """
    graph.append(
        Filter(
            command="sab",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="luma_radius", value=luma_radius),
                FilterOption(name="lr", value=lr),
                FilterOption(name="luma_pre_filter_radius", value=luma_pre_filter_radius),
                FilterOption(name="lpfr", value=lpfr),
                FilterOption(name="luma_strength", value=luma_strength),
                FilterOption(name="ls", value=ls),
                FilterOption(name="chroma_radius", value=chroma_radius),
                FilterOption(name="cr", value=cr),
                FilterOption(name="chroma_pre_filter_radius", value=chroma_pre_filter_radius),
                FilterOption(name="cpfr", value=cpfr),
                FilterOption(name="chroma_strength", value=chroma_strength),
                FilterOption(name="cs", value=cs),
            ],
        )
    )
    return graph


def agate(
    graph: Stream,
):
    """Audio gate."""
    graph.append(Filter(command="agate", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def sidechaingate(
    graph: Stream,
):
    """Audio sidechain gate."""
    graph.append(Filter(command="sidechaingate", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def lut2(
    graph: Stream,
):
    """Compute and apply a lookup table from two video inputs."""
    graph.append(Filter(command="lut2", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def tlut2(graph: Stream, c0: Optional[str] = None, c1: Optional[str] = None, c2: Optional[str] = None, c3: Optional[str] = None):
    """Compute and apply a lookup table from two successive frames.

    :param str c0: set component #0 expression
    :param str c1: set component #1 expression
    :param str c2: set component #2 expression
    :param str c3: set component #3 expression
    """
    graph.append(
        Filter(
            command="tlut2",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="c0", value=c0),
                FilterOption(name="c1", value=c1),
                FilterOption(name="c2", value=c2),
                FilterOption(name="c3", value=c3),
            ],
        )
    )
    return graph


def deband(
    graph: Stream,
    _1thr: Optional[float] = None,
    _2thr: Optional[float] = None,
    _3thr: Optional[float] = None,
    _4thr: Optional[float] = None,
    range: Optional[int] = None,
    direction: Optional[float] = None,
    blur: Optional[bool] = None,
    coupling: Optional[bool] = None,
):
    """Debands video.

    :param float 1thr: set 1st plane threshold
    :param float 2thr: set 2nd plane threshold
    :param float 3thr: set 3rd plane threshold
    :param float 4thr: set 4th plane threshold
    :param int range: set range
    :param float direction: set direction
    :param bool blur: set blur
    :param bool coupling: set plane coupling
    """
    graph.append(
        Filter(
            command="deband",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="1thr", value=_1thr),
                FilterOption(name="2thr", value=_2thr),
                FilterOption(name="3thr", value=_3thr),
                FilterOption(name="4thr", value=_4thr),
                FilterOption(name="range", value=range),
                FilterOption(name="direction", value=direction),
                FilterOption(name="blur", value=blur),
                FilterOption(name="coupling", value=coupling),
            ],
        )
    )
    return graph


def median(
    graph: Stream,
    radius: Optional[int] = None,
    planes: Optional[int] = None,
    radiusV: Optional[int] = None,
    percentile: Optional[float] = None,
):
    """Apply Median filter.

    :param int radius: set median radius
    :param int planes: set planes to filter
    :param int radiusV: set median vertical radius
    :param float percentile: set median percentile
    """
    graph.append(
        Filter(
            command="median",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="radius", value=radius),
                FilterOption(name="planes", value=planes),
                FilterOption(name="radiusV", value=radiusV),
                FilterOption(name="percentile", value=percentile),
            ],
        )
    )
    return graph


def paletteuse(
    graph: Stream,
    dither: Optional[str] = None,
    bayer_scale: Optional[int] = None,
    diff_mode: Optional[str] = None,
    new: Optional[bool] = None,
    alpha_threshold: Optional[int] = None,
    debug_kdtree: Optional[str] = None,
):
    """Use a palette to downsample an input video stream.

    :param str dither: select dithering mode
            possible values: bayer, heckbert, floyd_steinberg, sierra2, sierra2_4a, sierra3, burkes, atkinson
    :param int bayer_scale: set scale for bayer dithering
    :param str diff_mode: set frame difference mode
            possible values: rectangle
    :param bool new: take new palette for each output frame
    :param int alpha_threshold: set the alpha threshold for transparency
    :param str debug_kdtree: save Graphviz graph of the kdtree in specified file
    """
    graph.append(
        Filter(
            command="paletteuse",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="dither", value=dither),
                FilterOption(name="bayer_scale", value=bayer_scale),
                FilterOption(name="diff_mode", value=diff_mode),
                FilterOption(name="new", value=new),
                FilterOption(name="alpha_threshold", value=alpha_threshold),
                FilterOption(name="debug_kdtree", value=debug_kdtree),
            ],
        )
    )
    return graph


def amplify(
    graph: Stream,
    radius: Optional[int] = None,
    factor: Optional[float] = None,
    threshold: Optional[float] = None,
    tolerance: Optional[float] = None,
    low: Optional[float] = None,
    high: Optional[float] = None,
    planes: Optional[str] = None,
):
    """Amplify changes between successive video frames.

    :param int radius: set radius
    :param float factor: set factor
    :param float threshold: set threshold
    :param float tolerance: set tolerance
    :param float low: set low limit for amplification
    :param float high: set high limit for amplification
    :param str planes: set what planes to filter
    """
    graph.append(
        Filter(
            command="amplify",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="radius", value=radius),
                FilterOption(name="factor", value=factor),
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="tolerance", value=tolerance),
                FilterOption(name="low", value=low),
                FilterOption(name="high", value=high),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def asupercut(graph: Stream, cutoff: Optional[float] = None, order: Optional[int] = None, level: Optional[float] = None):
    """Cut super frequencies.

    :param float cutoff: set cutoff frequency
    :param int order: set filter order
    :param float level: set input level
    """
    graph.append(
        Filter(
            command="asupercut",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="cutoff", value=cutoff),
                FilterOption(name="order", value=order),
                FilterOption(name="level", value=level),
            ],
        )
    )
    return graph


def asubcut(graph: Stream, cutoff: Optional[float] = None, order: Optional[int] = None, level: Optional[float] = None):
    """Cut subwoofer frequencies.

    :param float cutoff: set cutoff frequency
    :param int order: set filter order
    :param float level: set input level
    """
    graph.append(
        Filter(
            command="asubcut",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="cutoff", value=cutoff),
                FilterOption(name="order", value=order),
                FilterOption(name="level", value=level),
            ],
        )
    )
    return graph


def asuperpass(
    graph: Stream,
):
    """Apply high order Butterworth band-pass filter."""
    graph.append(Filter(command="asuperpass", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def asuperstop(
    graph: Stream,
):
    """Apply high order Butterworth band-stop filter."""
    graph.append(Filter(command="asuperstop", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def dctdnoiz(
    graph: Stream, sigma: Optional[float] = None, overlap: Optional[int] = None, expr: Optional[str] = None, n: Optional[int] = None
):
    """Denoise frames using 2D DCT.

    :param float sigma: set noise sigma constant
    :param int overlap: set number of block overlapping pixels
    :param str expr: set coefficient factor expression
    :param int n: set the block size, expressed in bits
    """
    graph.append(
        Filter(
            command="dctdnoiz",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="sigma", value=sigma),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="expr", value=expr),
                FilterOption(name="n", value=n),
            ],
        )
    )
    return graph


def chromanr(
    graph: Stream,
    thres: Optional[float] = None,
    sizew: Optional[int] = None,
    sizeh: Optional[int] = None,
    stepw: Optional[int] = None,
    steph: Optional[int] = None,
    threy: Optional[float] = None,
    threu: Optional[float] = None,
    threv: Optional[float] = None,
    distance: Optional[str] = None,
):
    """Reduce chrominance noise.

    :param float thres: set y+u+v threshold
    :param int sizew: set horizontal patch size
    :param int sizeh: set vertical patch size
    :param int stepw: set horizontal step
    :param int steph: set vertical step
    :param float threy: set y threshold
    :param float threu: set u threshold
    :param float threv: set v threshold
    :param str distance: set distance type
            possible values: manhattan, euclidean
    """
    graph.append(
        Filter(
            command="chromanr",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="thres", value=thres),
                FilterOption(name="sizew", value=sizew),
                FilterOption(name="sizeh", value=sizeh),
                FilterOption(name="stepw", value=stepw),
                FilterOption(name="steph", value=steph),
                FilterOption(name="threy", value=threy),
                FilterOption(name="threu", value=threu),
                FilterOption(name="threv", value=threv),
                FilterOption(name="distance", value=distance),
            ],
        )
    )
    return graph


def colorkey(graph: Stream, color: Optional[str] = None, similarity: Optional[float] = None, blend: Optional[float] = None):
    """Turns a certain color into transparency. Operates on RGB colors.

    :param str color: set the colorkey key color
    :param float similarity: set the colorkey similarity value
    :param float blend: set the colorkey key blend value
    """
    graph.append(
        Filter(
            command="colorkey",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="color", value=color),
                FilterOption(name="similarity", value=similarity),
                FilterOption(name="blend", value=blend),
            ],
        )
    )
    return graph


def colorhold(graph: Stream, color: Optional[str] = None, similarity: Optional[float] = None, blend: Optional[float] = None):
    """Turns a certain color range into gray. Operates on RGB colors.

    :param str color: set the colorhold key color
    :param float similarity: set the colorhold similarity value
    :param float blend: set the colorhold blend value
    """
    graph.append(
        Filter(
            command="colorhold",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="color", value=color),
                FilterOption(name="similarity", value=similarity),
                FilterOption(name="blend", value=blend),
            ],
        )
    )
    return graph


def corr(
    graph: Stream,
):
    """Calculate the correlation between two video streams."""
    graph.append(Filter(command="corr", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def aresample(
    graph: Stream,
):
    """Resample audio data."""
    graph.append(Filter(command="aresample", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def removelogo(graph: Stream, filename: Optional[str] = None):
    """Remove a TV logo based on a mask image.

    :param str filename: set bitmap filename
    """
    graph.append(Filter(command="removelogo", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="filename", value=filename)]))
    return graph


def showfreqs(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    mode: Optional[str] = None,
    ascale: Optional[str] = None,
    fscale: Optional[str] = None,
    win_size: Optional[int] = None,
    win_func: Optional[str] = None,
    overlap: Optional[float] = None,
    averaging: Optional[int] = None,
    colors: Optional[str] = None,
    cmode: Optional[str] = None,
    minamp: Optional[float] = None,
    data: Optional[str] = None,
    channels: Optional[str] = None,
):
    """Convert input audio to a frequencies video output.

    :param int size: set video size
    :param str rate: set video rate
    :param str mode: set display mode
            possible values: line, bar, dot
    :param str ascale: set amplitude scale
            possible values: lin, sqrt, cbrt, log
    :param str fscale: set frequency scale
            possible values: lin, log, rlog
    :param int win_size: set window size
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    :param float overlap: set window overlap
    :param int averaging: set time averaging
    :param str colors: set channels colors
    :param str cmode: set channel mode
            possible values: combined, separate
    :param float minamp: set minimum amplitude
    :param str data: set data mode
            possible values: magnitude, phase, delay
    :param str channels: set channels to draw
    """
    graph.append(
        Filter(
            command="showfreqs",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="mode", value=mode),
                FilterOption(name="ascale", value=ascale),
                FilterOption(name="fscale", value=fscale),
                FilterOption(name="win_size", value=win_size),
                FilterOption(name="win_func", value=win_func),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="averaging", value=averaging),
                FilterOption(name="colors", value=colors),
                FilterOption(name="cmode", value=cmode),
                FilterOption(name="minamp", value=minamp),
                FilterOption(name="data", value=data),
                FilterOption(name="channels", value=channels),
            ],
        )
    )
    return graph


def trim(
    graph: Stream,
    start: Optional[int] = None,
    end: Optional[int] = None,
    start_pts: Optional[int] = None,
    end_pts: Optional[int] = None,
    duration: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    """Pick one continuous section from the input, drop the rest.

    :param int start: Timestamp of the first frame that should be passed
    :param int end: Timestamp of the first frame that should be dropped again
    :param int start_pts: Timestamp of the first frame that should be  passed
    :param int end_pts: Timestamp of the first frame that should be dropped again
    :param int duration: Maximum duration of the output
    :param int start_frame: Number of the first frame that should be passed to the output
    :param int end_frame: Number of the first frame that should be dropped again
    """
    graph.append(
        Filter(
            command="trim",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="start", value=start),
                FilterOption(name="end", value=end),
                FilterOption(name="start_pts", value=start_pts),
                FilterOption(name="end_pts", value=end_pts),
                FilterOption(name="duration", value=duration),
                FilterOption(name="start_frame", value=start_frame),
                FilterOption(name="end_frame", value=end_frame),
            ],
        )
    )
    return graph


def atrim(
    graph: Stream,
    start: Optional[int] = None,
    end: Optional[int] = None,
    start_pts: Optional[int] = None,
    end_pts: Optional[int] = None,
    duration: Optional[int] = None,
    start_sample: Optional[int] = None,
    end_sample: Optional[int] = None,
):
    """Pick one continuous section from the input, drop the rest.

    :param int start: Timestamp of the first frame that should be passed
    :param int end: Timestamp of the first frame that should be dropped again
    :param int start_pts: Timestamp of the first frame that should be  passed
    :param int end_pts: Timestamp of the first frame that should be dropped again
    :param int duration: Maximum duration of the output
    :param int start_sample: Number of the first audio sample that should be passed to the output
    :param int end_sample: Number of the first audio sample that should be dropped again
    """
    graph.append(
        Filter(
            command="atrim",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="start", value=start),
                FilterOption(name="end", value=end),
                FilterOption(name="start_pts", value=start_pts),
                FilterOption(name="end_pts", value=end_pts),
                FilterOption(name="duration", value=duration),
                FilterOption(name="start_sample", value=start_sample),
                FilterOption(name="end_sample", value=end_sample),
            ],
        )
    )
    return graph


def sendcmd(
    graph: Stream,
):
    """Send commands to filters."""
    graph.append(Filter(command="sendcmd", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def asendcmd(
    graph: Stream,
):
    """Send commands to filters."""
    graph.append(Filter(command="asendcmd", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def scdet(graph: Stream, threshold: Optional[float] = None, sc_pass: Optional[bool] = None):
    """Detect video scene change

    :param float threshold: set scene change detect threshold
    :param bool sc_pass: Set the flag to pass scene change frames
    """
    graph.append(
        Filter(
            command="scdet",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="threshold", value=threshold), FilterOption(name="sc_pass", value=sc_pass)],
        )
    )
    return graph


def blend(
    graph: Stream,
    c0_mode: Optional[str] = None,
    c1_mode: Optional[str] = None,
    c2_mode: Optional[str] = None,
    c3_mode: Optional[str] = None,
    all_mode: Optional[str] = None,
    c0_expr: Optional[str] = None,
    c1_expr: Optional[str] = None,
    c2_expr: Optional[str] = None,
    c3_expr: Optional[str] = None,
    all_expr: Optional[str] = None,
    c0_opacity: Optional[float] = None,
    c1_opacity: Optional[float] = None,
    c2_opacity: Optional[float] = None,
    c3_opacity: Optional[float] = None,
    all_opacity: Optional[float] = None,
):
    """Blend two video frames into each other.

    :param str c0_mode: set component #0 blend mode
            possible values: addition, addition128, grainmerge, and, average, burn, darken, difference, difference128, grainextract, divide, dodge, exclusion, extremity, freeze, glow, hardlight, hardmix, heat, lighten, linearlight, multiply, multiply128, negation, normal, or, overlay, phoenix, pinlight, reflect, screen, softlight, subtract, vividlight, xor, softdifference, geometric, harmonic, bleach, stain, interpolate, hardoverlay
    :param str c1_mode: set component #1 blend mode
            possible values: addition, addition128, grainmerge, and, average, burn, darken, difference, difference128, grainextract, divide, dodge, exclusion, extremity, freeze, glow, hardlight, hardmix, heat, lighten, linearlight, multiply, multiply128, negation, normal, or, overlay, phoenix, pinlight, reflect, screen, softlight, subtract, vividlight, xor, softdifference, geometric, harmonic, bleach, stain, interpolate, hardoverlay
    :param str c2_mode: set component #2 blend mode
            possible values: addition, addition128, grainmerge, and, average, burn, darken, difference, difference128, grainextract, divide, dodge, exclusion, extremity, freeze, glow, hardlight, hardmix, heat, lighten, linearlight, multiply, multiply128, negation, normal, or, overlay, phoenix, pinlight, reflect, screen, softlight, subtract, vividlight, xor, softdifference, geometric, harmonic, bleach, stain, interpolate, hardoverlay
    :param str c3_mode: set component #3 blend mode
            possible values: addition, addition128, grainmerge, and, average, burn, darken, difference, difference128, grainextract, divide, dodge, exclusion, extremity, freeze, glow, hardlight, hardmix, heat, lighten, linearlight, multiply, multiply128, negation, normal, or, overlay, phoenix, pinlight, reflect, screen, softlight, subtract, vividlight, xor, softdifference, geometric, harmonic, bleach, stain, interpolate, hardoverlay
    :param str all_mode: set blend mode for all components
            possible values: addition, addition128, grainmerge, and, average, burn, darken, difference, difference128, grainextract, divide, dodge, exclusion, extremity, freeze, glow, hardlight, hardmix, heat, lighten, linearlight, multiply, multiply128, negation, normal, or, overlay, phoenix, pinlight, reflect, screen, softlight, subtract, vividlight, xor, softdifference, geometric, harmonic, bleach, stain, interpolate, hardoverlay
    :param str c0_expr: set color component #0 expression
    :param str c1_expr: set color component #1 expression
    :param str c2_expr: set color component #2 expression
    :param str c3_expr: set color component #3 expression
    :param str all_expr: set expression for all color components
    :param float c0_opacity: set color component #0 opacity
    :param float c1_opacity: set color component #1 opacity
    :param float c2_opacity: set color component #2 opacity
    :param float c3_opacity: set color component #3 opacity
    :param float all_opacity: set opacity for all color components
    """
    graph.append(
        Filter(
            command="blend",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="c0_mode", value=c0_mode),
                FilterOption(name="c1_mode", value=c1_mode),
                FilterOption(name="c2_mode", value=c2_mode),
                FilterOption(name="c3_mode", value=c3_mode),
                FilterOption(name="all_mode", value=all_mode),
                FilterOption(name="c0_expr", value=c0_expr),
                FilterOption(name="c1_expr", value=c1_expr),
                FilterOption(name="c2_expr", value=c2_expr),
                FilterOption(name="c3_expr", value=c3_expr),
                FilterOption(name="all_expr", value=all_expr),
                FilterOption(name="c0_opacity", value=c0_opacity),
                FilterOption(name="c1_opacity", value=c1_opacity),
                FilterOption(name="c2_opacity", value=c2_opacity),
                FilterOption(name="c3_opacity", value=c3_opacity),
                FilterOption(name="all_opacity", value=all_opacity),
            ],
        )
    )
    return graph


def tblend(
    graph: Stream,
):
    """Blend successive frames."""
    graph.append(Filter(command="tblend", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def ciescope(
    graph: Stream,
    system: Optional[str] = None,
    cie: Optional[str] = None,
    gamuts: Optional[str] = None,
    size: Optional[int] = None,
    intensity: Optional[float] = None,
    contrast: Optional[float] = None,
    corrgamma: Optional[bool] = None,
    showwhite: Optional[bool] = None,
    gamma: Optional[float] = None,
    fill: Optional[bool] = None,
):
    """Video CIE scope.

    :param str system: set color system
            possible values: ntsc, 470m, ebu, 470bg, smpte, 240m, apple, widergb, cie1931, hdtv, rec709, uhdtv, rec2020, dcip3
    :param str cie: set cie system
            possible values: xyy, ucs, luv
    :param str gamuts: set what gamuts to draw
            possible values: ntsc, 470m, ebu, 470bg, smpte, 240m, apple, widergb, cie1931, hdtv, rec709, uhdtv, rec2020, dcip3
    :param int size: set ciescope size
    :param float intensity: set ciescope intensity
    :param bool fill: fill with CIE colors
    """
    graph.append(
        Filter(
            command="ciescope",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="system", value=system),
                FilterOption(name="cie", value=cie),
                FilterOption(name="gamuts", value=gamuts),
                FilterOption(name="size", value=size),
                FilterOption(name="intensity", value=intensity),
                FilterOption(name="contrast", value=contrast),
                FilterOption(name="corrgamma", value=corrgamma),
                FilterOption(name="showwhite", value=showwhite),
                FilterOption(name="gamma", value=gamma),
                FilterOption(name="fill", value=fill),
            ],
        )
    )
    return graph


def find_rect(
    graph: Stream,
    object: Optional[str] = None,
    threshold: Optional[float] = None,
    mipmaps: Optional[int] = None,
    xmin: Optional[int] = None,
    ymin: Optional[int] = None,
    xmax: Optional[int] = None,
    ymax: Optional[int] = None,
    discard: Optional[bool] = None,
):
    """Find a user specified object.

    :param str object: object bitmap filename
    :param float threshold: set threshold
    :param int mipmaps: set mipmaps
    """
    graph.append(
        Filter(
            command="find_rect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="object", value=object),
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="mipmaps", value=mipmaps),
                FilterOption(name="xmin", value=xmin),
                FilterOption(name="ymin", value=ymin),
                FilterOption(name="xmax", value=xmax),
                FilterOption(name="ymax", value=ymax),
                FilterOption(name="discard", value=discard),
            ],
        )
    )
    return graph


def fieldorder(graph: Stream, order: Optional[str] = None):
    """Set the field order.

    :param str order: output field order
            possible values: bff, tff
    """
    graph.append(Filter(command="fieldorder", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="order", value=order)]))
    return graph


def mix(
    graph: Stream,
    inputs: Optional[int] = None,
    weights: Optional[str] = None,
    scale: Optional[float] = None,
    planes: Optional[str] = None,
    duration: Optional[str] = None,
):
    """Mix video inputs.

    :param int inputs: set number of inputs
    :param str weights: set weight for each input
    :param float scale: set scale
    :param str planes: set what planes to filter
    :param str duration: how to determine end of stream
            possible values: longest, shortest, first
    """
    graph.append(
        Filter(
            command="mix",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="inputs", value=inputs),
                FilterOption(name="weights", value=weights),
                FilterOption(name="scale", value=scale),
                FilterOption(name="planes", value=planes),
                FilterOption(name="duration", value=duration),
            ],
        )
    )
    return graph


def tmix(
    graph: Stream, frames: Optional[int] = None, weights: Optional[str] = None, scale: Optional[float] = None, planes: Optional[str] = None
):
    """Mix successive video frames.

    :param int frames: set number of successive frames to mix
    :param str weights: set weight for each frame
    :param float scale: set scale
    :param str planes: set what planes to filter
    """
    graph.append(
        Filter(
            command="tmix",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="frames", value=frames),
                FilterOption(name="weights", value=weights),
                FilterOption(name="scale", value=scale),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def transpose_vulkan(graph: Stream, dir: Optional[str] = None, passthrough: Optional[str] = None):
    """Transpose Vulkan Filter

    :param str dir: set transpose direction
            possible values: cclock_flip, clock, cclock, clock_flip
    :param str passthrough: do not apply transposition if the input matches the specified geometry
            possible values: none, portrait, landscape
    """
    graph.append(
        Filter(
            command="transpose_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="dir", value=dir), FilterOption(name="passthrough", value=passthrough)],
        )
    )
    return graph


def derain(
    graph: Stream,
    filter_type: Optional[str] = None,
    dnn_backend: Optional[int] = None,
    model: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
):
    """Apply derain filter to the input.

    :param str filter_type: filter type(derain/dehaze)
            possible values: derain, dehaze
    :param int dnn_backend: DNN backend
    :param str model: path to model file
    :param str input: input name of the model
    :param str output: output name of the model
    """
    graph.append(
        Filter(
            command="derain",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="filter_type", value=filter_type),
                FilterOption(name="dnn_backend", value=dnn_backend),
                FilterOption(name="model", value=model),
                FilterOption(name="input", value=input),
                FilterOption(name="output", value=output),
            ],
        )
    )
    return graph


def limitdiff(
    graph: Stream,
    threshold: Optional[float] = None,
    elasticity: Optional[float] = None,
    reference: Optional[bool] = None,
    planes: Optional[int] = None,
):
    """Apply filtering with limiting difference.

    :param float threshold: set the threshold
    :param float elasticity: set the elasticity
    :param bool reference: enable reference stream
    :param int planes: set the planes to filter
    """
    graph.append(
        Filter(
            command="limitdiff",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="elasticity", value=elasticity),
                FilterOption(name="reference", value=reference),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def identity(
    graph: Stream,
):
    """Calculate the Identity between two video streams."""
    graph.append(Filter(command="identity", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def msad(
    graph: Stream,
):
    """Calculate the MSAD between two video streams."""
    graph.append(Filter(command="msad", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def erosion(
    graph: Stream,
):
    """Apply erosion effect."""
    graph.append(Filter(command="erosion", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def dilation(
    graph: Stream,
):
    """Apply dilation effect."""
    graph.append(Filter(command="dilation", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def deflate(
    graph: Stream,
):
    """Apply deflate effect."""
    graph.append(Filter(command="deflate", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def inflate(
    graph: Stream,
):
    """Apply inflate effect."""
    graph.append(Filter(command="inflate", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def maskedmin(
    graph: Stream,
):
    """Apply filtering with minimum difference of two streams."""
    graph.append(Filter(command="maskedmin", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def maskedmax(
    graph: Stream,
):
    """Apply filtering with maximum difference of two streams."""
    graph.append(Filter(command="maskedmax", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def histogram(
    graph: Stream,
    level_height: Optional[int] = None,
    scale_height: Optional[int] = None,
    display_mode: Optional[str] = None,
    levels_mode: Optional[str] = None,
    components: Optional[int] = None,
    fgopacity: Optional[float] = None,
    bgopacity: Optional[float] = None,
    colors_mode: Optional[str] = None,
):
    """Compute and draw a histogram.

    :param int level_height: set level height
    :param int scale_height: set scale height
    :param str display_mode: set display mode
            possible values: overlay, parade, stack
    :param str levels_mode: set levels mode
            possible values: linear, logarithmic
    :param int components: set color components to display
    :param float fgopacity: set foreground opacity
    :param float bgopacity: set background opacity
    :param str colors_mode: set colors mode
            possible values: whiteonblack, blackonwhite, whiteongray, blackongray, coloronblack, coloronwhite, colorongray, blackoncolor, whiteoncolor, grayoncolor
    """
    graph.append(
        Filter(
            command="histogram",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="level_height", value=level_height),
                FilterOption(name="scale_height", value=scale_height),
                FilterOption(name="display_mode", value=display_mode),
                FilterOption(name="levels_mode", value=levels_mode),
                FilterOption(name="components", value=components),
                FilterOption(name="fgopacity", value=fgopacity),
                FilterOption(name="bgopacity", value=bgopacity),
                FilterOption(name="colors_mode", value=colors_mode),
            ],
        )
    )
    return graph


def thistogram(
    graph: Stream,
    width: Optional[int] = None,
    display_mode: Optional[str] = None,
    levels_mode: Optional[str] = None,
    components: Optional[int] = None,
    bgopacity: Optional[float] = None,
    envelope: Optional[bool] = None,
    ecolor: Optional[str] = None,
    slide: Optional[str] = None,
):
    """Compute and draw a temporal histogram.

    :param int width: set width
    :param str display_mode: set display mode
            possible values: overlay, parade, stack
    :param str levels_mode: set levels mode
            possible values: linear, logarithmic
    :param int components: set color components to display
    :param float bgopacity: set background opacity
    :param bool envelope: display envelope
    :param str ecolor: set envelope color
    :param str slide: set slide mode
            possible values: frame, replace, scroll, rscroll, picture
    """
    graph.append(
        Filter(
            command="thistogram",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="width", value=width),
                FilterOption(name="display_mode", value=display_mode),
                FilterOption(name="levels_mode", value=levels_mode),
                FilterOption(name="components", value=components),
                FilterOption(name="bgopacity", value=bgopacity),
                FilterOption(name="envelope", value=envelope),
                FilterOption(name="ecolor", value=ecolor),
                FilterOption(name="slide", value=slide),
            ],
        )
    )
    return graph


def hqx(graph: Stream, n: Optional[int] = None):
    """Scale the input by 2, 3 or 4 using the hq*x magnification algorithm.

    :param int n: set scale factor
    """
    graph.append(Filter(command="hqx", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="n", value=n)]))
    return graph


def datascope(
    graph: Stream,
    size: Optional[int] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    mode: Optional[str] = None,
    axis: Optional[bool] = None,
    opacity: Optional[float] = None,
    format: Optional[str] = None,
    components: Optional[int] = None,
):
    """Video data analysis.

    :param int size: set output size
    :param int x: set x offset
    :param int y: set y offset
    :param str mode: set scope mode
            possible values: mono, color, color2
    :param bool axis: draw column/row numbers
    :param float opacity: set background opacity
    :param str format: set display number format
            possible values: hex, dec
    :param int components: set components to display
    """
    graph.append(
        Filter(
            command="datascope",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="mode", value=mode),
                FilterOption(name="axis", value=axis),
                FilterOption(name="opacity", value=opacity),
                FilterOption(name="format", value=format),
                FilterOption(name="components", value=components),
            ],
        )
    )
    return graph


def pixscope(
    graph: Stream,
    x: Optional[float] = None,
    y: Optional[float] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    o: Optional[float] = None,
    wx: Optional[float] = None,
    wy: Optional[float] = None,
):
    """Pixel data analysis.

    :param float x: set scope x offset
    :param float y: set scope y offset
    :param int w: set scope width
    :param int h: set scope height
    :param float o: set window opacity
    :param float wx: set window x offset
    :param float wy: set window y offset
    """
    graph.append(
        Filter(
            command="pixscope",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="o", value=o),
                FilterOption(name="wx", value=wx),
                FilterOption(name="wy", value=wy),
            ],
        )
    )
    return graph


def oscilloscope(
    graph: Stream,
    x: Optional[float] = None,
    y: Optional[float] = None,
    s: Optional[float] = None,
    t: Optional[float] = None,
    o: Optional[float] = None,
    tx: Optional[float] = None,
    ty: Optional[float] = None,
    tw: Optional[float] = None,
    th: Optional[float] = None,
    c: Optional[int] = None,
    g: Optional[bool] = None,
    st: Optional[bool] = None,
    sc: Optional[bool] = None,
):
    """2D Video Oscilloscope.

    :param float x: set scope x position
    :param float y: set scope y position
    :param float s: set scope size
    :param float t: set scope tilt
    :param float o: set trace opacity
    :param float tx: set trace x position
    :param float ty: set trace y position
    :param float tw: set trace width
    :param float th: set trace height
    :param int c: set components to trace
    :param bool g: draw trace grid
    :param bool st: draw statistics
    :param bool sc: draw scope
    """
    graph.append(
        Filter(
            command="oscilloscope",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="s", value=s),
                FilterOption(name="t", value=t),
                FilterOption(name="o", value=o),
                FilterOption(name="tx", value=tx),
                FilterOption(name="ty", value=ty),
                FilterOption(name="tw", value=tw),
                FilterOption(name="th", value=th),
                FilterOption(name="c", value=c),
                FilterOption(name="g", value=g),
                FilterOption(name="st", value=st),
                FilterOption(name="sc", value=sc),
            ],
        )
    )
    return graph


def guided(
    graph: Stream,
    radius: Optional[int] = None,
    eps: Optional[float] = None,
    mode: Optional[str] = None,
    sub: Optional[int] = None,
    guidance: Optional[str] = None,
    planes: Optional[int] = None,
):
    """Apply Guided filter.

    :param int radius: set the box radius
    :param float eps: set the regularization parameter (with square)
    :param str mode: set filtering mode (0: basic mode; 1: fast mode)
            possible values: basic, fast
    :param int sub: subsampling ratio for fast mode
    :param str guidance: set guidance mode (0: off mode; 1: on mode)
            possible values: off, on
    :param int planes: set planes to filter
    """
    graph.append(
        Filter(
            command="guided",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="radius", value=radius),
                FilterOption(name="eps", value=eps),
                FilterOption(name="mode", value=mode),
                FilterOption(name="sub", value=sub),
                FilterOption(name="guidance", value=guidance),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def ahistogram(
    graph: Stream,
    dmode: Optional[str] = None,
    rate: Optional[str] = None,
    size: Optional[int] = None,
    scale: Optional[str] = None,
    ascale: Optional[str] = None,
    acount: Optional[int] = None,
    rheight: Optional[float] = None,
    slide: Optional[str] = None,
    hmode: Optional[str] = None,
):
    """Convert input audio to histogram video output.

    :param str dmode: set method to display channels
            possible values: single, separate
    :param str rate: set video rate
    :param int size: set video size
    :param str scale: set display scale
            possible values: log, sqrt, cbrt, lin, rlog
    :param str ascale: set amplitude scale
            possible values: log, lin
    :param int acount: how much frames to accumulate
    :param float rheight: set histogram ratio of window height
    :param str slide: set sonogram sliding
            possible values: replace, scroll
    :param str hmode: set histograms mode
            possible values: abs, sign
    """
    graph.append(
        Filter(
            command="ahistogram",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="dmode", value=dmode),
                FilterOption(name="rate", value=rate),
                FilterOption(name="size", value=size),
                FilterOption(name="scale", value=scale),
                FilterOption(name="ascale", value=ascale),
                FilterOption(name="acount", value=acount),
                FilterOption(name="rheight", value=rheight),
                FilterOption(name="slide", value=slide),
                FilterOption(name="hmode", value=hmode),
            ],
        )
    )
    return graph


def qp(graph: Stream, qp: Optional[str] = None):
    """Change video quantization parameters.

    :param str qp: set qp expression
    """
    graph.append(Filter(command="qp", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="qp", value=qp)]))
    return graph


def geq(
    graph: Stream,
    lum_expr: Optional[str] = None,
    lum: Optional[str] = None,
    cb_expr: Optional[str] = None,
    cb: Optional[str] = None,
    cr_expr: Optional[str] = None,
    cr: Optional[str] = None,
    alpha_expr: Optional[str] = None,
    a: Optional[str] = None,
    red_expr: Optional[str] = None,
    r: Optional[str] = None,
    green_expr: Optional[str] = None,
    g: Optional[str] = None,
    blue_expr: Optional[str] = None,
    b: Optional[str] = None,
    interpolation: Optional[str] = None,
):
    """Apply generic equation to each pixel.

    :param str lum_expr: set luminance expression
    :param str lum: set luminance expression
    :param str cb_expr: set chroma blue expression
    :param str cb: set chroma blue expression
    :param str cr_expr: set chroma red expression
    :param str cr: set chroma red expression
    :param str alpha_expr: set alpha expression
    :param str a: set alpha expression
    :param str red_expr: set red expression
    :param str r: set red expression
    :param str green_expr: set green expression
    :param str g: set green expression
    :param str blue_expr: set blue expression
    :param str b: set blue expression
    :param str interpolation: set interpolation method
            possible values: nearest, n, bilinear, b
    """
    graph.append(
        Filter(
            command="geq",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="lum_expr", value=lum_expr),
                FilterOption(name="lum", value=lum),
                FilterOption(name="cb_expr", value=cb_expr),
                FilterOption(name="cb", value=cb),
                FilterOption(name="cr_expr", value=cr_expr),
                FilterOption(name="cr", value=cr),
                FilterOption(name="alpha_expr", value=alpha_expr),
                FilterOption(name="a", value=a),
                FilterOption(name="red_expr", value=red_expr),
                FilterOption(name="r", value=r),
                FilterOption(name="green_expr", value=green_expr),
                FilterOption(name="g", value=g),
                FilterOption(name="blue_expr", value=blue_expr),
                FilterOption(name="b", value=b),
                FilterOption(name="interpolation", value=interpolation),
            ],
        )
    )
    return graph


def curves(
    graph: Stream,
    preset: Optional[str] = None,
    master: Optional[str] = None,
    m: Optional[str] = None,
    red: Optional[str] = None,
    r: Optional[str] = None,
    green: Optional[str] = None,
    g: Optional[str] = None,
    blue: Optional[str] = None,
    b: Optional[str] = None,
    all: Optional[str] = None,
    psfile: Optional[str] = None,
    plot: Optional[str] = None,
    interp: Optional[str] = None,
):
    """Adjust components curves.

    :param str preset: select a color curves preset
            possible values: none, color_negative, cross_process, darker, increase_contrast, lighter, linear_contrast, medium_contrast, negative, strong_contrast, vintage
    :param str master: set master points coordinates
    :param str m: set master points coordinates
    :param str red: set red points coordinates
    :param str r: set red points coordinates
    :param str green: set green points coordinates
    :param str g: set green points coordinates
    :param str blue: set blue points coordinates
    :param str b: set blue points coordinates
    :param str all: set points coordinates for all components
    :param str psfile: set Photoshop curves file name
    :param str plot: save Gnuplot script of the curves in specified file
    :param str interp: specify the kind of interpolation
            possible values: natural, pchip
    """
    graph.append(
        Filter(
            command="curves",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="preset", value=preset),
                FilterOption(name="master", value=master),
                FilterOption(name="m", value=m),
                FilterOption(name="red", value=red),
                FilterOption(name="r", value=r),
                FilterOption(name="green", value=green),
                FilterOption(name="g", value=g),
                FilterOption(name="blue", value=blue),
                FilterOption(name="b", value=b),
                FilterOption(name="all", value=all),
                FilterOption(name="psfile", value=psfile),
                FilterOption(name="plot", value=plot),
                FilterOption(name="interp", value=interp),
            ],
        )
    )
    return graph


def mcdeint(graph: Stream, mode: Optional[str] = None, parity: Optional[str] = None, qp: Optional[int] = None):
    """Apply motion compensating deinterlacing.

    :param str mode: set mode
            possible values: fast, medium, slow, extra_slow
    :param str parity: set the assumed picture field parity
            possible values: tff, bff
    :param int qp: set qp
    """
    graph.append(
        Filter(
            command="mcdeint",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="mode", value=mode), FilterOption(name="parity", value=parity), FilterOption(name="qp", value=qp)],
        )
    )
    return graph


def freezedetect(graph: Stream, n: Optional[float] = None, d: Optional[int] = None):
    """Detects frozen video input.

    :param float n: set noise tolerance
    :param int d: set minimum duration in seconds
    """
    graph.append(
        Filter(
            command="freezedetect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="n", value=n), FilterOption(name="d", value=d)],
        )
    )
    return graph


def fps(
    graph: Stream,
    fps: Optional[str] = None,
    start_time: Optional[float] = None,
    round: Optional[str] = None,
    eof_action: Optional[str] = None,
):
    """Force constant framerate.

    :param str fps: A string describing desired output framerate
    :param float start_time: Assume the first PTS should be this value.
    :param str round: set rounding method for timestamps
            possible values: zero, inf, down, up, near
    :param str eof_action: action performed for last frame
            possible values: round, pass
    """
    graph.append(
        Filter(
            command="fps",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="fps", value=fps),
                FilterOption(name="start_time", value=start_time),
                FilterOption(name="round", value=round),
                FilterOption(name="eof_action", value=eof_action),
            ],
        )
    )
    return graph


def chromaber_vulkan(graph: Stream, dist_x: Optional[float] = None, dist_y: Optional[float] = None):
    """Offset chroma of input video (chromatic aberration)

    :param float dist_x: Set horizontal distortion amount
    :param float dist_y: Set vertical distortion amount
    """
    graph.append(
        Filter(
            command="chromaber_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="dist_x", value=dist_x), FilterOption(name="dist_y", value=dist_y)],
        )
    )
    return graph


def reverse(
    graph: Stream,
):
    """Reverse a clip."""
    graph.append(Filter(command="reverse", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def areverse(
    graph: Stream,
):
    """Reverse an audio clip."""
    graph.append(Filter(command="areverse", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def tpad(
    graph: Stream,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    start_mode: Optional[str] = None,
    stop_mode: Optional[int] = None,
    start_duration: Optional[int] = None,
    stop_duration: Optional[int] = None,
    color: Optional[str] = None,
):
    """Temporarily pad video frames.

    :param int start: set the number of frames to delay input
    :param int stop: set the number of frames to add after input finished
    :param str start_mode: set the mode of added frames to start
            possible values: add, clone
    :param int stop_mode: set the mode of added frames to end
    :param int start_duration: set the duration to delay input
    :param int stop_duration: set the duration to pad input
    :param str color: set the color of the added frames
    """
    graph.append(
        Filter(
            command="tpad",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="start", value=start),
                FilterOption(name="stop", value=stop),
                FilterOption(name="start_mode", value=start_mode),
                FilterOption(name="stop_mode", value=stop_mode),
                FilterOption(name="start_duration", value=start_duration),
                FilterOption(name="stop_duration", value=stop_duration),
                FilterOption(name="color", value=color),
            ],
        )
    )
    return graph


def fftfilt(
    graph: Stream,
    dc_Y: Optional[int] = None,
    dc_U: Optional[int] = None,
    dc_V: Optional[int] = None,
    weight_Y: Optional[str] = None,
    weight_U: Optional[str] = None,
    weight_V: Optional[str] = None,
    eval: Optional[str] = None,
):
    """Apply arbitrary expressions to pixels in frequency domain.

    :param int dc_Y: adjust gain in Y plane
    :param int dc_U: adjust gain in U plane
    :param int dc_V: adjust gain in V plane
    :param str weight_Y: set luminance expression in Y plane
    :param str weight_U: set chrominance expression in U plane
    :param str weight_V: set chrominance expression in V plane
    :param str eval: specify when to evaluate expressions
            possible values: init, frame
    """
    graph.append(
        Filter(
            command="fftfilt",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="dc_Y", value=dc_Y),
                FilterOption(name="dc_U", value=dc_U),
                FilterOption(name="dc_V", value=dc_V),
                FilterOption(name="weight_Y", value=weight_Y),
                FilterOption(name="weight_U", value=weight_U),
                FilterOption(name="weight_V", value=weight_V),
                FilterOption(name="eval", value=eval),
            ],
        )
    )
    return graph


def compensationdelay(
    graph: Stream,
    mm: Optional[int] = None,
    cm: Optional[int] = None,
    m: Optional[int] = None,
    dry: Optional[float] = None,
    wet: Optional[float] = None,
    temp: Optional[int] = None,
):
    """Audio Compensation Delay Line.

    :param int mm: set mm distance
    :param int cm: set cm distance
    :param int m: set meter distance
    :param float dry: set dry amount
    :param float wet: set wet amount
    :param int temp: set temperature °C
    """
    graph.append(
        Filter(
            command="compensationdelay",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="mm", value=mm),
                FilterOption(name="cm", value=cm),
                FilterOption(name="m", value=m),
                FilterOption(name="dry", value=dry),
                FilterOption(name="wet", value=wet),
                FilterOption(name="temp", value=temp),
            ],
        )
    )
    return graph


def arls(
    graph: Stream,
    order: Optional[int] = None,
    _lambda: Optional[float] = None,
    delta: Optional[float] = None,
    out_mode: Optional[str] = None,
    precision: Optional[str] = None,
):
    """Apply Recursive Least Squares algorithm to first audio stream.

    :param int order: set the filter order
    :param float lambda: set the filter lambda
    :param float delta: set the filter delta
    :param str out_mode: set output mode
            possible values: i, d, o, n, e
    :param str precision: set processing precision
            possible values: auto, float, double
    """
    graph.append(
        Filter(
            command="arls",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="order", value=order),
                FilterOption(name="lambda", value=_lambda),
                FilterOption(name="delta", value=delta),
                FilterOption(name="out_mode", value=out_mode),
                FilterOption(name="precision", value=precision),
            ],
        )
    )
    return graph


def entropy(graph: Stream, mode: Optional[str] = None):
    """Measure video frames entropy.

    :param str mode: set kind of histogram entropy measurement
            possible values: normal, diff
    """
    graph.append(Filter(command="entropy", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="mode", value=mode)]))
    return graph


def ccrepack(
    graph: Stream,
):
    """Repack CEA-708 closed caption metadata"""
    graph.append(Filter(command="ccrepack", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def mpdecimate(
    graph: Stream,
    max: Optional[int] = None,
    keep: Optional[int] = None,
    hi: Optional[int] = None,
    lo: Optional[int] = None,
    frac: Optional[float] = None,
):
    """Remove near-duplicate frames.

    :param int max: set the maximum number of consecutive dropped frames (positive), or the minimum interval between dropped frames (negative)
    :param int keep: set the number of similar consecutive frames to be kept before starting to drop similar frames
    :param int hi: set high dropping threshold
    :param int lo: set low dropping threshold
    :param float frac: set fraction dropping threshold
    """
    graph.append(
        Filter(
            command="mpdecimate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="max", value=max),
                FilterOption(name="keep", value=keep),
                FilterOption(name="hi", value=hi),
                FilterOption(name="lo", value=lo),
                FilterOption(name="frac", value=frac),
            ],
        )
    )
    return graph


def avgblur(graph: Stream, sizeX: Optional[int] = None, planes: Optional[int] = None, sizeY: Optional[int] = None):
    """Apply Average Blur filter.

    :param int sizeX: set horizontal size
    :param int planes: set planes to filter
    :param int sizeY: set vertical size
    """
    graph.append(
        Filter(
            command="avgblur",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="sizeX", value=sizeX),
                FilterOption(name="planes", value=planes),
                FilterOption(name="sizeY", value=sizeY),
            ],
        )
    )
    return graph


def transpose(graph: Stream, dir: Optional[str] = None, passthrough: Optional[str] = None):
    """Transpose input video.

    :param str dir: set transpose direction
            possible values: cclock_flip, clock, cclock, clock_flip
    :param str passthrough: do not apply transposition if the input matches the specified geometry
            possible values: none, portrait, landscape
    """
    graph.append(
        Filter(
            command="transpose",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="dir", value=dir), FilterOption(name="passthrough", value=passthrough)],
        )
    )
    return graph


def transpose_vaapi(graph: Stream, dir: Optional[str] = None, passthrough: Optional[str] = None):
    """VAAPI VPP for transpose

    :param str dir: set transpose direction
            possible values: cclock_flip, clock, cclock, clock_flip, reversal, hflip, vflip
    :param str passthrough: do not apply transposition if the input matches the specified geometry
            possible values: none, portrait, landscape
    """
    graph.append(
        Filter(
            command="transpose_vaapi",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="dir", value=dir), FilterOption(name="passthrough", value=passthrough)],
        )
    )
    return graph


def virtualbass(graph: Stream, cutoff: Optional[float] = None, strength: Optional[float] = None):
    """Audio Virtual Bass.

    :param float cutoff: set virtual bass cutoff
    :param float strength: set virtual bass strength
    """
    graph.append(
        Filter(
            command="virtualbass",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="cutoff", value=cutoff), FilterOption(name="strength", value=strength)],
        )
    )
    return graph


def hflip(
    graph: Stream,
):
    """Horizontally flip the input video."""
    graph.append(Filter(command="hflip", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def afir(
    graph: Stream,
    dry: Optional[float] = None,
    wet: Optional[float] = None,
    length: Optional[float] = None,
    gtype: Optional[str] = None,
    irnorm: Optional[float] = None,
    irlink: Optional[bool] = None,
    irgain: Optional[float] = None,
    irfmt: Optional[str] = None,
    maxir: Optional[float] = None,
    response: Optional[bool] = None,
    channel: Optional[int] = None,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    minp: Optional[int] = None,
    maxp: Optional[int] = None,
    nbirs: Optional[int] = None,
    ir: Optional[int] = None,
    precision: Optional[str] = None,
    irload: Optional[str] = None,
):
    """Apply Finite Impulse Response filter with supplied coefficients in additional stream(s).

    :param float dry: set dry gain
    :param float wet: set wet gain
    :param float length: set IR length
    :param str gtype: set IR auto gain type
            possible values: none, peak, dc, gn, ac, rms
    :param float irnorm: set IR norm
    :param bool irlink: set IR link
    :param float irgain: set IR gain
    :param str irfmt: set IR format
            possible values: mono, input
    :param float maxir: set max IR length
    :param bool response: show IR frequency response
    :param int channel: set IR channel to display frequency response
    :param int size: set video size
    :param str rate: set video rate
    :param int minp: set min partition size
    :param int maxp: set max partition size
    :param int nbirs: set number of input IRs
    :param int ir: select IR
    :param str precision: set processing precision
            possible values: auto, float, double
    :param str irload: set IR loading type
            possible values: init, access
    """
    graph.append(
        Filter(
            command="afir",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="dry", value=dry),
                FilterOption(name="wet", value=wet),
                FilterOption(name="length", value=length),
                FilterOption(name="gtype", value=gtype),
                FilterOption(name="irnorm", value=irnorm),
                FilterOption(name="irlink", value=irlink),
                FilterOption(name="irgain", value=irgain),
                FilterOption(name="irfmt", value=irfmt),
                FilterOption(name="maxir", value=maxir),
                FilterOption(name="response", value=response),
                FilterOption(name="channel", value=channel),
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="minp", value=minp),
                FilterOption(name="maxp", value=maxp),
                FilterOption(name="nbirs", value=nbirs),
                FilterOption(name="ir", value=ir),
                FilterOption(name="precision", value=precision),
                FilterOption(name="irload", value=irload),
            ],
        )
    )
    return graph


def waveform(
    graph: Stream,
    mode: Optional[str] = None,
    intensity: Optional[float] = None,
    mirror: Optional[bool] = None,
    display: Optional[str] = None,
    components: Optional[int] = None,
    envelope: Optional[str] = None,
    filter: Optional[str] = None,
    graticule: Optional[str] = None,
    opacity: Optional[float] = None,
    flags: Optional[str] = None,
    scale: Optional[str] = None,
    bgopacity: Optional[float] = None,
    tint0: Optional[float] = None,
    t0: Optional[float] = None,
    tint1: Optional[float] = None,
    t1: Optional[float] = None,
    fitmode: Optional[str] = None,
    input: Optional[str] = None,
):
    """Video waveform monitor.

    :param str mode: set mode
            possible values: row, column
    :param float intensity: set intensity
    :param bool mirror: set mirroring
    :param str display: set display mode
            possible values: overlay, stack, parade
    :param int components: set components to display
    :param str envelope: set envelope to display
            possible values: none, instant, peak, peak+instant
    :param str filter: set filter
            possible values: lowpass, flat, aflat, chroma, color, acolor, xflat, yflat
    :param str graticule: set graticule
            possible values: none, green, orange, invert
    :param float opacity: set graticule opacity
    :param str flags: set graticule flags
            possible values: numbers, dots
    :param str scale: set scale
            possible values: digital, millivolts, ire
    :param float bgopacity: set background opacity
    :param float tint0: set 1st tint
    :param float t0: set 1st tint
    :param float tint1: set 2nd tint
    :param float t1: set 2nd tint
    :param str fitmode: set fit mode
            possible values: none, size
    :param str input: set input formats selection
            possible values: all, first
    """
    graph.append(
        Filter(
            command="waveform",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="intensity", value=intensity),
                FilterOption(name="mirror", value=mirror),
                FilterOption(name="display", value=display),
                FilterOption(name="components", value=components),
                FilterOption(name="envelope", value=envelope),
                FilterOption(name="filter", value=filter),
                FilterOption(name="graticule", value=graticule),
                FilterOption(name="opacity", value=opacity),
                FilterOption(name="flags", value=flags),
                FilterOption(name="scale", value=scale),
                FilterOption(name="bgopacity", value=bgopacity),
                FilterOption(name="tint0", value=tint0),
                FilterOption(name="t0", value=t0),
                FilterOption(name="tint1", value=tint1),
                FilterOption(name="t1", value=t1),
                FilterOption(name="fitmode", value=fitmode),
                FilterOption(name="input", value=input),
            ],
        )
    )
    return graph


def silencedetect(graph: Stream, n: Optional[float] = None, d: Optional[int] = None, mono: Optional[bool] = None):
    """Detect silence.

    :param float n: set noise tolerance
    :param int d: set minimum duration in seconds
    :param bool mono: check each channel separately
    """
    graph.append(
        Filter(
            command="silencedetect",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="n", value=n), FilterOption(name="d", value=d), FilterOption(name="mono", value=mono)],
        )
    )
    return graph


def anullsrc(
    graph: Stream,
    channel_layout: Optional[str] = None,
    sample_rate: Optional[int] = None,
    nb_samples: Optional[int] = None,
    duration: Optional[int] = None,
):
    """Null audio source, return empty audio frames.

    :param str channel_layout: set channel_layout
    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param int duration: set the audio duration
    """
    graph.append(
        Filter(
            command="anullsrc",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="channel_layout", value=channel_layout),
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="duration", value=duration),
            ],
        )
    )
    return graph


def afreqshift(graph: Stream, shift: Optional[float] = None, level: Optional[float] = None, order: Optional[int] = None):
    """Apply frequency shifting to input audio.

    :param float shift: set frequency shift
    :param float level: set output level
    :param int order: set filter order
    """
    graph.append(
        Filter(
            command="afreqshift",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="shift", value=shift),
                FilterOption(name="level", value=level),
                FilterOption(name="order", value=order),
            ],
        )
    )
    return graph


def aphaseshift(graph: Stream, shift: Optional[float] = None, level: Optional[float] = None, order: Optional[int] = None):
    """Apply phase shifting to input audio.

    :param float shift: set phase shift
    :param float level: set output level
    :param int order: set filter order
    """
    graph.append(
        Filter(
            command="aphaseshift",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="shift", value=shift),
                FilterOption(name="level", value=level),
                FilterOption(name="order", value=order),
            ],
        )
    )
    return graph


def colormatrix(graph: Stream, src: Optional[str] = None, dst: Optional[str] = None):
    """Convert color matrix.

    :param str src: set source color matrix
            possible values: bt709, fcc, bt601, bt470, bt470bg, smpte170m, smpte240m, bt2020
    :param str dst: set destination color matrix
            possible values: bt709, fcc, bt601, bt470, bt470bg, smpte170m, smpte240m, bt2020
    """
    graph.append(
        Filter(
            command="colormatrix",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="src", value=src), FilterOption(name="dst", value=dst)],
        )
    )
    return graph


def pad_vaapi(
    graph: Stream,
    width: Optional[str] = None,
    height: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    aspect: Optional[int] = None,
):
    """Pad the input video.

    :param str width: set the pad area width
    :param str height: set the pad area height
    :param str x: set the x offset for the input image position
    :param str y: set the y offset for the input image position
    :param str color: set the color of the padded area border
    :param int aspect: pad to fit an aspect instead of a resolution
    """
    graph.append(
        Filter(
            command="pad_vaapi",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="width", value=width),
                FilterOption(name="height", value=height),
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="color", value=color),
                FilterOption(name="aspect", value=aspect),
            ],
        )
    )
    return graph


def vectorscope(
    graph: Stream,
    mode: Optional[str] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    intensity: Optional[float] = None,
    envelope: Optional[str] = None,
    graticule: Optional[str] = None,
    opacity: Optional[float] = None,
    flags: Optional[str] = None,
    bgopacity: Optional[float] = None,
    lthreshold: Optional[float] = None,
    hthreshold: Optional[float] = None,
    colorspace: Optional[str] = None,
    tint0: Optional[float] = None,
    t0: Optional[float] = None,
    tint1: Optional[float] = None,
    t1: Optional[float] = None,
):
    """Video vectorscope.

    :param str mode: set vectorscope mode
            possible values: gray, tint, color, color2, color3, color4, color5
    :param int x: set color component on X axis
    :param int y: set color component on Y axis
    :param float intensity: set intensity
    :param str envelope: set envelope
            possible values: none, instant, peak, peak+instant
    :param str graticule: set graticule
            possible values: none, green, color, invert
    :param float opacity: set graticule opacity
    :param str flags: set graticule flags
            possible values: white, black, name
    :param float bgopacity: set background opacity
    :param float lthreshold: set low threshold
    :param float hthreshold: set high threshold
    :param str colorspace: set colorspace
            possible values: auto, 601, 709
    :param float tint0: set 1st tint
    :param float t0: set 1st tint
    :param float tint1: set 2nd tint
    :param float t1: set 2nd tint
    """
    graph.append(
        Filter(
            command="vectorscope",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="intensity", value=intensity),
                FilterOption(name="envelope", value=envelope),
                FilterOption(name="graticule", value=graticule),
                FilterOption(name="opacity", value=opacity),
                FilterOption(name="flags", value=flags),
                FilterOption(name="bgopacity", value=bgopacity),
                FilterOption(name="lthreshold", value=lthreshold),
                FilterOption(name="hthreshold", value=hthreshold),
                FilterOption(name="colorspace", value=colorspace),
                FilterOption(name="tint0", value=tint0),
                FilterOption(name="t0", value=t0),
                FilterOption(name="tint1", value=tint1),
                FilterOption(name="t1", value=t1),
            ],
        )
    )
    return graph


def showvolume(
    graph: Stream,
    rate: Optional[str] = None,
    b: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    f: Optional[float] = None,
    c: Optional[str] = None,
    t: Optional[bool] = None,
    v: Optional[bool] = None,
    dm: Optional[float] = None,
    dmc: Optional[str] = None,
    o: Optional[str] = None,
    s: Optional[int] = None,
    p: Optional[float] = None,
    m: Optional[str] = None,
    ds: Optional[str] = None,
):
    """Convert input audio volume to video output.

    :param str rate: set video rate
    :param int b: set border width
    :param int w: set channel width
    :param int h: set channel height
    :param float f: set fade
    :param str c: set volume color expression
    :param bool t: display channel names
    :param bool v: display volume value
    :param float dm: duration for max value display
    :param str dmc: set color of the max value line
    :param str o: set orientation
            possible values: h, v
    :param int s: set step size
    :param float p: set background opacity
    :param str m: set mode
            possible values: p, r
    :param str ds: set display scale
            possible values: lin, log
    """
    graph.append(
        Filter(
            command="showvolume",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rate", value=rate),
                FilterOption(name="b", value=b),
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="f", value=f),
                FilterOption(name="c", value=c),
                FilterOption(name="t", value=t),
                FilterOption(name="v", value=v),
                FilterOption(name="dm", value=dm),
                FilterOption(name="dmc", value=dmc),
                FilterOption(name="o", value=o),
                FilterOption(name="s", value=s),
                FilterOption(name="p", value=p),
                FilterOption(name="m", value=m),
                FilterOption(name="ds", value=ds),
            ],
        )
    )
    return graph


def colorize(
    graph: Stream,
    hue: Optional[float] = None,
    saturation: Optional[float] = None,
    lightness: Optional[float] = None,
    mix: Optional[float] = None,
):
    """Overlay a solid color on the video stream.

    :param float hue: set the hue
    :param float saturation: set the saturation
    :param float lightness: set the lightness
    :param float mix: set the mix of source lightness
    """
    graph.append(
        Filter(
            command="colorize",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="hue", value=hue),
                FilterOption(name="saturation", value=saturation),
                FilterOption(name="lightness", value=lightness),
                FilterOption(name="mix", value=mix),
            ],
        )
    )
    return graph


def drawgraph(
    graph: Stream,
    m1: Optional[str] = None,
    fg1: Optional[str] = None,
    m2: Optional[str] = None,
    fg2: Optional[str] = None,
    m3: Optional[str] = None,
    fg3: Optional[str] = None,
    m4: Optional[str] = None,
    fg4: Optional[str] = None,
    bg: Optional[str] = None,
    min: Optional[float] = None,
    max: Optional[float] = None,
    mode: Optional[str] = None,
    slide: Optional[str] = None,
    size: Optional[int] = None,
    rate: Optional[str] = None,
):
    """Draw a graph using input video metadata.

    :param str m1: set 1st metadata key
    :param str fg1: set 1st foreground color expression
    :param str m2: set 2nd metadata key
    :param str fg2: set 2nd foreground color expression
    :param str m3: set 3rd metadata key
    :param str fg3: set 3rd foreground color expression
    :param str m4: set 4th metadata key
    :param str fg4: set 4th foreground color expression
    :param str bg: set background color
    :param float min: set minimal value
    :param float max: set maximal value
    :param str mode: set graph mode
            possible values: bar, dot, line
    :param str slide: set slide mode
            possible values: frame, replace, scroll, rscroll, picture
    :param int size: set graph size
    :param str rate: set video rate
    """
    graph.append(
        Filter(
            command="drawgraph",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="m1", value=m1),
                FilterOption(name="fg1", value=fg1),
                FilterOption(name="m2", value=m2),
                FilterOption(name="fg2", value=fg2),
                FilterOption(name="m3", value=m3),
                FilterOption(name="fg3", value=fg3),
                FilterOption(name="m4", value=m4),
                FilterOption(name="fg4", value=fg4),
                FilterOption(name="bg", value=bg),
                FilterOption(name="min", value=min),
                FilterOption(name="max", value=max),
                FilterOption(name="mode", value=mode),
                FilterOption(name="slide", value=slide),
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
            ],
        )
    )
    return graph


def adrawgraph(
    graph: Stream,
):
    """Draw a graph using input audio metadata."""
    graph.append(Filter(command="adrawgraph", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def convolve(graph: Stream, planes: Optional[int] = None, impulse: Optional[str] = None, noise: Optional[float] = None):
    """Convolve first video stream with second video stream.

    :param int planes: set planes to convolve
    :param str impulse: when to process impulses
            possible values: first, all
    :param float noise: set noise
    """
    graph.append(
        Filter(
            command="convolve",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="planes", value=planes),
                FilterOption(name="impulse", value=impulse),
                FilterOption(name="noise", value=noise),
            ],
        )
    )
    return graph


def deconvolve(graph: Stream, planes: Optional[int] = None, impulse: Optional[str] = None, noise: Optional[float] = None):
    """Deconvolve first video stream with second video stream.

    :param int planes: set planes to deconvolve
    :param str impulse: when to process impulses
            possible values: first, all
    :param float noise: set noise
    """
    graph.append(
        Filter(
            command="deconvolve",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="planes", value=planes),
                FilterOption(name="impulse", value=impulse),
                FilterOption(name="noise", value=noise),
            ],
        )
    )
    return graph


def xcorrelate(graph: Stream, planes: Optional[int] = None, secondary: Optional[str] = None):
    """Cross-correlate first video stream with second video stream.

    :param int planes: set planes to cross-correlate
    :param str secondary: when to process secondary frame
            possible values: first, all
    """
    graph.append(
        Filter(
            command="xcorrelate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="planes", value=planes), FilterOption(name="secondary", value=secondary)],
        )
    )
    return graph


def gblur(
    graph: Stream, sigma: Optional[float] = None, steps: Optional[int] = None, planes: Optional[int] = None, sigmaV: Optional[float] = None
):
    """Apply Gaussian Blur filter.

    :param float sigma: set sigma
    :param int steps: set number of steps
    :param int planes: set planes to filter
    :param float sigmaV: set vertical sigma
    """
    graph.append(
        Filter(
            command="gblur",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="sigma", value=sigma),
                FilterOption(name="steps", value=steps),
                FilterOption(name="planes", value=planes),
                FilterOption(name="sigmaV", value=sigmaV),
            ],
        )
    )
    return graph


def colorspace(
    graph: Stream,
    all: Optional[str] = None,
    space: Optional[str] = None,
    range: Optional[str] = None,
    primaries: Optional[str] = None,
    trc: Optional[str] = None,
    format: Optional[str] = None,
    fast: Optional[bool] = None,
    dither: Optional[str] = None,
    wpadapt: Optional[str] = None,
    iall: Optional[int] = None,
    ispace: Optional[int] = None,
    irange: Optional[int] = None,
    iprimaries: Optional[int] = None,
    itrc: Optional[int] = None,
):
    """Convert between colorspaces.

    :param str all: Set all color properties together
            possible values: bt470m, bt470bg, bt601-6-525, bt601-6-625, bt709, smpte170m, smpte240m, bt2020
    :param str space: Output colorspace
            possible values: bt709, fcc, bt470bg, smpte170m, smpte240m, ycgco, gbr, bt2020nc, bt2020ncl
    :param str range: Output color range
            possible values: tv, mpeg, pc, jpeg
    :param str primaries: Output color primaries
            possible values: bt709, bt470m, bt470bg, smpte170m, smpte240m, smpte428, film, smpte431, smpte432, bt2020, jedec-p22, ebu3213
    :param str trc: Output transfer characteristics
            possible values: bt709, bt470m, gamma22, bt470bg, gamma28, smpte170m, smpte240m, linear, srgb, iec61966-2-1, xvycc, iec61966-2-4, bt2020-10, bt2020-12
    :param str format: Output pixel format
            possible values: yuv420p, yuv420p10, yuv420p12, yuv422p, yuv422p10, yuv422p12, yuv444p, yuv444p10, yuv444p12
    :param bool fast: Ignore primary chromaticity and gamma correction
    :param str dither: Dithering mode
            possible values: none, fsb
    :param str wpadapt: Whitepoint adaptation method
            possible values: bradford, vonkries, identity
    :param int iall: Set all input color properties together
    :param int ispace: Input colorspace
    :param int irange: Input color range
    :param int iprimaries: Input color primaries
    :param int itrc: Input transfer characteristics
    """
    graph.append(
        Filter(
            command="colorspace",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="all", value=all),
                FilterOption(name="space", value=space),
                FilterOption(name="range", value=range),
                FilterOption(name="primaries", value=primaries),
                FilterOption(name="trc", value=trc),
                FilterOption(name="format", value=format),
                FilterOption(name="fast", value=fast),
                FilterOption(name="dither", value=dither),
                FilterOption(name="wpadapt", value=wpadapt),
                FilterOption(name="iall", value=iall),
                FilterOption(name="ispace", value=ispace),
                FilterOption(name="irange", value=irange),
                FilterOption(name="iprimaries", value=iprimaries),
                FilterOption(name="itrc", value=itrc),
            ],
        )
    )
    return graph


def shear(
    graph: Stream, shx: Optional[float] = None, shy: Optional[float] = None, fillcolor: Optional[str] = None, interp: Optional[str] = None
):
    """Shear transform the input image.

    :param float shx: set x shear factor
    :param float shy: set y shear factor
    :param str fillcolor: set background fill color
    :param str interp: set interpolation
            possible values: nearest, bilinear
    """
    graph.append(
        Filter(
            command="shear",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="shx", value=shx),
                FilterOption(name="shy", value=shy),
                FilterOption(name="fillcolor", value=fillcolor),
                FilterOption(name="interp", value=interp),
            ],
        )
    )
    return graph


def cue(
    graph: Stream,
):
    """Delay filtering to match a cue."""
    graph.append(Filter(command="cue", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def acue(
    graph: Stream,
):
    """Delay filtering to match a cue."""
    graph.append(Filter(command="acue", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def photosensitivity(
    graph: Stream,
    frames: Optional[int] = None,
    threshold: Optional[float] = None,
    skip: Optional[int] = None,
    bypass: Optional[bool] = None,
):
    """Filter out photosensitive epilepsy seizure-inducing flashes.

    :param int frames: set how many frames to use
    :param float threshold: set detection threshold factor (lower is stricter)
    :param int skip: set pixels to skip when sampling frames
    :param bool bypass: leave frames unchanged
    """
    graph.append(
        Filter(
            command="photosensitivity",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="frames", value=frames),
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="skip", value=skip),
                FilterOption(name="bypass", value=bypass),
            ],
        )
    )
    return graph


def drmeter(graph: Stream, length: Optional[float] = None):
    """Measure audio dynamic range.

    :param float length: set the window length
    """
    graph.append(Filter(command="drmeter", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="length", value=length)]))
    return graph


def astats(
    graph: Stream,
    length: Optional[float] = None,
    metadata: Optional[bool] = None,
    reset: Optional[int] = None,
    measure_perchannel: Optional[str] = None,
    measure_overall: Optional[str] = None,
):
    """Show time domain statistics about audio frames.

    :param float length: set the window length
    :param bool metadata: inject metadata in the filtergraph
    :param int reset: Set the number of frames over which cumulative stats are calculated before being reset
    :param str measure_perchannel: Select the parameters which are measured per channel
            possible values: none, all, Bit_depth, Crest_factor, DC_offset, Dynamic_range, Entropy, Flat_factor, Max_difference, Max_level, Mean_difference, Min_difference, Min_level, Noise_floor, Noise_floor_count, Number_of_Infs, Number_of_NaNs, Number_of_denormals, Number_of_samples, Peak_count, Peak_level, RMS_difference, RMS_level, RMS_peak, RMS_trough, Zero_crossings, Zero_crossings_rate, Abs_Peak_count
    :param str measure_overall: Select the parameters which are measured overall
    """
    graph.append(
        Filter(
            command="astats",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="length", value=length),
                FilterOption(name="metadata", value=metadata),
                FilterOption(name="reset", value=reset),
                FilterOption(name="measure_perchannel", value=measure_perchannel),
                FilterOption(name="measure_overall", value=measure_overall),
            ],
        )
    )
    return graph


def bwdif_vulkan(
    graph: Stream,
):
    """Deinterlace Vulkan frames via bwdif"""
    graph.append(Filter(command="bwdif_vulkan", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def fftdnoiz(
    graph: Stream,
    sigma: Optional[float] = None,
    amount: Optional[float] = None,
    block: Optional[int] = None,
    overlap: Optional[float] = None,
    method: Optional[str] = None,
    prev: Optional[int] = None,
    next: Optional[int] = None,
    planes: Optional[int] = None,
    window: Optional[str] = None,
):
    """Denoise frames using 3D FFT.

    :param float sigma: set denoise strength
    :param float amount: set amount of denoising
    :param int block: set block size
    :param float overlap: set block overlap
    :param str method: set method of denoising
            possible values: wiener, hard
    :param int prev: set number of previous frames for temporal denoising
    :param int next: set number of next frames for temporal denoising
    :param int planes: set planes to filter
    :param str window: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    """
    graph.append(
        Filter(
            command="fftdnoiz",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="sigma", value=sigma),
                FilterOption(name="amount", value=amount),
                FilterOption(name="block", value=block),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="method", value=method),
                FilterOption(name="prev", value=prev),
                FilterOption(name="next", value=next),
                FilterOption(name="planes", value=planes),
                FilterOption(name="window", value=window),
            ],
        )
    )
    return graph


def dblur(graph: Stream, angle: Optional[float] = None, radius: Optional[float] = None, planes: Optional[int] = None):
    """Apply Directional Blur filter.

    :param float angle: set angle
    :param float radius: set radius
    :param int planes: set planes to filter
    """
    graph.append(
        Filter(
            command="dblur",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="angle", value=angle),
                FilterOption(name="radius", value=radius),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def atadenoise(
    graph: Stream,
    _0a: Optional[float] = None,
    _0b: Optional[float] = None,
    _1a: Optional[float] = None,
    _1b: Optional[float] = None,
    _2a: Optional[float] = None,
    _2b: Optional[float] = None,
    s: Optional[int] = None,
    p: Optional[str] = None,
    a: Optional[str] = None,
    _0s: Optional[float] = None,
    _1s: Optional[float] = None,
    _2s: Optional[float] = None,
):
    """Apply an Adaptive Temporal Averaging Denoiser.

    :param float 0a: set threshold A for 1st plane
    :param float 0b: set threshold B for 1st plane
    :param float 1a: set threshold A for 2nd plane
    :param float 1b: set threshold B for 2nd plane
    :param float 2a: set threshold A for 3rd plane
    :param float 2b: set threshold B for 3rd plane
    :param int s: set how many frames to use
    :param str p: set what planes to filter
    :param str a: set variant of algorithm
            possible values: p, s
    :param float 0s: set sigma for 1st plane
    :param float 1s: set sigma for 2nd plane
    :param float 2s: set sigma for 3rd plane
    """
    graph.append(
        Filter(
            command="atadenoise",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="0a", value=_0a),
                FilterOption(name="0b", value=_0b),
                FilterOption(name="1a", value=_1a),
                FilterOption(name="1b", value=_1b),
                FilterOption(name="2a", value=_2a),
                FilterOption(name="2b", value=_2b),
                FilterOption(name="s", value=s),
                FilterOption(name="p", value=p),
                FilterOption(name="a", value=a),
                FilterOption(name="0s", value=_0s),
                FilterOption(name="1s", value=_1s),
                FilterOption(name="2s", value=_2s),
            ],
        )
    )
    return graph


def psnr(graph: Stream, stats_file: Optional[str] = None, stats_version: Optional[int] = None, output_max: Optional[bool] = None):
    """Calculate the PSNR between two video streams.

    :param str stats_file: Set file where to store per-frame difference information
    :param int stats_version: Set the format version for the stats file.
    :param bool output_max: Add raw stats (max values) to the output log.
    """
    graph.append(
        Filter(
            command="psnr",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="stats_file", value=stats_file),
                FilterOption(name="stats_version", value=stats_version),
                FilterOption(name="output_max", value=output_max),
            ],
        )
    )
    return graph


def dialoguenhance(graph: Stream, original: Optional[float] = None, enhance: Optional[float] = None, voice: Optional[float] = None):
    """Audio Dialogue Enhancement.

    :param float original: set original center factor
    :param float enhance: set dialogue enhance factor
    :param float voice: set voice detection factor
    """
    graph.append(
        Filter(
            command="dialoguenhance",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="original", value=original),
                FilterOption(name="enhance", value=enhance),
                FilterOption(name="voice", value=voice),
            ],
        )
    )
    return graph


def v360(
    graph: Stream,
    input: Optional[str] = None,
    output: Optional[str] = None,
    interp: Optional[str] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    in_stereo: Optional[str] = None,
    out_stereo: Optional[str] = None,
    in_forder: Optional[str] = None,
    out_forder: Optional[str] = None,
    in_frot: Optional[str] = None,
    out_frot: Optional[str] = None,
    in_pad: Optional[float] = None,
    out_pad: Optional[float] = None,
    fin_pad: Optional[int] = None,
    fout_pad: Optional[int] = None,
    yaw: Optional[float] = None,
    pitch: Optional[float] = None,
    roll: Optional[float] = None,
    rorder: Optional[str] = None,
    h_fov: Optional[float] = None,
    v_fov: Optional[float] = None,
    d_fov: Optional[float] = None,
    h_flip: Optional[bool] = None,
    v_flip: Optional[bool] = None,
    d_flip: Optional[bool] = None,
    ih_flip: Optional[bool] = None,
    iv_flip: Optional[bool] = None,
    in_trans: Optional[bool] = None,
    out_trans: Optional[bool] = None,
    ih_fov: Optional[float] = None,
    iv_fov: Optional[float] = None,
    id_fov: Optional[float] = None,
    h_offset: Optional[float] = None,
    v_offset: Optional[float] = None,
    alpha_mask: Optional[bool] = None,
    reset_rot: Optional[bool] = None,
):
    """Convert 360 projection of video.

    :param str input: set input projection
            possible values: e, equirect, c3x2, c6x1, eac, dfisheye, flat, rectilinear, gnomonic, barrel, fb, c1x6, sg, mercator, ball, hammer, sinusoidal, fisheye, pannini, cylindrical, tetrahedron, barrelsplit, tsp, hequirect, he, equisolid, og, octahedron, cylindricalea
    :param str output: set output projection
            possible values: e, equirect, c3x2, c6x1, eac, dfisheye, flat, rectilinear, gnomonic, barrel, fb, c1x6, sg, mercator, ball, hammer, sinusoidal, fisheye, pannini, cylindrical, perspective, tetrahedron, barrelsplit, tsp, hequirect, he, equisolid, og, octahedron, cylindricalea
    :param str interp: set interpolation method
            possible values: near, nearest, line, linear, lagrange9, cube, cubic, lanc, lanczos, sp16, spline16, gauss, gaussian, mitchell
    :param int w: output width
    :param int h: output height
    :param str in_stereo: input stereo format
            possible values: 2d, sbs, tb
    :param str out_stereo: output stereo format
            possible values: 2d, sbs, tb
    :param str in_forder: input cubemap face order
    :param str out_forder: output cubemap face order
    :param str in_frot: input cubemap face rotation
    :param str out_frot: output cubemap face rotation
    :param float in_pad: percent input cubemap pads
    :param float out_pad: percent output cubemap pads
    :param int fin_pad: fixed input cubemap pads
    :param int fout_pad: fixed output cubemap pads
    :param float yaw: yaw rotation
    :param float pitch: pitch rotation
    :param float roll: roll rotation
    :param str rorder: rotation order
    :param float h_fov: output horizontal field of view
    :param float v_fov: output vertical field of view
    :param float d_fov: output diagonal field of view
    :param bool h_flip: flip out video horizontally
    :param bool v_flip: flip out video vertically
    :param bool d_flip: flip out video indepth
    :param bool ih_flip: flip in video horizontally
    :param bool iv_flip: flip in video vertically
    :param bool in_trans: transpose video input
    :param bool out_trans: transpose video output
    :param float ih_fov: input horizontal field of view
    :param float iv_fov: input vertical field of view
    :param float id_fov: input diagonal field of view
    :param float h_offset: output horizontal off-axis offset
    :param float v_offset: output vertical off-axis offset
    :param bool alpha_mask: build mask in alpha plane
    :param bool reset_rot: reset rotation
    """
    graph.append(
        Filter(
            command="v360",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="input", value=input),
                FilterOption(name="output", value=output),
                FilterOption(name="interp", value=interp),
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="in_stereo", value=in_stereo),
                FilterOption(name="out_stereo", value=out_stereo),
                FilterOption(name="in_forder", value=in_forder),
                FilterOption(name="out_forder", value=out_forder),
                FilterOption(name="in_frot", value=in_frot),
                FilterOption(name="out_frot", value=out_frot),
                FilterOption(name="in_pad", value=in_pad),
                FilterOption(name="out_pad", value=out_pad),
                FilterOption(name="fin_pad", value=fin_pad),
                FilterOption(name="fout_pad", value=fout_pad),
                FilterOption(name="yaw", value=yaw),
                FilterOption(name="pitch", value=pitch),
                FilterOption(name="roll", value=roll),
                FilterOption(name="rorder", value=rorder),
                FilterOption(name="h_fov", value=h_fov),
                FilterOption(name="v_fov", value=v_fov),
                FilterOption(name="d_fov", value=d_fov),
                FilterOption(name="h_flip", value=h_flip),
                FilterOption(name="v_flip", value=v_flip),
                FilterOption(name="d_flip", value=d_flip),
                FilterOption(name="ih_flip", value=ih_flip),
                FilterOption(name="iv_flip", value=iv_flip),
                FilterOption(name="in_trans", value=in_trans),
                FilterOption(name="out_trans", value=out_trans),
                FilterOption(name="ih_fov", value=ih_fov),
                FilterOption(name="iv_fov", value=iv_fov),
                FilterOption(name="id_fov", value=id_fov),
                FilterOption(name="h_offset", value=h_offset),
                FilterOption(name="v_offset", value=v_offset),
                FilterOption(name="alpha_mask", value=alpha_mask),
                FilterOption(name="reset_rot", value=reset_rot),
            ],
        )
    )
    return graph


def fspp(
    graph: Stream,
    quality: Optional[int] = None,
    qp: Optional[int] = None,
    strength: Optional[int] = None,
    use_bframe_qp: Optional[bool] = None,
):
    """Apply Fast Simple Post-processing filter.

    :param int quality: set quality
    :param int qp: force a constant quantizer parameter
    :param int strength: set filter strength
    :param bool use_bframe_qp: use B-frames' QP
    """
    graph.append(
        Filter(
            command="fspp",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="quality", value=quality),
                FilterOption(name="qp", value=qp),
                FilterOption(name="strength", value=strength),
                FilterOption(name="use_bframe_qp", value=use_bframe_qp),
            ],
        )
    )
    return graph


def yaepblur(graph: Stream, radius: Optional[int] = None, planes: Optional[int] = None, sigma: Optional[int] = None):
    """Yet another edge preserving blur filter.

    :param int radius: set window radius
    :param int planes: set planes to filter
    :param int sigma: set blur strength
    """
    graph.append(
        Filter(
            command="yaepblur",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="radius", value=radius),
                FilterOption(name="planes", value=planes),
                FilterOption(name="sigma", value=sigma),
            ],
        )
    )
    return graph


def vflip(
    graph: Stream,
):
    """Flip the input video vertically."""
    graph.append(Filter(command="vflip", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def normalize(
    graph: Stream,
    blackpt: Optional[str] = None,
    whitept: Optional[str] = None,
    smoothing: Optional[int] = None,
    independence: Optional[float] = None,
    strength: Optional[float] = None,
):
    """Normalize RGB video.
    :param str blackpt: output color to which darkest input color is mapped

    :param str whitept: output color to which brightest input color is mapped
    :param int smoothing: amount of temporal smoothing of the input range, to reduce flicker
    :param float independence: proportion of independent to linked channel normalization
    :param float strength: strength of filter, from no effect to full normalization
    """
    graph.append(
        Filter(
            command="normalize",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="blackpt", value=blackpt),
                FilterOption(name="whitept", value=whitept),
                FilterOption(name="smoothing", value=smoothing),
                FilterOption(name="independence", value=independence),
                FilterOption(name="strength", value=strength),
            ],
        )
    )
    return graph


def minterpolate(
    graph: Stream,
    fps: Optional[str] = None,
    mi_mode: Optional[str] = None,
    mc_mode: Optional[str] = None,
    me_mode: Optional[str] = None,
    me: Optional[str] = None,
    mb_size: Optional[int] = None,
    search_param: Optional[int] = None,
    vsbmc: Optional[int] = None,
    scd: Optional[str] = None,
    scd_threshold: Optional[float] = None,
):
    """Frame rate conversion using Motion Interpolation.

    :param str fps: output's frame rate
    :param str mi_mode: motion interpolation mode
            possible values: dup, blend, mci
    :param str mc_mode: motion compensation mode
            possible values: obmc, aobmc
    :param str me_mode: motion estimation mode
            possible values: bidir, bilat
    :param str me: motion estimation method
            possible values: esa, tss, tdls, ntss, fss, ds, hexbs, epzs, umh
    :param int mb_size: macroblock size
    :param int search_param: search parameter
    :param int vsbmc: variable-size block motion compensation
    :param str scd: scene change detection method
            possible values: none, fdiff
    :param float scd_threshold: scene change threshold
    """
    graph.append(
        Filter(
            command="minterpolate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="fps", value=fps),
                FilterOption(name="mi_mode", value=mi_mode),
                FilterOption(name="mc_mode", value=mc_mode),
                FilterOption(name="me_mode", value=me_mode),
                FilterOption(name="me", value=me),
                FilterOption(name="mb_size", value=mb_size),
                FilterOption(name="search_param", value=search_param),
                FilterOption(name="vsbmc", value=vsbmc),
                FilterOption(name="scd", value=scd),
                FilterOption(name="scd_threshold", value=scd_threshold),
            ],
        )
    )
    return graph


def convolution(
    graph: Stream,
    _0m: Optional[str] = None,
    _1m: Optional[str] = None,
    _2m: Optional[str] = None,
    _3m: Optional[str] = None,
    _0rdiv: Optional[float] = None,
    _1rdiv: Optional[float] = None,
    _2rdiv: Optional[float] = None,
    _3rdiv: Optional[float] = None,
    _0bias: Optional[float] = None,
    _1bias: Optional[float] = None,
    _2bias: Optional[float] = None,
    _3bias: Optional[float] = None,
    _0mode: Optional[str] = None,
    _1mode: Optional[str] = None,
    _2mode: Optional[str] = None,
    _3mode: Optional[str] = None,
):
    """Apply convolution filter.

    :param str 0m: set matrix for 1st plane
    :param str 1m: set matrix for 2nd plane
    :param str 2m: set matrix for 3rd plane
    :param str 3m: set matrix for 4th plane
    :param float 0rdiv: set rdiv for 1st plane
    :param float 1rdiv: set rdiv for 2nd plane
    :param float 2rdiv: set rdiv for 3rd plane
    :param float 3rdiv: set rdiv for 4th plane
    :param float 0bias: set bias for 1st plane
    :param float 1bias: set bias for 2nd plane
    :param float 2bias: set bias for 3rd plane
    :param float 3bias: set bias for 4th plane
    :param str 0mode: set matrix mode for 1st plane
            possible values: square, row, column
    :param str 1mode: set matrix mode for 2nd plane
            possible values: square, row, column
    :param str 2mode: set matrix mode for 3rd plane
            possible values: square, row, column
    :param str 3mode: set matrix mode for 4th plane
            possible values: square, row, column
    """
    graph.append(
        Filter(
            command="convolution",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="0m", value=_0m),
                FilterOption(name="1m", value=_1m),
                FilterOption(name="2m", value=_2m),
                FilterOption(name="3m", value=_3m),
                FilterOption(name="0rdiv", value=_0rdiv),
                FilterOption(name="1rdiv", value=_1rdiv),
                FilterOption(name="2rdiv", value=_2rdiv),
                FilterOption(name="3rdiv", value=_3rdiv),
                FilterOption(name="0bias", value=_0bias),
                FilterOption(name="1bias", value=_1bias),
                FilterOption(name="2bias", value=_2bias),
                FilterOption(name="3bias", value=_3bias),
                FilterOption(name="0mode", value=_0mode),
                FilterOption(name="1mode", value=_1mode),
                FilterOption(name="2mode", value=_2mode),
                FilterOption(name="3mode", value=_3mode),
            ],
        )
    )
    return graph


def prewitt(
    graph: Stream,
):
    """Apply prewitt operator."""
    graph.append(Filter(command="prewitt", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def sobel(
    graph: Stream,
):
    """Apply sobel operator."""
    graph.append(Filter(command="sobel", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def roberts(
    graph: Stream,
):
    """Apply roberts cross operator."""
    graph.append(Filter(command="roberts", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def kirsch(
    graph: Stream,
):
    """Apply kirsch operator."""
    graph.append(Filter(command="kirsch", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def scharr(
    graph: Stream,
):
    """Apply scharr operator."""
    graph.append(Filter(command="scharr", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def ametadata(
    graph: Stream,
    mode: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
    function: Optional[str] = None,
    expr: Optional[str] = None,
    file: Optional[str] = None,
    direct: Optional[bool] = None,
):
    """Manipulate audio frame metadata.

    :param str mode: set a mode of operation
            possible values: select, add, modify, delete, print
    :param str key: set metadata key
    :param str value: set metadata value
    :param str function: function for comparing values
            possible values: same_str, starts_with, less, equal, greater, expr, ends_with
    :param str expr: set expression for expr function
    :param str file: set file where to print metadata information
    :param bool direct: reduce buffering when printing to user-set file or pipe
    """
    graph.append(
        Filter(
            command="ametadata",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="key", value=key),
                FilterOption(name="value", value=value),
                FilterOption(name="function", value=function),
                FilterOption(name="expr", value=expr),
                FilterOption(name="file", value=file),
                FilterOption(name="direct", value=direct),
            ],
        )
    )
    return graph


def metadata(
    graph: Stream,
    mode: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
    function: Optional[str] = None,
    expr: Optional[str] = None,
    file: Optional[str] = None,
    direct: Optional[bool] = None,
):
    """Manipulate video frame metadata.

    :param str mode: set a mode of operation
            possible values: select, add, modify, delete, print
    :param str key: set metadata key
    :param str value: set metadata value
    :param str function: function for comparing values
            possible values: same_str, starts_with, less, equal, greater, expr, ends_with
    :param str expr: set expression for expr function
    :param str file: set file where to print metadata information
    :param bool direct: reduce buffering when printing to user-set file or pipe
    """
    graph.append(
        Filter(
            command="metadata",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="key", value=key),
                FilterOption(name="value", value=value),
                FilterOption(name="function", value=function),
                FilterOption(name="expr", value=expr),
                FilterOption(name="file", value=file),
                FilterOption(name="direct", value=direct),
            ],
        )
    )
    return graph


def eq(
    graph: Stream,
    contrast: Optional[str] = None,
    brightness: Optional[str] = None,
    saturation: Optional[str] = None,
    gamma: Optional[str] = None,
    gamma_r: Optional[str] = None,
    gamma_g: Optional[str] = None,
    gamma_b: Optional[str] = None,
    gamma_weight: Optional[str] = None,
    eval: Optional[str] = None,
):
    """Adjust brightness, contrast, gamma, and saturation.

    :param str contrast: set the contrast adjustment, negative values give a negative image
    :param str brightness: set the brightness adjustment
    :param str saturation: set the saturation adjustment
    :param str gamma: set the initial gamma value
    :param str gamma_r: gamma value for red
    :param str gamma_g: gamma value for green
    :param str gamma_b: gamma value for blue
    :param str gamma_weight: set the gamma weight which reduces the effect of gamma on bright areas
    :param str eval: specify when to evaluate expressions
            possible values: init, frame
    """
    graph.append(
        Filter(
            command="eq",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="contrast", value=contrast),
                FilterOption(name="brightness", value=brightness),
                FilterOption(name="saturation", value=saturation),
                FilterOption(name="gamma", value=gamma),
                FilterOption(name="gamma_r", value=gamma_r),
                FilterOption(name="gamma_g", value=gamma_g),
                FilterOption(name="gamma_b", value=gamma_b),
                FilterOption(name="gamma_weight", value=gamma_weight),
                FilterOption(name="eval", value=eval),
            ],
        )
    )
    return graph


def chorus(
    graph: Stream,
    in_gain: Optional[float] = None,
    out_gain: Optional[float] = None,
    delays: Optional[str] = None,
    decays: Optional[str] = None,
    speeds: Optional[str] = None,
    depths: Optional[str] = None,
):
    """Add a chorus effect to the audio.

    :param float in_gain: set input gain
    :param float out_gain: set output gain
    :param str delays: set delays
    :param str decays: set decays
    :param str speeds: set speeds
    :param str depths: set depths
    """
    graph.append(
        Filter(
            command="chorus",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="in_gain", value=in_gain),
                FilterOption(name="out_gain", value=out_gain),
                FilterOption(name="delays", value=delays),
                FilterOption(name="decays", value=decays),
                FilterOption(name="speeds", value=speeds),
                FilterOption(name="depths", value=depths),
            ],
        )
    )
    return graph


def readvitc(graph: Stream, scan_max: Optional[int] = None, thr_b: Optional[float] = None, thr_w: Optional[float] = None):
    """Read vertical interval timecode and write it to frame metadata.

    :param int scan_max: maximum line numbers to scan for VITC data
    :param float thr_b: black color threshold
    :param float thr_w: white color threshold
    """
    graph.append(
        Filter(
            command="readvitc",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="scan_max", value=scan_max),
                FilterOption(name="thr_b", value=thr_b),
                FilterOption(name="thr_w", value=thr_w),
            ],
        )
    )
    return graph


def codecview(
    graph: Stream,
    mv: Optional[str] = None,
    qp: Optional[bool] = None,
    mv_type: Optional[str] = None,
    frame_type: Optional[str] = None,
    block: Optional[bool] = None,
):
    """Visualize information about some codecs.

    :param str mv: set motion vectors to visualize
            possible values: pf, bf, bb
    :param str mv_type: set motion vectors type
            possible values: fp, bp
    :param str frame_type: set frame types to visualize motion vectors of
            possible values: if, pf, bf
    :param bool block: set block partitioning structure to visualize
    """
    graph.append(
        Filter(
            command="codecview",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mv", value=mv),
                FilterOption(name="qp", value=qp),
                FilterOption(name="mv_type", value=mv_type),
                FilterOption(name="frame_type", value=frame_type),
                FilterOption(name="block", value=block),
            ],
        )
    )
    return graph


def vif(
    graph: Stream,
):
    """Calculate the VIF between two video streams."""
    graph.append(Filter(command="vif", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def aformat(graph: Stream, sample_fmts: Optional[str] = None, sample_rates: Optional[str] = None, channel_layouts: Optional[str] = None):
    """Convert the input audio to one of the specified formats.

    :param str sample_fmts: A '|'-separated list of sample formats.
    :param str sample_rates: A '|'-separated list of sample rates.
    :param str channel_layouts: A '|'-separated list of channel layouts.
    """
    graph.append(
        Filter(
            command="aformat",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="sample_fmts", value=sample_fmts),
                FilterOption(name="sample_rates", value=sample_rates),
                FilterOption(name="channel_layouts", value=channel_layouts),
            ],
        )
    )
    return graph


def asoftclip(
    graph: Stream,
    type: Optional[str] = None,
    threshold: Optional[float] = None,
    output: Optional[float] = None,
    param: Optional[float] = None,
    oversample: Optional[int] = None,
):
    """Audio Soft Clipper.

    :param str type: set softclip type
            possible values: hard, tanh, atan, cubic, exp, alg, quintic, sin, erf
    :param float threshold: set softclip threshold
    :param float output: set softclip output gain
    :param float param: set softclip parameter
    :param int oversample: set oversample factor
    """
    graph.append(
        Filter(
            command="asoftclip",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="type", value=type),
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="output", value=output),
                FilterOption(name="param", value=param),
                FilterOption(name="oversample", value=oversample),
            ],
        )
    )
    return graph


def bitplanenoise(graph: Stream, bitplane: Optional[int] = None, filter: Optional[bool] = None):
    """Measure bit plane noise.

    :param int bitplane: set bit plane to use for measuring noise
    :param bool filter: show noisy pixels
    """
    graph.append(
        Filter(
            command="bitplanenoise",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="bitplane", value=bitplane), FilterOption(name="filter", value=filter)],
        )
    )
    return graph


def stereotools(
    graph: Stream,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    balance_in: Optional[float] = None,
    balance_out: Optional[float] = None,
    softclip: Optional[bool] = None,
    mutel: Optional[bool] = None,
    muter: Optional[bool] = None,
    phasel: Optional[bool] = None,
    phaser: Optional[bool] = None,
    mode: Optional[str] = None,
    slev: Optional[float] = None,
    sbal: Optional[float] = None,
    mlev: Optional[float] = None,
    mpan: Optional[float] = None,
    base: Optional[float] = None,
    delay: Optional[float] = None,
    sclevel: Optional[float] = None,
    phase: Optional[float] = None,
    bmode_in: Optional[str] = None,
    bmode_out: Optional[int] = None,
):
    """Apply various stereo tools.

    :param float level_in: set level in
    :param float level_out: set level out
    :param float balance_in: set balance in
    :param float balance_out: set balance out
    :param bool softclip: enable softclip
    :param bool mutel: mute L
    :param bool muter: mute R
    :param bool phasel: phase L
    :param bool phaser: phase R
    :param str mode: set stereo mode
            possible values: lr>lr, lr>ms, ms>lr, lr>ll, lr>rr, lr>l+r, lr>rl, ms>ll, ms>rr, ms>rl, lr>l-r
    :param float slev: set side level
    :param float sbal: set side balance
    :param float mlev: set middle level
    :param float mpan: set middle pan
    :param float base: set stereo base
    :param float delay: set delay
    :param float sclevel: set S/C level
    :param float phase: set stereo phase
    :param str bmode_in: set balance in mode
            possible values: balance, amplitude, power
    :param int bmode_out: set balance out mode
    """
    graph.append(
        Filter(
            command="stereotools",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="balance_in", value=balance_in),
                FilterOption(name="balance_out", value=balance_out),
                FilterOption(name="softclip", value=softclip),
                FilterOption(name="mutel", value=mutel),
                FilterOption(name="muter", value=muter),
                FilterOption(name="phasel", value=phasel),
                FilterOption(name="phaser", value=phaser),
                FilterOption(name="mode", value=mode),
                FilterOption(name="slev", value=slev),
                FilterOption(name="sbal", value=sbal),
                FilterOption(name="mlev", value=mlev),
                FilterOption(name="mpan", value=mpan),
                FilterOption(name="base", value=base),
                FilterOption(name="delay", value=delay),
                FilterOption(name="sclevel", value=sclevel),
                FilterOption(name="phase", value=phase),
                FilterOption(name="bmode_in", value=bmode_in),
                FilterOption(name="bmode_out", value=bmode_out),
            ],
        )
    )
    return graph


def framerate(
    graph: Stream,
    fps: Optional[str] = None,
    interp_start: Optional[int] = None,
    interp_end: Optional[int] = None,
    scene: Optional[float] = None,
    flags: Optional[str] = None,
):
    """Upsamples or downsamples progressive source between specified frame rates.

    :param str fps: required output frames per second rate
    :param int interp_start: point to start linear interpolation
    :param int interp_end: point to end linear interpolation
    :param float scene: scene change level
    :param str flags: set flags
            possible values: scene_change_detect, scd
    """
    graph.append(
        Filter(
            command="framerate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="fps", value=fps),
                FilterOption(name="interp_start", value=interp_start),
                FilterOption(name="interp_end", value=interp_end),
                FilterOption(name="scene", value=scene),
                FilterOption(name="flags", value=flags),
            ],
        )
    )
    return graph


def nlmeans_opencl(
    graph: Stream,
    s: Optional[float] = None,
    p: Optional[int] = None,
    pc: Optional[int] = None,
    r: Optional[int] = None,
    rc: Optional[int] = None,
):
    """Non-local means denoiser through OpenCL

    :param float s: denoising strength
    :param int p: patch size
    :param int pc: patch size for chroma planes
    :param int r: research window
    :param int rc: research window for chroma planes
    """
    graph.append(
        Filter(
            command="nlmeans_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="s", value=s),
                FilterOption(name="p", value=p),
                FilterOption(name="pc", value=pc),
                FilterOption(name="r", value=r),
                FilterOption(name="rc", value=rc),
            ],
        )
    )
    return graph


def hwupload_cuda(
    graph: Stream,
):
    """Upload a system memory frame to a CUDA device."""
    graph.append(Filter(command="hwupload_cuda", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def telecine(graph: Stream, first_field: Optional[str] = None, pattern: Optional[str] = None):
    """Apply a telecine pattern.

    :param str first_field: select first field
            possible values: top, t, bottom, b
    :param str pattern: pattern that describe for how many fields a frame is to be displayed
    """
    graph.append(
        Filter(
            command="telecine",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="first_field", value=first_field), FilterOption(name="pattern", value=pattern)],
        )
    )
    return graph


def pixelize(
    graph: Stream,
    width: Optional[int] = None,
    w: Optional[int] = None,
    height: Optional[int] = None,
    h: Optional[int] = None,
    mode: Optional[str] = None,
    planes: Optional[str] = None,
):
    """Pixelize video.

    :param int width: set block width
    :param int w: set block width
    :param int height: set block height
    :param int h: set block height
    :param str mode: set the pixelize mode
            possible values: avg, min, max
    :param str planes: set what planes to filter
    """
    graph.append(
        Filter(
            command="pixelize",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="width", value=width),
                FilterOption(name="w", value=w),
                FilterOption(name="height", value=height),
                FilterOption(name="h", value=h),
                FilterOption(name="mode", value=mode),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def gradients(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    c0: Optional[str] = None,
    c1: Optional[str] = None,
    c2: Optional[str] = None,
    c3: Optional[str] = None,
    c4: Optional[str] = None,
    c5: Optional[str] = None,
    c6: Optional[str] = None,
    c7: Optional[str] = None,
    x0: Optional[int] = None,
    y0: Optional[int] = None,
    x1: Optional[int] = None,
    y1: Optional[int] = None,
    nb_colors: Optional[int] = None,
    seed: Optional[int] = None,
    duration: Optional[int] = None,
    speed: Optional[float] = None,
    type: Optional[str] = None,
):
    """Draw a gradients.

    :param int size: set frame size
    :param str rate: set frame rate
    :param str c0: set 1st color
    :param str c1: set 2nd color
    :param str c2: set 3rd color
    :param str c3: set 4th color
    :param str c4: set 5th color
    :param str c5: set 6th color
    :param str c6: set 7th color
    :param str c7: set 8th color
    :param int x0: set gradient line source x0
    :param int y0: set gradient line source y0
    :param int x1: set gradient line destination x1
    :param int y1: set gradient line destination y1
    :param int nb_colors: set the number of colors
    :param int seed: set the seed
    :param int duration: set video duration
    :param float speed: set gradients rotation speed
    :param str type: set gradient type
            possible values: linear, radial, circular, spiral, square
    """
    graph.append(
        Filter(
            command="gradients",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="c0", value=c0),
                FilterOption(name="c1", value=c1),
                FilterOption(name="c2", value=c2),
                FilterOption(name="c3", value=c3),
                FilterOption(name="c4", value=c4),
                FilterOption(name="c5", value=c5),
                FilterOption(name="c6", value=c6),
                FilterOption(name="c7", value=c7),
                FilterOption(name="x0", value=x0),
                FilterOption(name="y0", value=y0),
                FilterOption(name="x1", value=x1),
                FilterOption(name="y1", value=y1),
                FilterOption(name="nb_colors", value=nb_colors),
                FilterOption(name="seed", value=seed),
                FilterOption(name="duration", value=duration),
                FilterOption(name="speed", value=speed),
                FilterOption(name="type", value=type),
            ],
        )
    )
    return graph


def smartblur(
    graph: Stream,
    luma_radius: Optional[float] = None,
    lr: Optional[float] = None,
    luma_strength: Optional[float] = None,
    ls: Optional[float] = None,
    luma_threshold: Optional[int] = None,
    lt: Optional[int] = None,
    chroma_radius: Optional[float] = None,
    cr: Optional[float] = None,
    chroma_strength: Optional[float] = None,
    cs: Optional[float] = None,
    chroma_threshold: Optional[int] = None,
    ct: Optional[int] = None,
):
    """Blur the input video without impacting the outlines.

    :param float luma_radius: set luma radius
    :param float lr: set luma radius
    :param float luma_strength: set luma strength
    :param float ls: set luma strength
    :param int luma_threshold: set luma threshold
    :param int lt: set luma threshold
    :param float chroma_radius: set chroma radius
    :param float cr: set chroma radius
    :param float chroma_strength: set chroma strength
    :param float cs: set chroma strength
    :param int chroma_threshold: set chroma threshold
    :param int ct: set chroma threshold
    """
    graph.append(
        Filter(
            command="smartblur",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="luma_radius", value=luma_radius),
                FilterOption(name="lr", value=lr),
                FilterOption(name="luma_strength", value=luma_strength),
                FilterOption(name="ls", value=ls),
                FilterOption(name="luma_threshold", value=luma_threshold),
                FilterOption(name="lt", value=lt),
                FilterOption(name="chroma_radius", value=chroma_radius),
                FilterOption(name="cr", value=cr),
                FilterOption(name="chroma_strength", value=chroma_strength),
                FilterOption(name="cs", value=cs),
                FilterOption(name="chroma_threshold", value=chroma_threshold),
                FilterOption(name="ct", value=ct),
            ],
        )
    )
    return graph


def acontrast(graph: Stream, contrast: Optional[float] = None):
    """Simple audio dynamic range compression/expansion filter.

    :param float contrast: set contrast
    """
    graph.append(Filter(command="acontrast", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="contrast", value=contrast)]))
    return graph


def ssim360(
    graph: Stream,
    stats_file: Optional[str] = None,
    compute_chroma: Optional[int] = None,
    frame_skip_ratio: Optional[int] = None,
    ref_projection: Optional[str] = None,
    main_projection: Optional[int] = None,
    ref_stereo: Optional[str] = None,
    main_stereo: Optional[int] = None,
    ref_pad: Optional[float] = None,
    main_pad: Optional[float] = None,
    use_tape: Optional[int] = None,
    heatmap_str: Optional[str] = None,
    default_heatmap_width: Optional[int] = None,
    default_heatmap_height: Optional[int] = None,
):
    """Calculate the SSIM between two 360 video streams.

    :param str stats_file: Set file where to store per-frame difference information
    :param int compute_chroma: Specifies if non-luma channels must be computed
    :param int frame_skip_ratio: Specifies the number of frames to be skipped from evaluation, for every evaluated frame
    :param str ref_projection: projection of the reference video
            possible values: e, equirect, c3x2, c2x3, barrel, barrelsplit
    :param int main_projection: projection of the main video
    :param str ref_stereo: stereo format of the reference video
            possible values: mono, tb, lr
    :param int main_stereo: stereo format of main video
    :param float ref_pad: Expansion (padding) coefficient for each cube face of the reference video
    :param float main_pad: Expansion (padding) coeffiecient for each cube face of the main video
    :param int use_tape: Specifies if the tape based SSIM 360 algorithm must be used independent of the input video types
    :param str heatmap_str: Heatmap data for view-based evaluation. For heatmap file format, please refer to EntSphericalVideoHeatmapData.
    :param int default_heatmap_width: Default heatmap dimension. Will be used when dimension is not specified in heatmap data.
    :param int default_heatmap_height: Default heatmap dimension. Will be used when dimension is not specified in heatmap data.
    """
    graph.append(
        Filter(
            command="ssim360",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="stats_file", value=stats_file),
                FilterOption(name="compute_chroma", value=compute_chroma),
                FilterOption(name="frame_skip_ratio", value=frame_skip_ratio),
                FilterOption(name="ref_projection", value=ref_projection),
                FilterOption(name="main_projection", value=main_projection),
                FilterOption(name="ref_stereo", value=ref_stereo),
                FilterOption(name="main_stereo", value=main_stereo),
                FilterOption(name="ref_pad", value=ref_pad),
                FilterOption(name="main_pad", value=main_pad),
                FilterOption(name="use_tape", value=use_tape),
                FilterOption(name="heatmap_str", value=heatmap_str),
                FilterOption(name="default_heatmap_width", value=default_heatmap_width),
                FilterOption(name="default_heatmap_height", value=default_heatmap_height),
            ],
        )
    )
    return graph


def mcompand(graph: Stream, args: Optional[str] = None):
    """Multiband Compress or expand audio dynamic range.

    :param str args: set parameters for each band
    """
    graph.append(Filter(command="mcompand", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="args", value=args)]))
    return graph


def anlmdn(
    graph: Stream,
    strength: Optional[float] = None,
    patch: Optional[int] = None,
    research: Optional[int] = None,
    output: Optional[str] = None,
    smooth: Optional[float] = None,
):
    """Reduce broadband noise from stream using Non-Local Means.

    :param float strength: set denoising strength
    :param int patch: set patch duration
    :param int research: set research duration
    :param str output: set output mode
            possible values: i, o, n
    :param float smooth: set smooth factor
    """
    graph.append(
        Filter(
            command="anlmdn",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="strength", value=strength),
                FilterOption(name="patch", value=patch),
                FilterOption(name="research", value=research),
                FilterOption(name="output", value=output),
                FilterOption(name="smooth", value=smooth),
            ],
        )
    )
    return graph


def fade(
    graph: Stream,
    type: Optional[str] = None,
    start_frame: Optional[int] = None,
    nb_frames: Optional[int] = None,
    alpha: Optional[bool] = None,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
    color: Optional[str] = None,
):
    """Fade in/out input video.

    :param str type: set the fade direction
            possible values: in, out
    :param int start_frame: Number of the first frame to which to apply the effect.
    :param int nb_frames: Number of frames to which the effect should be applied.
    :param bool alpha: fade alpha if it is available on the input
    :param int start_time: Number of seconds of the beginning of the effect.
    :param int duration: Duration of the effect in seconds.
    :param str color: set color
    """
    graph.append(
        Filter(
            command="fade",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="type", value=type),
                FilterOption(name="start_frame", value=start_frame),
                FilterOption(name="nb_frames", value=nb_frames),
                FilterOption(name="alpha", value=alpha),
                FilterOption(name="start_time", value=start_time),
                FilterOption(name="duration", value=duration),
                FilterOption(name="color", value=color),
            ],
        )
    )
    return graph


def freezeframes(graph: Stream, first: Optional[int] = None, last: Optional[int] = None, replace: Optional[int] = None):
    """Freeze video frames.

    :param int first: set first frame to freeze
    :param int last: set last frame to freeze
    :param int replace: set frame to replace
    """
    graph.append(
        Filter(
            command="freezeframes",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="first", value=first),
                FilterOption(name="last", value=last),
                FilterOption(name="replace", value=replace),
            ],
        )
    )
    return graph


def phase(graph: Stream, mode: Optional[str] = None):
    """Phase shift fields.

    :param str mode: set phase mode
            possible values: p, t, b, T, B, u, U, a, A
    """
    graph.append(Filter(command="phase", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="mode", value=mode)]))
    return graph


def headphone(
    graph: Stream,
    map: Optional[str] = None,
    gain: Optional[float] = None,
    lfe: Optional[float] = None,
    type: Optional[str] = None,
    size: Optional[int] = None,
    hrir: Optional[str] = None,
):
    """Apply headphone binaural spatialization with HRTFs in additional streams.

    :param str map: set channels convolution mappings
    :param float gain: set gain in dB
    :param float lfe: set lfe gain in dB
    :param str type: set processing
            possible values: time, freq
    :param int size: set frame size
    :param str hrir: set hrir format
            possible values: stereo, multich
    """
    graph.append(
        Filter(
            command="headphone",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="map", value=map),
                FilterOption(name="gain", value=gain),
                FilterOption(name="lfe", value=lfe),
                FilterOption(name="type", value=type),
                FilterOption(name="size", value=size),
                FilterOption(name="hrir", value=hrir),
            ],
        )
    )
    return graph


def gblur_vulkan(
    graph: Stream,
    sigma: Optional[float] = None,
    sigmaV: Optional[float] = None,
    planes: Optional[int] = None,
    size: Optional[int] = None,
    sizeV: Optional[int] = None,
):
    """Gaussian Blur in Vulkan

    :param float sigma: Set sigma
    :param float sigmaV: Set vertical sigma
    :param int planes: Set planes to filter
    :param int size: Set kernel size
    :param int sizeV: Set vertical kernel size
    """
    graph.append(
        Filter(
            command="gblur_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="sigma", value=sigma),
                FilterOption(name="sigmaV", value=sigmaV),
                FilterOption(name="planes", value=planes),
                FilterOption(name="size", value=size),
                FilterOption(name="sizeV", value=sizeV),
            ],
        )
    )
    return graph


def yadif(
    graph: Stream,
):
    """Deinterlace the input image."""
    graph.append(Filter(command="yadif", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def hysteresis(graph: Stream, planes: Optional[int] = None, threshold: Optional[int] = None):
    """Grow first stream into second stream by connecting components.

    :param int planes: set planes
    :param int threshold: set threshold
    """
    graph.append(
        Filter(
            command="hysteresis",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="planes", value=planes), FilterOption(name="threshold", value=threshold)],
        )
    )
    return graph


def dnn_detect(
    graph: Stream,
    dnn_backend: Optional[int] = None,
    model: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
    backend_configs: Optional[str] = None,
    options: Optional[str] = None,
    _async: Optional[bool] = None,
    confidence: Optional[float] = None,
    labels: Optional[str] = None,
    model_type: Optional[str] = None,
    cell_w: Optional[int] = None,
    cell_h: Optional[int] = None,
    nb_classes: Optional[int] = None,
    anchors: Optional[str] = None,
):
    """Apply DNN detect filter to the input.

    :param int dnn_backend: DNN backend
    :param str model: path to model file
    :param str input: input name of the model
    :param str output: output name of the model
    :param str backend_configs: backend configs
    :param str options: backend configs (deprecated, use backend_configs)
    :param bool async: use DNN async inference (ignored, use backend_configs='async=1')
    :param float confidence: threshold of confidence
    :param str labels: path to labels file
    :param str model_type: DNN detection model type
            possible values: ssd, yolo, yolov3, yolov4
    :param int cell_w: cell width
    :param int cell_h: cell height
    :param int nb_classes: The number of class
    :param str anchors: anchors, splited by '&'
    """
    graph.append(
        Filter(
            command="dnn_detect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="dnn_backend", value=dnn_backend),
                FilterOption(name="model", value=model),
                FilterOption(name="input", value=input),
                FilterOption(name="output", value=output),
                FilterOption(name="backend_configs", value=backend_configs),
                FilterOption(name="options", value=options),
                FilterOption(name="async", value=_async),
                FilterOption(name="confidence", value=confidence),
                FilterOption(name="labels", value=labels),
                FilterOption(name="model_type", value=model_type),
                FilterOption(name="cell_w", value=cell_w),
                FilterOption(name="cell_h", value=cell_h),
                FilterOption(name="nb_classes", value=nb_classes),
                FilterOption(name="anchors", value=anchors),
            ],
        )
    )
    return graph


def floodfill(
    graph: Stream,
    x: Optional[int] = None,
    y: Optional[int] = None,
    s0: Optional[int] = None,
    s1: Optional[int] = None,
    s2: Optional[int] = None,
    s3: Optional[int] = None,
    d0: Optional[int] = None,
    d1: Optional[int] = None,
    d2: Optional[int] = None,
    d3: Optional[int] = None,
):
    """Fill area with same color with another color.

    :param int x: set pixel x coordinate
    :param int y: set pixel y coordinate
    :param int s0: set source #0 component value
    :param int s1: set source #1 component value
    :param int s2: set source #2 component value
    :param int s3: set source #3 component value
    :param int d0: set destination #0 component value
    :param int d1: set destination #1 component value
    :param int d2: set destination #2 component value
    :param int d3: set destination #3 component value
    """
    graph.append(
        Filter(
            command="floodfill",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="s0", value=s0),
                FilterOption(name="s1", value=s1),
                FilterOption(name="s2", value=s2),
                FilterOption(name="s3", value=s3),
                FilterOption(name="d0", value=d0),
                FilterOption(name="d1", value=d1),
                FilterOption(name="d2", value=d2),
                FilterOption(name="d3", value=d3),
            ],
        )
    )
    return graph


def delogo(
    graph: Stream,
    x: Optional[str] = None,
    y: Optional[str] = None,
    w: Optional[str] = None,
    h: Optional[str] = None,
    show: Optional[bool] = None,
):
    """Remove logo from input video.

    :param str x: set logo x position
    :param str y: set logo y position
    :param str w: set logo width
    :param str h: set logo height
    :param bool show: show delogo area
    """
    graph.append(
        Filter(
            command="delogo",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="show", value=show),
            ],
        )
    )
    return graph


def repeatfields(
    graph: Stream,
):
    """Hard repeat fields based on MPEG repeat field flag."""
    graph.append(Filter(command="repeatfields", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def color(
    graph: Stream,
    color: Optional[str] = None,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    duration: Optional[int] = None,
    sar: Optional[int] = None,
):
    """Provide an uniformly colored input.

    :param str color: set color
    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    """
    graph.append(
        Filter(
            command="color",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="color", value=color),
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
            ],
        )
    )
    return graph


def haldclutsrc(
    graph: Stream, level: Optional[int] = None, rate: Optional[str] = None, duration: Optional[int] = None, sar: Optional[int] = None
):
    """Provide an identity Hald CLUT.

    :param int level: set level
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    """
    graph.append(
        Filter(
            command="haldclutsrc",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="level", value=level),
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
            ],
        )
    )
    return graph


def nullsrc(
    graph: Stream,
):
    """Null video source, return unprocessed video frames."""
    graph.append(Filter(command="nullsrc", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def testsrc(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    duration: Optional[int] = None,
    sar: Optional[int] = None,
    decimals: Optional[int] = None,
):
    """Generate test pattern.

    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param int decimals: set number of decimals to show
    """
    graph.append(
        Filter(
            command="testsrc",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
                FilterOption(name="decimals", value=decimals),
            ],
        )
    )
    return graph


def testsrc2(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    duration: Optional[int] = None,
    sar: Optional[int] = None,
    alpha: Optional[int] = None,
):
    """Generate another test pattern.

    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param int alpha: set global alpha (opacity)
    """
    graph.append(
        Filter(
            command="testsrc2",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
                FilterOption(name="alpha", value=alpha),
            ],
        )
    )
    return graph


def rgbtestsrc(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    duration: Optional[int] = None,
    sar: Optional[int] = None,
    complement: Optional[bool] = None,
):
    """Generate RGB test pattern.

    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param bool complement: set complement colors
    """
    graph.append(
        Filter(
            command="rgbtestsrc",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
                FilterOption(name="complement", value=complement),
            ],
        )
    )
    return graph


def yuvtestsrc(
    graph: Stream,
):
    """Generate YUV test pattern."""
    graph.append(Filter(command="yuvtestsrc", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def pal75bars(
    graph: Stream,
):
    """Generate PAL 75% color bars."""
    graph.append(Filter(command="pal75bars", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def pal100bars(
    graph: Stream,
):
    """Generate PAL 100% color bars."""
    graph.append(Filter(command="pal100bars", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def smptebars(
    graph: Stream,
):
    """Generate SMPTE color bars."""
    graph.append(Filter(command="smptebars", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def smptehdbars(
    graph: Stream,
):
    """Generate SMPTE HD color bars."""
    graph.append(Filter(command="smptehdbars", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def allyuv(
    graph: Stream,
):
    """Generate all yuv colors."""
    graph.append(Filter(command="allyuv", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def allrgb(
    graph: Stream,
):
    """Generate all RGB colors."""
    graph.append(Filter(command="allrgb", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def colorspectrum(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    duration: Optional[int] = None,
    sar: Optional[int] = None,
    type: Optional[str] = None,
):
    """Generate colors spectrum.

    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param str type: set the color spectrum type
            possible values: black, white, all
    """
    graph.append(
        Filter(
            command="colorspectrum",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
                FilterOption(name="type", value=type),
            ],
        )
    )
    return graph


def colorchart(
    graph: Stream,
    rate: Optional[str] = None,
    duration: Optional[int] = None,
    sar: Optional[int] = None,
    patch_size: Optional[int] = None,
    preset: Optional[str] = None,
):
    """Generate color checker chart.

    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param int patch_size: set the single patch size
    :param str preset: set the color checker chart preset
            possible values: reference, skintones
    """
    graph.append(
        Filter(
            command="colorchart",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
                FilterOption(name="patch_size", value=patch_size),
                FilterOption(name="preset", value=preset),
            ],
        )
    )
    return graph


def zoneplate(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    duration: Optional[int] = None,
    sar: Optional[int] = None,
    precision: Optional[int] = None,
    xo: Optional[int] = None,
    yo: Optional[int] = None,
    to: Optional[int] = None,
    k0: Optional[int] = None,
    kx: Optional[int] = None,
    ky: Optional[int] = None,
    kt: Optional[int] = None,
    kxt: Optional[int] = None,
    kyt: Optional[int] = None,
    kxy: Optional[int] = None,
    kx2: Optional[int] = None,
    ky2: Optional[int] = None,
    kt2: Optional[int] = None,
    ku: Optional[int] = None,
    kv: Optional[int] = None,
):
    """Generate zone-plate.

    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param int precision: set LUT precision
    :param int xo: set X-axis offset
    :param int yo: set Y-axis offset
    :param int to: set T-axis offset
    :param int k0: set 0-order phase
    :param int kx: set 1-order X-axis phase
    :param int ky: set 1-order Y-axis phase
    :param int kt: set 1-order T-axis phase
    :param int kxt: set X-axis*T-axis product phase
    :param int kyt: set Y-axis*T-axis product phase
    :param int kxy: set X-axis*Y-axis product phase
    :param int kx2: set 2-order X-axis phase
    :param int ky2: set 2-order Y-axis phase
    :param int kt2: set 2-order T-axis phase
    :param int ku: set 0-order U-color phase
    :param int kv: set 0-order V-color phase
    """
    graph.append(
        Filter(
            command="zoneplate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
                FilterOption(name="precision", value=precision),
                FilterOption(name="xo", value=xo),
                FilterOption(name="yo", value=yo),
                FilterOption(name="to", value=to),
                FilterOption(name="k0", value=k0),
                FilterOption(name="kx", value=kx),
                FilterOption(name="ky", value=ky),
                FilterOption(name="kt", value=kt),
                FilterOption(name="kxt", value=kxt),
                FilterOption(name="kyt", value=kyt),
                FilterOption(name="kxy", value=kxy),
                FilterOption(name="kx2", value=kx2),
                FilterOption(name="ky2", value=ky2),
                FilterOption(name="kt2", value=kt2),
                FilterOption(name="ku", value=ku),
                FilterOption(name="kv", value=kv),
            ],
        )
    )
    return graph


def colorchannelmixer(
    graph: Stream,
    rr: Optional[float] = None,
    rg: Optional[float] = None,
    rb: Optional[float] = None,
    ra: Optional[float] = None,
    gr: Optional[float] = None,
    gg: Optional[float] = None,
    gb: Optional[float] = None,
    ga: Optional[float] = None,
    br: Optional[float] = None,
    bg: Optional[float] = None,
    bb: Optional[float] = None,
    ba: Optional[float] = None,
    ar: Optional[float] = None,
    ag: Optional[float] = None,
    ab: Optional[float] = None,
    aa: Optional[float] = None,
    pc: Optional[str] = None,
    pa: Optional[float] = None,
):
    """Adjust colors by mixing color channels.

    :param float rr: set the red gain for the red channel
    :param float rg: set the green gain for the red channel
    :param float rb: set the blue gain for the red channel
    :param float ra: set the alpha gain for the red channel
    :param float gr: set the red gain for the green channel
    :param float gg: set the green gain for the green channel
    :param float gb: set the blue gain for the green channel
    :param float ga: set the alpha gain for the green channel
    :param float br: set the red gain for the blue channel
    :param float bg: set the green gain for the blue channel
    :param float bb: set the blue gain for the blue channel
    :param float ba: set the alpha gain for the blue channel
    :param float ar: set the red gain for the alpha channel
    :param float ag: set the green gain for the alpha channel
    :param float ab: set the blue gain for the alpha channel
    :param float aa: set the alpha gain for the alpha channel
    :param str pc: set the preserve color mode
            possible values: none, lum, max, avg, sum, nrm, pwr
    :param float pa: set the preserve color amount
    """
    graph.append(
        Filter(
            command="colorchannelmixer",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rr", value=rr),
                FilterOption(name="rg", value=rg),
                FilterOption(name="rb", value=rb),
                FilterOption(name="ra", value=ra),
                FilterOption(name="gr", value=gr),
                FilterOption(name="gg", value=gg),
                FilterOption(name="gb", value=gb),
                FilterOption(name="ga", value=ga),
                FilterOption(name="br", value=br),
                FilterOption(name="bg", value=bg),
                FilterOption(name="bb", value=bb),
                FilterOption(name="ba", value=ba),
                FilterOption(name="ar", value=ar),
                FilterOption(name="ag", value=ag),
                FilterOption(name="ab", value=ab),
                FilterOption(name="aa", value=aa),
                FilterOption(name="pc", value=pc),
                FilterOption(name="pa", value=pa),
            ],
        )
    )
    return graph


def backgroundkey(graph: Stream, threshold: Optional[float] = None, similarity: Optional[float] = None, blend: Optional[float] = None):
    """Turns a static background into transparency.

    :param float threshold: set the scene change threshold
    :param float similarity: set the similarity
    :param float blend: set the blend value
    """
    graph.append(
        Filter(
            command="backgroundkey",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="similarity", value=similarity),
                FilterOption(name="blend", value=blend),
            ],
        )
    )
    return graph


def displace(graph: Stream, edge: Optional[str] = None):
    """Displace pixels.

    :param str edge: set edge mode
            possible values: blank, smear, wrap, mirror
    """
    graph.append(Filter(command="displace", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="edge", value=edge)]))
    return graph


def pan(graph: Stream, args: Optional[str] = None):
    """Remix channels with coefficients (panning)."""
    graph.append(Filter(command="pan", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="args", value=args)]))
    return graph


def afdelaysrc(
    graph: Stream,
    delay: Optional[float] = None,
    sample_rate: Optional[int] = None,
    nb_samples: Optional[int] = None,
    taps: Optional[int] = None,
    channel_layout: Optional[str] = None,
):
    """Generate a Fractional delay FIR coefficients.

    :param float delay: set fractional delay
    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param int taps: set number of taps for delay filter
    :param str channel_layout: set channel layout
    """
    graph.append(
        Filter(
            command="afdelaysrc",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="delay", value=delay),
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="taps", value=taps),
                FilterOption(name="channel_layout", value=channel_layout),
            ],
        )
    )
    return graph


def tinterlace(graph: Stream, mode: Optional[str] = None, flags: Optional[str] = None):
    """Perform temporal field interlacing.

    :param str mode: select interlace mode
            possible values: merge, drop_even, drop_odd, pad, interleave_top, interleave_bottom, interlacex2, mergex2
    :param str flags: set flags
            possible values: low_pass_filter, vlpf, complex_filter, cvlpf, exact_tb, bypass_il
    """
    graph.append(
        Filter(
            command="tinterlace",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="mode", value=mode), FilterOption(name="flags", value=flags)],
        )
    )
    return graph


def interlace(graph: Stream, scan: Optional[str] = None, lowpass: Optional[str] = None):
    """Convert progressive video into interlaced.

    :param str scan: scanning mode
            possible values: tff, bff
    :param str lowpass: set vertical low-pass filter
            possible values: off, linear, complex
    """
    graph.append(
        Filter(
            command="interlace",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="scan", value=scan), FilterOption(name="lowpass", value=lowpass)],
        )
    )
    return graph


def format(
    graph: Stream,
):
    """Convert the input video to one of the specified pixel formats."""
    graph.append(Filter(command="format", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def noformat(
    graph: Stream,
):
    """Force libavfilter not to use any of the specified pixel formats for the input to the next filter."""
    graph.append(Filter(command="noformat", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def framestep(graph: Stream, step: Optional[int] = None):
    """Select one frame every N frames.

    :param int step: set frame step
    """
    graph.append(Filter(command="framestep", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="step", value=step)]))
    return graph


def ssim(graph: Stream, stats_file: Optional[str] = None):
    """Calculate the SSIM between two video streams.

    :param str stats_file: Set file where to store per-frame difference information
    """
    graph.append(Filter(command="ssim", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="stats_file", value=stats_file)]))
    return graph


def cropdetect(
    graph: Stream,
    limit: Optional[float] = None,
    round: Optional[int] = None,
    reset: Optional[int] = None,
    skip: Optional[int] = None,
    max_outliers: Optional[int] = None,
    mode: Optional[str] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    mv_threshold: Optional[int] = None,
):
    """Auto-detect crop size.

    :param float limit: Threshold below which the pixel is considered black
    :param int round: Value by which the width/height should be divisible
    :param int reset: Recalculate the crop area after this many frames
    :param int skip: Number of initial frames to skip
    :param int max_outliers: Threshold count of outliers
    :param str mode: set mode
            possible values: black, mvedges
    :param float high: Set high threshold for edge detection
    :param float low: Set low threshold for edge detection
    :param int mv_threshold: motion vector threshold when estimating video window size
    """
    graph.append(
        Filter(
            command="cropdetect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="limit", value=limit),
                FilterOption(name="round", value=round),
                FilterOption(name="reset", value=reset),
                FilterOption(name="skip", value=skip),
                FilterOption(name="max_outliers", value=max_outliers),
                FilterOption(name="mode", value=mode),
                FilterOption(name="high", value=high),
                FilterOption(name="low", value=low),
                FilterOption(name="mv_threshold", value=mv_threshold),
            ],
        )
    )
    return graph


def chromashift(
    graph: Stream,
    cbh: Optional[int] = None,
    cbv: Optional[int] = None,
    crh: Optional[int] = None,
    crv: Optional[int] = None,
    edge: Optional[str] = None,
):
    """Shift chroma.

    :param int cbh: shift chroma-blue horizontally
    :param int cbv: shift chroma-blue vertically
    :param int crh: shift chroma-red horizontally
    :param int crv: shift chroma-red vertically
    :param str edge: set edge operation
            possible values: smear, wrap
    """
    graph.append(
        Filter(
            command="chromashift",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="cbh", value=cbh),
                FilterOption(name="cbv", value=cbv),
                FilterOption(name="crh", value=crh),
                FilterOption(name="crv", value=crv),
                FilterOption(name="edge", value=edge),
            ],
        )
    )
    return graph


def rgbashift(
    graph: Stream,
    rh: Optional[int] = None,
    rv: Optional[int] = None,
    gh: Optional[int] = None,
    gv: Optional[int] = None,
    bh: Optional[int] = None,
    bv: Optional[int] = None,
    ah: Optional[int] = None,
    av: Optional[int] = None,
    edge: Optional[str] = None,
):
    """Shift RGBA.

    :param int rh: shift red horizontally
    :param int rv: shift red vertically
    :param int gh: shift green horizontally
    :param int gv: shift green vertically
    :param int bh: shift blue horizontally
    :param int bv: shift blue vertically
    :param int ah: shift alpha horizontally
    :param int av: shift alpha vertically
    :param str edge: set edge operation
            possible values: smear, wrap
    """
    graph.append(
        Filter(
            command="rgbashift",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rh", value=rh),
                FilterOption(name="rv", value=rv),
                FilterOption(name="gh", value=gh),
                FilterOption(name="gv", value=gv),
                FilterOption(name="bh", value=bh),
                FilterOption(name="bv", value=bv),
                FilterOption(name="ah", value=ah),
                FilterOption(name="av", value=av),
                FilterOption(name="edge", value=edge),
            ],
        )
    )
    return graph


def showspatial(
    graph: Stream, size: Optional[int] = None, win_size: Optional[int] = None, win_func: Optional[str] = None, rate: Optional[str] = None
):
    """Convert input audio to a spatial video output.

    :param int size: set video size
    :param int win_size: set window size
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    :param str rate: set video rate
    """
    graph.append(
        Filter(
            command="showspatial",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="win_size", value=win_size),
                FilterOption(name="win_func", value=win_func),
                FilterOption(name="rate", value=rate),
            ],
        )
    )
    return graph


# def overlay(
#     graph: Stream,
#     x: Optional[str] = None,
#     y: Optional[str] = None,
#     eof_action: Optional[str] = None,
#     eval: Optional[str] = None,
#     shortest: Optional[bool] = None,
#     format: Optional[str] = None,
#     repeatlast: Optional[bool] = None,
#     alpha: Optional[str] = None,
# ):
#     """Overlay a video source on top of the input.
#     :param str x: set the x expression
#     :param str y: set the y expression
#     :param str eof_action: Action to take when encountering EOF from secondary input
#             possible values: repeat, endall, pass
#     :param str eval: specify when to evaluate expressions
#             possible values: init, frame
#     :param bool shortest: force termination when the shortest input terminates
#     :param str format: set output format
#             possible values: yuv420, yuv420p10, yuv422, yuv422p10, yuv444, yuv444p10, rgb, gbrp, auto
#     :param bool repeatlast: repeat overlay of the last overlay frame
#     :param str alpha: alpha format
#             possible values: straight, premultiplied"""
#     graph.append(
#         Filter(
#             command="overlay",
#             filter_type="AVMEDIA_TYPE_VIDEO",
#             params=[
#                 FilterOption(name="x", value=x),
#                 FilterOption(name="y", value=y),
#                 FilterOption(name="eof_action", value=eof_action),
#                 FilterOption(name="eval", value=eval),
#                 FilterOption(name="shortest", value=shortest),
#                 FilterOption(name="format", value=format),
#                 FilterOption(name="repeatlast", value=repeatlast),
#                 FilterOption(name="alpha", value=alpha),
#             ],
#         )
#     )
#     return graph


def null(
    graph: Stream,
):
    """Pass the source unchanged to the output."""
    graph.append(Filter(command="null", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def deshake(
    graph: Stream,
    x: Optional[int] = None,
    y: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    rx: Optional[int] = None,
    ry: Optional[int] = None,
    edge: Optional[str] = None,
    blocksize: Optional[int] = None,
    contrast: Optional[int] = None,
    search: Optional[str] = None,
    filename: Optional[str] = None,
    opencl: Optional[bool] = None,
):
    """Stabilize shaky video.

    :param int x: set x for the rectangular search area
    :param int y: set y for the rectangular search area
    :param int w: set width for the rectangular search area
    :param int h: set height for the rectangular search area
    :param int rx: set x for the rectangular search area
    :param int ry: set y for the rectangular search area
    :param str edge: set edge mode
            possible values: blank, original, clamp, mirror
    :param int blocksize: set motion search blocksize
    :param int contrast: set contrast threshold for blocks
    :param str search: set search strategy
            possible values: exhaustive, less
    :param str filename: set motion search detailed log file name
    :param bool opencl: ignored
    """
    graph.append(
        Filter(
            command="deshake",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="rx", value=rx),
                FilterOption(name="ry", value=ry),
                FilterOption(name="edge", value=edge),
                FilterOption(name="blocksize", value=blocksize),
                FilterOption(name="contrast", value=contrast),
                FilterOption(name="search", value=search),
                FilterOption(name="filename", value=filename),
                FilterOption(name="opencl", value=opencl),
            ],
        )
    )
    return graph


def estdif(
    graph: Stream,
    mode: Optional[str] = None,
    parity: Optional[str] = None,
    deint: Optional[str] = None,
    rslope: Optional[int] = None,
    redge: Optional[int] = None,
    ecost: Optional[int] = None,
    mcost: Optional[int] = None,
    dcost: Optional[int] = None,
    interp: Optional[str] = None,
):
    """Apply Edge Slope Tracing deinterlace.

    :param str mode: specify the mode
            possible values: frame, field
    :param str parity: specify the assumed picture field parity
            possible values: tff, bff, auto
    :param str deint: specify which frames to deinterlace
            possible values: all, interlaced
    :param int rslope: specify the search radius for edge slope tracing
    :param int redge: specify the search radius for best edge matching
    :param int ecost: specify the edge cost for edge matching
    :param int mcost: specify the middle cost for edge matching
    :param int dcost: specify the distance cost for edge matching
    :param str interp: specify the type of interpolation
            possible values: 2p, 4p, 6p
    """
    graph.append(
        Filter(
            command="estdif",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="parity", value=parity),
                FilterOption(name="deint", value=deint),
                FilterOption(name="rslope", value=rslope),
                FilterOption(name="redge", value=redge),
                FilterOption(name="ecost", value=ecost),
                FilterOption(name="mcost", value=mcost),
                FilterOption(name="dcost", value=dcost),
                FilterOption(name="interp", value=interp),
            ],
        )
    )
    return graph


def anoisesrc(
    graph: Stream,
    sample_rate: Optional[int] = None,
    amplitude: Optional[float] = None,
    duration: Optional[int] = None,
    color: Optional[str] = None,
    seed: Optional[int] = None,
    nb_samples: Optional[int] = None,
    density: Optional[float] = None,
):
    """Generate a noise audio signal.

    :param int sample_rate: set sample rate
    :param float amplitude: set amplitude
    :param int duration: set duration
    :param str color: set noise color
            possible values: white, pink, brown, blue, violet, velvet
    :param int seed: set random seed
    :param int nb_samples: set the number of samples per requested frame
    :param float density: set density
    """
    graph.append(
        Filter(
            command="anoisesrc",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="amplitude", value=amplitude),
                FilterOption(name="duration", value=duration),
                FilterOption(name="color", value=color),
                FilterOption(name="seed", value=seed),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="density", value=density),
            ],
        )
    )
    return graph


def aecho(
    graph: Stream,
    in_gain: Optional[float] = None,
    out_gain: Optional[float] = None,
    delays: Optional[str] = None,
    decays: Optional[str] = None,
):
    """Add echoing to the audio.

    :param float in_gain: set signal input gain
    :param float out_gain: set signal output gain
    :param str delays: set list of signal delays
    :param str decays: set list of signal decays
    """
    graph.append(
        Filter(
            command="aecho",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="in_gain", value=in_gain),
                FilterOption(name="out_gain", value=out_gain),
                FilterOption(name="delays", value=delays),
                FilterOption(name="decays", value=decays),
            ],
        )
    )
    return graph


def scale(
    graph: Stream,
    w: Optional[str] = None,
    h: Optional[str] = None,
    flags: Optional[str] = None,
    interl: Optional[bool] = None,
    size: Optional[str] = None,
    in_color_matrix: Optional[str] = None,
    out_color_matrix: Optional[str] = None,
    in_range: Optional[str] = None,
    out_range: Optional[str] = None,
    in_v_chr_pos: Optional[int] = None,
    in_h_chr_pos: Optional[int] = None,
    out_v_chr_pos: Optional[int] = None,
    out_h_chr_pos: Optional[int] = None,
    force_original_aspect_ratio: Optional[str] = None,
    force_divisible_by: Optional[int] = None,
    param0: Optional[float] = None,
    param1: Optional[float] = None,
    eval: Optional[str] = None,
):
    """Scale the input video size and/or convert the image format.

    :param str w: Output video width
    :param str h: Output video height
    :param str flags: Flags to pass to libswscale
    :param bool interl: set interlacing
    :param str size: set video size
    :param str in_color_matrix: set input YCbCr type
            possible values: auto, bt601, bt470, smpte170m, bt709, fcc, smpte240m, bt2020
    :param str out_color_matrix: set output YCbCr type
            possible values: auto, bt601, bt470, smpte170m, bt709, fcc, smpte240m, bt2020
    :param str in_range: set input color range
            possible values: auto, unknown, full, limited, jpeg, mpeg, tv, pc
    :param str out_range: set output color range
            possible values: auto, unknown, full, limited, jpeg, mpeg, tv, pc
    :param int in_v_chr_pos: input vertical chroma position in luma grid/256
    :param int in_h_chr_pos: input horizontal chroma position in luma grid/256
    :param int out_v_chr_pos: output vertical chroma position in luma grid/256
    :param int out_h_chr_pos: output horizontal chroma position in luma grid/256
    :param str force_original_aspect_ratio: decrease or increase w/h if necessary to keep the original AR
            possible values: disable, decrease, increase
    :param int force_divisible_by: enforce that the output resolution is divisible by a defined integer when force_original_aspect_ratio is used
    :param float param0: Scaler param 0
    :param float param1: Scaler param 1
    :param str eval: specify when to evaluate expressions
            possible values: init, frame
    """
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


def decimate(
    graph: Stream,
    cycle: Optional[int] = None,
    dupthresh: Optional[float] = None,
    scthresh: Optional[float] = None,
    blockx: Optional[int] = None,
    blocky: Optional[int] = None,
    ppsrc: Optional[bool] = None,
    chroma: Optional[bool] = None,
    mixed: Optional[bool] = None,
):
    """Decimate frames (post field matching filter).

    :param int cycle: set the number of frame from which one will be dropped
    :param float dupthresh: set duplicate threshold
    :param float scthresh: set scene change threshold
    :param int blockx: set the size of the x-axis blocks used during metric calculations
    :param int blocky: set the size of the y-axis blocks used during metric calculations
    :param bool ppsrc: mark main input as a pre-processed input and activate clean source input stream
    :param bool chroma: set whether or not chroma is considered in the metric calculations
    :param bool mixed: set whether or not the input only partially contains content to be decimated
    """
    graph.append(
        Filter(
            command="decimate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="cycle", value=cycle),
                FilterOption(name="dupthresh", value=dupthresh),
                FilterOption(name="scthresh", value=scthresh),
                FilterOption(name="blockx", value=blockx),
                FilterOption(name="blocky", value=blocky),
                FilterOption(name="ppsrc", value=ppsrc),
                FilterOption(name="chroma", value=chroma),
                FilterOption(name="mixed", value=mixed),
            ],
        )
    )
    return graph


def monochrome(
    graph: Stream, cb: Optional[float] = None, cr: Optional[float] = None, size: Optional[float] = None, high: Optional[float] = None
):
    """Convert video to gray using custom color filter.

    :param float cb: set the chroma blue spot
    :param float cr: set the chroma red spot
    :param float size: set the color filter size
    :param float high: set the highlights strength
    """
    graph.append(
        Filter(
            command="monochrome",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="cb", value=cb),
                FilterOption(name="cr", value=cr),
                FilterOption(name="size", value=size),
                FilterOption(name="high", value=high),
            ],
        )
    )
    return graph


def equalizer(
    graph: Stream,
    frequency: Optional[float] = None,
    width_type: Optional[str] = None,
    width: Optional[float] = None,
    gain: Optional[float] = None,
    mix: Optional[float] = None,
    channels: Optional[str] = None,
    normalize: Optional[bool] = None,
    transform: Optional[str] = None,
    precision: Optional[str] = None,
    blocksize: Optional[int] = None,
):
    """Apply two-pole peaking equalization (EQ) filter.

    :param float frequency: set central frequency
    :param str width_type: set filter-width type
            possible values: h, q, o, s, k
    :param float width: set width
    :param float gain: set gain
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param str transform: set transform type
            possible values: di, dii, tdi, tdii, latt, svf, zdf
    :param str precision: set filtering precision
            possible values: auto, s16, s32, f32, f64
    :param int blocksize: set the block size
    """
    graph.append(
        Filter(
            command="equalizer",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="frequency", value=frequency),
                FilterOption(name="width_type", value=width_type),
                FilterOption(name="width", value=width),
                FilterOption(name="gain", value=gain),
                FilterOption(name="mix", value=mix),
                FilterOption(name="channels", value=channels),
                FilterOption(name="normalize", value=normalize),
                FilterOption(name="transform", value=transform),
                FilterOption(name="precision", value=precision),
                FilterOption(name="blocksize", value=blocksize),
            ],
        )
    )
    return graph


def bass(
    graph: Stream,
):
    """Boost or cut lower frequencies."""
    graph.append(Filter(command="bass", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def lowshelf(
    graph: Stream,
):
    """Apply a low shelf filter."""
    graph.append(Filter(command="lowshelf", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def treble(
    graph: Stream,
):
    """Boost or cut upper frequencies."""
    graph.append(Filter(command="treble", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def highshelf(
    graph: Stream,
):
    """Apply a high shelf filter."""
    graph.append(Filter(command="highshelf", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def tiltshelf(
    graph: Stream,
):
    """Apply a tilt shelf filter."""
    graph.append(Filter(command="tiltshelf", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def bandpass(
    graph: Stream,
    frequency: Optional[float] = None,
    width_type: Optional[str] = None,
    width: Optional[float] = None,
    csg: Optional[bool] = None,
    mix: Optional[float] = None,
    channels: Optional[str] = None,
    normalize: Optional[bool] = None,
    transform: Optional[str] = None,
    precision: Optional[str] = None,
    blocksize: Optional[int] = None,
):
    """Apply a two-pole Butterworth band-pass filter.

    :param float frequency: set central frequency
    :param str width_type: set filter-width type
            possible values: h, q, o, s, k
    :param float width: set width
    :param bool csg: use constant skirt gain
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param str transform: set transform type
            possible values: di, dii, tdi, tdii, latt, svf, zdf
    :param str precision: set filtering precision
            possible values: auto, s16, s32, f32, f64
    :param int blocksize: set the block size
    """
    graph.append(
        Filter(
            command="bandpass",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="frequency", value=frequency),
                FilterOption(name="width_type", value=width_type),
                FilterOption(name="width", value=width),
                FilterOption(name="csg", value=csg),
                FilterOption(name="mix", value=mix),
                FilterOption(name="channels", value=channels),
                FilterOption(name="normalize", value=normalize),
                FilterOption(name="transform", value=transform),
                FilterOption(name="precision", value=precision),
                FilterOption(name="blocksize", value=blocksize),
            ],
        )
    )
    return graph


def bandreject(
    graph: Stream,
    frequency: Optional[float] = None,
    width_type: Optional[str] = None,
    width: Optional[float] = None,
    mix: Optional[float] = None,
    channels: Optional[str] = None,
    normalize: Optional[bool] = None,
    transform: Optional[str] = None,
    precision: Optional[str] = None,
    blocksize: Optional[int] = None,
):
    """Apply a two-pole Butterworth band-reject filter.

    :param float frequency: set central frequency
    :param str width_type: set filter-width type
            possible values: h, q, o, s, k
    :param float width: set width
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param str transform: set transform type
            possible values: di, dii, tdi, tdii, latt, svf, zdf
    :param str precision: set filtering precision
            possible values: auto, s16, s32, f32, f64
    :param int blocksize: set the block size
    """
    graph.append(
        Filter(
            command="bandreject",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="frequency", value=frequency),
                FilterOption(name="width_type", value=width_type),
                FilterOption(name="width", value=width),
                FilterOption(name="mix", value=mix),
                FilterOption(name="channels", value=channels),
                FilterOption(name="normalize", value=normalize),
                FilterOption(name="transform", value=transform),
                FilterOption(name="precision", value=precision),
                FilterOption(name="blocksize", value=blocksize),
            ],
        )
    )
    return graph


def lowpass(
    graph: Stream,
    frequency: Optional[float] = None,
    width_type: Optional[str] = None,
    width: Optional[float] = None,
    poles: Optional[int] = None,
    mix: Optional[float] = None,
    channels: Optional[str] = None,
    normalize: Optional[bool] = None,
    transform: Optional[str] = None,
    precision: Optional[str] = None,
    blocksize: Optional[int] = None,
):
    """Apply a low-pass filter with 3dB point frequency.

    :param float frequency: set frequency
    :param str width_type: set filter-width type
            possible values: h, q, o, s, k
    :param float width: set width
    :param int poles: set number of poles
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param str transform: set transform type
            possible values: di, dii, tdi, tdii, latt, svf, zdf
    :param str precision: set filtering precision
            possible values: auto, s16, s32, f32, f64
    :param int blocksize: set the block size
    """
    graph.append(
        Filter(
            command="lowpass",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="frequency", value=frequency),
                FilterOption(name="width_type", value=width_type),
                FilterOption(name="width", value=width),
                FilterOption(name="poles", value=poles),
                FilterOption(name="mix", value=mix),
                FilterOption(name="channels", value=channels),
                FilterOption(name="normalize", value=normalize),
                FilterOption(name="transform", value=transform),
                FilterOption(name="precision", value=precision),
                FilterOption(name="blocksize", value=blocksize),
            ],
        )
    )
    return graph


def highpass(
    graph: Stream,
    frequency: Optional[float] = None,
    width_type: Optional[str] = None,
    width: Optional[float] = None,
    poles: Optional[int] = None,
    mix: Optional[float] = None,
    channels: Optional[str] = None,
    normalize: Optional[bool] = None,
    transform: Optional[str] = None,
    precision: Optional[str] = None,
    blocksize: Optional[int] = None,
):
    """Apply a high-pass filter with 3dB point frequency.

    :param float frequency: set frequency
    :param str width_type: set filter-width type
            possible values: h, q, o, s, k
    :param float width: set width
    :param int poles: set number of poles
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param str transform: set transform type
            possible values: di, dii, tdi, tdii, latt, svf, zdf
    :param str precision: set filtering precision
            possible values: auto, s16, s32, f32, f64
    :param int blocksize: set the block size
    """
    graph.append(
        Filter(
            command="highpass",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="frequency", value=frequency),
                FilterOption(name="width_type", value=width_type),
                FilterOption(name="width", value=width),
                FilterOption(name="poles", value=poles),
                FilterOption(name="mix", value=mix),
                FilterOption(name="channels", value=channels),
                FilterOption(name="normalize", value=normalize),
                FilterOption(name="transform", value=transform),
                FilterOption(name="precision", value=precision),
                FilterOption(name="blocksize", value=blocksize),
            ],
        )
    )
    return graph


def allpass(
    graph: Stream,
    frequency: Optional[float] = None,
    width_type: Optional[str] = None,
    width: Optional[float] = None,
    mix: Optional[float] = None,
    channels: Optional[str] = None,
    normalize: Optional[bool] = None,
    order: Optional[int] = None,
    transform: Optional[str] = None,
    precision: Optional[str] = None,
):
    """Apply a two-pole all-pass filter.

    :param float frequency: set central frequency
    :param str width_type: set filter-width type
            possible values: h, q, o, s, k
    :param float width: set width
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param int order: set filter order
    :param str transform: set transform type
            possible values: di, dii, tdi, tdii, latt, svf, zdf
    :param str precision: set filtering precision
            possible values: auto, s16, s32, f32, f64
    """
    graph.append(
        Filter(
            command="allpass",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="frequency", value=frequency),
                FilterOption(name="width_type", value=width_type),
                FilterOption(name="width", value=width),
                FilterOption(name="mix", value=mix),
                FilterOption(name="channels", value=channels),
                FilterOption(name="normalize", value=normalize),
                FilterOption(name="order", value=order),
                FilterOption(name="transform", value=transform),
                FilterOption(name="precision", value=precision),
            ],
        )
    )
    return graph


def biquad(
    graph: Stream,
    a0: Optional[float] = None,
    a1: Optional[float] = None,
    a2: Optional[float] = None,
    b0: Optional[float] = None,
    b1: Optional[float] = None,
    b2: Optional[float] = None,
    mix: Optional[float] = None,
    channels: Optional[str] = None,
    normalize: Optional[bool] = None,
    transform: Optional[str] = None,
    precision: Optional[str] = None,
    blocksize: Optional[int] = None,
):
    """Apply a biquad IIR filter with the given coefficients.

    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param str transform: set transform type
            possible values: di, dii, tdi, tdii, latt, svf, zdf
    :param str precision: set filtering precision
            possible values: auto, s16, s32, f32, f64
    :param int blocksize: set the block size
    """
    graph.append(
        Filter(
            command="biquad",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="a0", value=a0),
                FilterOption(name="a1", value=a1),
                FilterOption(name="a2", value=a2),
                FilterOption(name="b0", value=b0),
                FilterOption(name="b1", value=b1),
                FilterOption(name="b2", value=b2),
                FilterOption(name="mix", value=mix),
                FilterOption(name="channels", value=channels),
                FilterOption(name="normalize", value=normalize),
                FilterOption(name="transform", value=transform),
                FilterOption(name="precision", value=precision),
                FilterOption(name="blocksize", value=blocksize),
            ],
        )
    )
    return graph


def spectrumsynth(
    graph: Stream,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    scale: Optional[str] = None,
    slide: Optional[str] = None,
    win_func: Optional[str] = None,
    overlap: Optional[float] = None,
    orientation: Optional[str] = None,
):
    """Convert input spectrum videos to audio output.

    :param int sample_rate: set sample rate
    :param int channels: set channels
    :param str scale: set input amplitude scale
            possible values: lin, log
    :param str slide: set input sliding mode
            possible values: replace, scroll, fullframe, rscroll
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    :param float overlap: set window overlap
    :param str orientation: set orientation
            possible values: vertical, horizontal
    """
    graph.append(
        Filter(
            command="spectrumsynth",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="channels", value=channels),
                FilterOption(name="scale", value=scale),
                FilterOption(name="slide", value=slide),
                FilterOption(name="win_func", value=win_func),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="orientation", value=orientation),
            ],
        )
    )
    return graph


def amix(
    graph: Stream,
    inputs: Optional[int] = None,
    duration: Optional[str] = None,
    dropout_transition: Optional[float] = None,
    weights: Optional[str] = None,
    normalize: Optional[bool] = None,
):
    """Audio mixing.

    :param int inputs: Number of inputs.
    :param str duration: How to determine the end-of-stream.
            possible values: longest, shortest, first
    :param float dropout_transition: Transition time, in seconds, for volume renormalization when an input stream ends.
    :param str weights: Set weight for each input.
    :param bool normalize: Scale inputs
    """
    graph.append(
        Filter(
            command="amix",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="inputs", value=inputs),
                FilterOption(name="duration", value=duration),
                FilterOption(name="dropout_transition", value=dropout_transition),
                FilterOption(name="weights", value=weights),
                FilterOption(name="normalize", value=normalize),
            ],
        )
    )
    return graph


def blackframe(graph: Stream, amount: Optional[int] = None, threshold: Optional[int] = None):
    """Detect frames that are (almost) black.

    :param int amount: percentage of the pixels that have to be below the threshold for the frame to be considered black
    :param int threshold: threshold below which a pixel value is considered black
    """
    graph.append(
        Filter(
            command="blackframe",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="amount", value=amount), FilterOption(name="threshold", value=threshold)],
        )
    )
    return graph


def asdr(
    graph: Stream,
):
    """Measure Audio Signal-to-Distortion Ratio."""
    graph.append(Filter(command="asdr", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def apsnr(
    graph: Stream,
):
    """Measure Audio Peak Signal-to-Noise Ratio."""
    graph.append(Filter(command="apsnr", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def asisdr(
    graph: Stream,
):
    """Measure Audio Scale-Invariant Signal-to-Distortion Ratio."""
    graph.append(Filter(command="asisdr", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def nlmeans_vulkan(
    graph: Stream,
    s: Optional[float] = None,
    p: Optional[int] = None,
    r: Optional[int] = None,
    t: Optional[int] = None,
    s1: Optional[float] = None,
    s2: Optional[float] = None,
    s3: Optional[float] = None,
    s4: Optional[float] = None,
    p1: Optional[int] = None,
    p2: Optional[int] = None,
    p3: Optional[int] = None,
    p4: Optional[int] = None,
):
    """Non-local means denoiser (Vulkan)

    :param float s: denoising strength for all components
    :param int p: patch size for all components
    :param int r: research window radius
    :param int t: parallelism
    :param float s1: denoising strength for component 1
    :param float s2: denoising strength for component 2
    :param float s3: denoising strength for component 3
    :param float s4: denoising strength for component 4
    :param int p1: patch size for component 1
    :param int p2: patch size for component 2
    :param int p3: patch size for component 3
    :param int p4: patch size for component 4
    """
    graph.append(
        Filter(
            command="nlmeans_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="s", value=s),
                FilterOption(name="p", value=p),
                FilterOption(name="r", value=r),
                FilterOption(name="t", value=t),
                FilterOption(name="s1", value=s1),
                FilterOption(name="s2", value=s2),
                FilterOption(name="s3", value=s3),
                FilterOption(name="s4", value=s4),
                FilterOption(name="p1", value=p1),
                FilterOption(name="p2", value=p2),
                FilterOption(name="p3", value=p3),
                FilterOption(name="p4", value=p4),
            ],
        )
    )
    return graph


def chromakey(
    graph: Stream,
    color: Optional[str] = None,
    similarity: Optional[float] = None,
    blend: Optional[float] = None,
    yuv: Optional[bool] = None,
):
    """Turns a certain color into transparency. Operates on YUV colors.

    :param str color: set the chromakey key color
    :param float similarity: set the chromakey similarity value
    :param float blend: set the chromakey key blend value
    :param bool yuv: color parameter is in yuv instead of rgb
    """
    graph.append(
        Filter(
            command="chromakey",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="color", value=color),
                FilterOption(name="similarity", value=similarity),
                FilterOption(name="blend", value=blend),
                FilterOption(name="yuv", value=yuv),
            ],
        )
    )
    return graph


def chromahold(
    graph: Stream,
    color: Optional[str] = None,
    similarity: Optional[float] = None,
    blend: Optional[float] = None,
    yuv: Optional[bool] = None,
):
    """Turns a certain color range into gray.

    :param str color: set the chromahold key color
    :param float similarity: set the chromahold similarity value
    :param float blend: set the chromahold blend value
    :param bool yuv: color parameter is in yuv instead of rgb
    """
    graph.append(
        Filter(
            command="chromahold",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="color", value=color),
                FilterOption(name="similarity", value=similarity),
                FilterOption(name="blend", value=blend),
                FilterOption(name="yuv", value=yuv),
            ],
        )
    )
    return graph


def dynaudnorm(
    graph: Stream,
    framelen: Optional[int] = None,
    gausssize: Optional[int] = None,
    peak: Optional[float] = None,
    maxgain: Optional[float] = None,
    targetrms: Optional[float] = None,
    coupling: Optional[bool] = None,
    correctdc: Optional[bool] = None,
    altboundary: Optional[bool] = None,
    compress: Optional[float] = None,
    threshold: Optional[float] = None,
    channels: Optional[str] = None,
    overlap: Optional[float] = None,
    curve: Optional[str] = None,
):
    """Dynamic Audio Normalizer.

    :param int framelen: set the frame length in msec
    :param int gausssize: set the filter size
    :param float peak: set the peak value
    :param float maxgain: set the max amplification
    :param float targetrms: set the target RMS
    :param bool coupling: set channel coupling
    :param bool correctdc: set DC correction
    :param bool altboundary: set alternative boundary mode
    :param float compress: set the compress factor
    :param float threshold: set the threshold value
    :param str channels: set channels to filter
    :param float overlap: set the frame overlap
    :param str curve: set the custom peak mapping curve
    """
    graph.append(
        Filter(
            command="dynaudnorm",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="framelen", value=framelen),
                FilterOption(name="gausssize", value=gausssize),
                FilterOption(name="peak", value=peak),
                FilterOption(name="maxgain", value=maxgain),
                FilterOption(name="targetrms", value=targetrms),
                FilterOption(name="coupling", value=coupling),
                FilterOption(name="correctdc", value=correctdc),
                FilterOption(name="altboundary", value=altboundary),
                FilterOption(name="compress", value=compress),
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="channels", value=channels),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="curve", value=curve),
            ],
        )
    )
    return graph


def cellauto(
    graph: Stream,
    filename: Optional[str] = None,
    pattern: Optional[str] = None,
    rate: Optional[str] = None,
    size: Optional[int] = None,
    rule: Optional[int] = None,
    random_fill_ratio: Optional[float] = None,
    random_seed: Optional[int] = None,
    scroll: Optional[bool] = None,
    start_full: Optional[bool] = None,
    stitch: Optional[bool] = None,
):
    """Create pattern generated by an elementary cellular automaton.

    :param str filename: read initial pattern from file
    :param str pattern: set initial pattern
    :param str rate: set video rate
    :param int size: set video size
    :param int rule: set rule
    :param float random_fill_ratio: set fill ratio for filling initial grid randomly
    :param int random_seed: set the seed for filling the initial grid randomly
    :param bool scroll: scroll pattern downward
    :param bool start_full: start filling the whole video
    :param bool stitch: stitch boundaries
    """
    graph.append(
        Filter(
            command="cellauto",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="filename", value=filename),
                FilterOption(name="pattern", value=pattern),
                FilterOption(name="rate", value=rate),
                FilterOption(name="size", value=size),
                FilterOption(name="rule", value=rule),
                FilterOption(name="random_fill_ratio", value=random_fill_ratio),
                FilterOption(name="random_seed", value=random_seed),
                FilterOption(name="scroll", value=scroll),
                FilterOption(name="start_full", value=start_full),
                FilterOption(name="stitch", value=stitch),
            ],
        )
    )
    return graph


def showcwt(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    scale: Optional[str] = None,
    iscale: Optional[str] = None,
    min: Optional[float] = None,
    max: Optional[float] = None,
    imin: Optional[float] = None,
    imax: Optional[float] = None,
    logb: Optional[float] = None,
    deviation: Optional[float] = None,
    pps: Optional[int] = None,
    mode: Optional[str] = None,
    slide: Optional[str] = None,
    direction: Optional[str] = None,
    bar: Optional[float] = None,
    rotation: Optional[float] = None,
):
    """Convert input audio to a CWT (Continuous Wavelet Transform) spectrum video output.

    :param int size: set video size
    :param str rate: set video rate
    :param str scale: set frequency scale
            possible values: linear, log, bark, mel, erbs, sqrt, cbrt, qdrt, fm
    :param str iscale: set intensity scale
            possible values: linear, log, sqrt, cbrt, qdrt
    :param float min: set minimum frequency
    :param float max: set maximum frequency
    :param float imin: set minimum intensity
    :param float imax: set maximum intensity
    :param float logb: set logarithmic basis
    :param float deviation: set frequency deviation
    :param int pps: set pixels per second
    :param str mode: set output mode
            possible values: magnitude, phase, magphase, channel, stereo
    :param str slide: set slide mode
            possible values: replace, scroll, frame
    :param str direction: set direction mode
            possible values: lr, rl, ud, du
    :param float bar: set bargraph ratio
    :param float rotation: set color rotation
    """
    graph.append(
        Filter(
            command="showcwt",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="scale", value=scale),
                FilterOption(name="iscale", value=iscale),
                FilterOption(name="min", value=min),
                FilterOption(name="max", value=max),
                FilterOption(name="imin", value=imin),
                FilterOption(name="imax", value=imax),
                FilterOption(name="logb", value=logb),
                FilterOption(name="deviation", value=deviation),
                FilterOption(name="pps", value=pps),
                FilterOption(name="mode", value=mode),
                FilterOption(name="slide", value=slide),
                FilterOption(name="direction", value=direction),
                FilterOption(name="bar", value=bar),
                FilterOption(name="rotation", value=rotation),
            ],
        )
    )
    return graph


def atempo(graph: Stream, tempo: Optional[float] = None):
    """Adjust audio tempo.

    :param float tempo: set tempo scale factor
    """
    graph.append(Filter(command="atempo", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="tempo", value=tempo)]))
    return graph


def color_vulkan(
    graph: Stream,
    color: Optional[str] = None,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    duration: Optional[int] = None,
    sar: Optional[int] = None,
    format: Optional[str] = None,
    out_range: Optional[str] = None,
):
    """Generate a constant color (Vulkan)

    :param str color: set color
    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param str format: Output video format (software format of hardware frames)
    :param str out_range: Output colour range (from 0 to 2) (default 0)
            possible values: full, limited, jpeg, mpeg, tv, pc
    """
    graph.append(
        Filter(
            command="color_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="color", value=color),
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="sar", value=sar),
                FilterOption(name="format", value=format),
                FilterOption(name="out_range", value=out_range),
            ],
        )
    )
    return graph


def midequalizer(graph: Stream, planes: Optional[int] = None):
    """Apply Midway Equalization.

    :param int planes: set planes
    """
    graph.append(Filter(command="midequalizer", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="planes", value=planes)]))
    return graph


def blend_vulkan(
    graph: Stream,
    c0_mode: Optional[str] = None,
    c1_mode: Optional[str] = None,
    c2_mode: Optional[str] = None,
    c3_mode: Optional[str] = None,
    all_mode: Optional[str] = None,
    c0_opacity: Optional[float] = None,
    c1_opacity: Optional[float] = None,
    c2_opacity: Optional[float] = None,
    c3_opacity: Optional[float] = None,
    all_opacity: Optional[float] = None,
):
    """Blend two video frames in Vulkan

    :param str c0_mode: set component #0 blend mode
            possible values: normal, multiply
    :param str c1_mode: set component #1 blend mode
            possible values: normal, multiply
    :param str c2_mode: set component #2 blend mode
            possible values: normal, multiply
    :param str c3_mode: set component #3 blend mode
            possible values: normal, multiply
    :param str all_mode: set blend mode for all components
            possible values: normal, multiply
    :param float c0_opacity: set color component #0 opacity
    :param float c1_opacity: set color component #1 opacity
    :param float c2_opacity: set color component #2 opacity
    :param float c3_opacity: set color component #3 opacity
    :param float all_opacity: set opacity for all color components
    """
    graph.append(
        Filter(
            command="blend_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="c0_mode", value=c0_mode),
                FilterOption(name="c1_mode", value=c1_mode),
                FilterOption(name="c2_mode", value=c2_mode),
                FilterOption(name="c3_mode", value=c3_mode),
                FilterOption(name="all_mode", value=all_mode),
                FilterOption(name="c0_opacity", value=c0_opacity),
                FilterOption(name="c1_opacity", value=c1_opacity),
                FilterOption(name="c2_opacity", value=c2_opacity),
                FilterOption(name="c3_opacity", value=c3_opacity),
                FilterOption(name="all_opacity", value=all_opacity),
            ],
        )
    )
    return graph


def a3dscope(
    graph: Stream,
    rate: Optional[str] = None,
    size: Optional[int] = None,
    fov: Optional[float] = None,
    roll: Optional[float] = None,
    pitch: Optional[float] = None,
    yaw: Optional[float] = None,
    xzoom: Optional[float] = None,
    yzoom: Optional[float] = None,
    zzoom: Optional[float] = None,
    xpos: Optional[float] = None,
    ypos: Optional[float] = None,
    zpos: Optional[float] = None,
    length: Optional[int] = None,
):
    """Convert input audio to 3d scope video output.

    :param str rate: set video rate
    :param int size: set video size
    :param float fov: set camera FoV
    :param float roll: set camera roll
    :param float pitch: set camera pitch
    :param float yaw: set camera yaw
    :param float xzoom: set camera zoom
    :param float yzoom: set camera zoom
    :param float zzoom: set camera zoom
    :param float xpos: set camera position
    :param float ypos: set camera position
    :param float zpos: set camera position
    :param int length: set length
    """
    graph.append(
        Filter(
            command="a3dscope",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rate", value=rate),
                FilterOption(name="size", value=size),
                FilterOption(name="fov", value=fov),
                FilterOption(name="roll", value=roll),
                FilterOption(name="pitch", value=pitch),
                FilterOption(name="yaw", value=yaw),
                FilterOption(name="xzoom", value=xzoom),
                FilterOption(name="yzoom", value=yzoom),
                FilterOption(name="zzoom", value=zzoom),
                FilterOption(name="xpos", value=xpos),
                FilterOption(name="ypos", value=ypos),
                FilterOption(name="zpos", value=zpos),
                FilterOption(name="length", value=length),
            ],
        )
    )
    return graph


def thumbnail(graph: Stream, n: Optional[int] = None, log: Optional[str] = None):
    """Select the most representative frame in a given sequence of consecutive frames.

    :param int n: set the frames batch size
    :param str log: force stats logging level
            possible values: quiet, info, verbose
    """
    graph.append(
        Filter(
            command="thumbnail",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="n", value=n), FilterOption(name="log", value=log)],
        )
    )
    return graph


def deblock(
    graph: Stream,
    filter: Optional[str] = None,
    block: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    delta: Optional[float] = None,
    planes: Optional[int] = None,
):
    """Deblock video.

    :param str filter: set type of filter
            possible values: weak, strong
    :param int block: set size of block
    :param float alpha: set 1st detection threshold
    :param float beta: set 2nd detection threshold
    :param float gamma: set 3rd detection threshold
    :param float delta: set 4th detection threshold
    :param int planes: set planes to filter
    """
    graph.append(
        Filter(
            command="deblock",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="filter", value=filter),
                FilterOption(name="block", value=block),
                FilterOption(name="alpha", value=alpha),
                FilterOption(name="beta", value=beta),
                FilterOption(name="gamma", value=gamma),
                FilterOption(name="delta", value=delta),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def crossfeed(
    graph: Stream,
    strength: Optional[float] = None,
    range: Optional[float] = None,
    slope: Optional[float] = None,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    block_size: Optional[int] = None,
):
    """Apply headphone crossfeed filter.

    :param float strength: set crossfeed strength
    :param float range: set soundstage wideness
    :param float slope: set curve slope
    :param float level_in: set level in
    :param float level_out: set level out
    :param int block_size: set the block size
    """
    graph.append(
        Filter(
            command="crossfeed",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="strength", value=strength),
                FilterOption(name="range", value=range),
                FilterOption(name="slope", value=slope),
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="block_size", value=block_size),
            ],
        )
    )
    return graph


def deflicker(graph: Stream, size: Optional[int] = None, mode: Optional[str] = None, bypass: Optional[bool] = None):
    """Remove temporal frame luminance variations.

    :param int size: set how many frames to use
    :param str mode: set how to smooth luminance
            possible values: am, gm, hm, qm, cm, pm, median
    :param bool bypass: leave frames unchanged
    """
    graph.append(
        Filter(
            command="deflicker",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="mode", value=mode),
                FilterOption(name="bypass", value=bypass),
            ],
        )
    )
    return graph


def showwaves(
    graph: Stream,
    size: Optional[int] = None,
    mode: Optional[str] = None,
    n: Optional[int] = None,
    rate: Optional[str] = None,
    split_channels: Optional[bool] = None,
    colors: Optional[str] = None,
    scale: Optional[str] = None,
    draw: Optional[str] = None,
):
    """Convert input audio to a video output.

    :param int size: set video size
    :param str mode: select display mode
            possible values: point, line, p2p, cline
    :param int n: set how many samples to show in the same point
    :param str rate: set video rate
    :param bool split_channels: draw channels separately
    :param str colors: set channels colors
    :param str scale: set amplitude scale
            possible values: lin, log, sqrt, cbrt
    :param str draw: set draw mode
            possible values: scale, full
    """
    graph.append(
        Filter(
            command="showwaves",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="mode", value=mode),
                FilterOption(name="n", value=n),
                FilterOption(name="rate", value=rate),
                FilterOption(name="split_channels", value=split_channels),
                FilterOption(name="colors", value=colors),
                FilterOption(name="scale", value=scale),
                FilterOption(name="draw", value=draw),
            ],
        )
    )
    return graph


def showwavespic(
    graph: Stream,
    size: Optional[int] = None,
    split_channels: Optional[bool] = None,
    colors: Optional[str] = None,
    scale: Optional[str] = None,
    draw: Optional[str] = None,
    filter: Optional[str] = None,
):
    """Convert input audio to a video output single picture.

    :param int size: set video size
    :param bool split_channels: draw channels separately
    :param str colors: set channels colors
    :param str scale: set amplitude scale
            possible values: lin, log, sqrt, cbrt
    :param str draw: set draw mode
            possible values: scale, full
    :param str filter: set filter mode
            possible values: average, peak
    """
    graph.append(
        Filter(
            command="showwavespic",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="split_channels", value=split_channels),
                FilterOption(name="colors", value=colors),
                FilterOption(name="scale", value=scale),
                FilterOption(name="draw", value=draw),
                FilterOption(name="filter", value=filter),
            ],
        )
    )
    return graph


def aemphasis(
    graph: Stream,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    mode: Optional[str] = None,
    type: Optional[str] = None,
):
    """Audio emphasis.

    :param float level_in: set input gain
    :param float level_out: set output gain
    :param str mode: set filter mode
            possible values: reproduction, production
    :param str type: set filter type
            possible values: col, emi, bsi, riaa, cd, 50fm, 75fm, 50kf, 75kf
    """
    graph.append(
        Filter(
            command="aemphasis",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="mode", value=mode),
                FilterOption(name="type", value=type),
            ],
        )
    )
    return graph


def tile(
    graph: Stream,
    layout: Optional[int] = None,
    nb_frames: Optional[int] = None,
    margin: Optional[int] = None,
    padding: Optional[int] = None,
    color: Optional[str] = None,
    overlap: Optional[int] = None,
    init_padding: Optional[int] = None,
):
    """Tile several successive frames together.

    :param int layout: set grid size
    :param int nb_frames: set maximum number of frame to render
    :param int margin: set outer border margin in pixels
    :param int padding: set inner border thickness in pixels
    :param str color: set the color of the unused area
    :param int overlap: set how many frames to overlap for each render
    :param int init_padding: set how many frames to initially pad
    """
    graph.append(
        Filter(
            command="tile",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="layout", value=layout),
                FilterOption(name="nb_frames", value=nb_frames),
                FilterOption(name="margin", value=margin),
                FilterOption(name="padding", value=padding),
                FilterOption(name="color", value=color),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="init_padding", value=init_padding),
            ],
        )
    )
    return graph


def colorkey_opencl(graph: Stream, color: Optional[str] = None, similarity: Optional[float] = None, blend: Optional[float] = None):
    """Turns a certain color into transparency. Operates on RGB colors.

    :param str color: set the colorkey key color
    :param float similarity: set the colorkey similarity value
    :param float blend: set the colorkey key blend value
    """
    graph.append(
        Filter(
            command="colorkey_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="color", value=color),
                FilterOption(name="similarity", value=similarity),
                FilterOption(name="blend", value=blend),
            ],
        )
    )
    return graph


def fillborders(
    graph: Stream,
    left: Optional[int] = None,
    right: Optional[int] = None,
    top: Optional[int] = None,
    bottom: Optional[int] = None,
    mode: Optional[str] = None,
    color: Optional[str] = None,
):
    """Fill borders of the input video.

    :param int left: set the left fill border
    :param int right: set the right fill border
    :param int top: set the top fill border
    :param int bottom: set the bottom fill border
    :param str mode: set the fill borders mode
            possible values: smear, mirror, fixed, reflect, wrap, fade, margins
    :param str color: set the color for the fixed/fade mode
    """
    graph.append(
        Filter(
            command="fillborders",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="left", value=left),
                FilterOption(name="right", value=right),
                FilterOption(name="top", value=top),
                FilterOption(name="bottom", value=bottom),
                FilterOption(name="mode", value=mode),
                FilterOption(name="color", value=color),
            ],
        )
    )
    return graph


def maskedmerge(graph: Stream, planes: Optional[int] = None):
    """Merge first stream with second stream using third stream as mask.

    :param int planes: set planes
    """
    graph.append(Filter(command="maskedmerge", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="planes", value=planes)]))
    return graph


def bwdif(graph: Stream, mode: Optional[str] = None, parity: Optional[str] = None, deint: Optional[str] = None):
    """Deinterlace the input image.

    :param str mode: specify the interlacing mode
            possible values: send_frame, send_field
    :param str parity: specify the assumed picture field parity
            possible values: tff, bff, auto
    :param str deint: specify which frames to deinterlace
            possible values: all, interlaced
    """
    graph.append(
        Filter(
            command="bwdif",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="parity", value=parity),
                FilterOption(name="deint", value=deint),
            ],
        )
    )
    return graph


def avsynctest(
    graph: Stream,
    size: Optional[int] = None,
    framerate: Optional[str] = None,
    samplerate: Optional[int] = None,
    amplitude: Optional[float] = None,
    period: Optional[int] = None,
    delay: Optional[int] = None,
    cycle: Optional[bool] = None,
    duration: Optional[int] = None,
    fg: Optional[str] = None,
    bg: Optional[str] = None,
    ag: Optional[str] = None,
):
    """Generate an Audio Video Sync Test.

    :param int size: set frame size
    :param str framerate: set frame rate
    :param int samplerate: set sample rate
    :param float amplitude: set beep amplitude
    :param int period: set beep period
    :param int delay: set flash delay
    :param bool cycle: set delay cycle
    :param int duration: set duration
    :param str fg: set foreground color
    :param str bg: set background color
    :param str ag: set additional color
    """
    graph.append(
        Filter(
            command="avsynctest",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="framerate", value=framerate),
                FilterOption(name="samplerate", value=samplerate),
                FilterOption(name="amplitude", value=amplitude),
                FilterOption(name="period", value=period),
                FilterOption(name="delay", value=delay),
                FilterOption(name="cycle", value=cycle),
                FilterOption(name="duration", value=duration),
                FilterOption(name="fg", value=fg),
                FilterOption(name="bg", value=bg),
                FilterOption(name="ag", value=ag),
            ],
        )
    )
    return graph


def dnn_classify(
    graph: Stream,
    dnn_backend: Optional[int] = None,
    model: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
    backend_configs: Optional[str] = None,
    options: Optional[str] = None,
    _async: Optional[bool] = None,
    confidence: Optional[float] = None,
    labels: Optional[str] = None,
    target: Optional[str] = None,
):
    """Apply DNN classify filter to the input.

    :param int dnn_backend: DNN backend
    :param str model: path to model file
    :param str input: input name of the model
    :param str output: output name of the model
    :param str backend_configs: backend configs
    :param str options: backend configs (deprecated, use backend_configs)
    :param bool async: use DNN async inference (ignored, use backend_configs='async=1')
    :param float confidence: threshold of confidence
    :param str labels: path to labels file
    :param str target: which one to be classified
    """
    graph.append(
        Filter(
            command="dnn_classify",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="dnn_backend", value=dnn_backend),
                FilterOption(name="model", value=model),
                FilterOption(name="input", value=input),
                FilterOption(name="output", value=output),
                FilterOption(name="backend_configs", value=backend_configs),
                FilterOption(name="options", value=options),
                FilterOption(name="async", value=_async),
                FilterOption(name="confidence", value=confidence),
                FilterOption(name="labels", value=labels),
                FilterOption(name="target", value=target),
            ],
        )
    )
    return graph


def blurdetect(
    graph: Stream,
    high: Optional[float] = None,
    low: Optional[float] = None,
    radius: Optional[int] = None,
    block_pct: Optional[int] = None,
    block_width: Optional[int] = None,
    block_height: Optional[int] = None,
    planes: Optional[int] = None,
):
    """Blurdetect filter.

    :param float high: set high threshold
    :param float low: set low threshold
    :param int radius: search radius for maxima detection
    :param int block_pct: block pooling threshold when calculating blurriness
    :param int block_width: block size for block-based abbreviation of blurriness
    :param int block_height: block size for block-based abbreviation of blurriness
    :param int planes: set planes to filter
    """
    graph.append(
        Filter(
            command="blurdetect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="high", value=high),
                FilterOption(name="low", value=low),
                FilterOption(name="radius", value=radius),
                FilterOption(name="block_pct", value=block_pct),
                FilterOption(name="block_width", value=block_width),
                FilterOption(name="block_height", value=block_height),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def grayworld(
    graph: Stream,
):
    """Adjust white balance using LAB gray world algorithm"""
    graph.append(Filter(command="grayworld", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def cover_rect(graph: Stream, cover: Optional[str] = None, mode: Optional[str] = None):
    """Find and cover a user specified object.

    :param str cover: cover bitmap filename
    :param str mode: set removal mode
            possible values: cover, blur
    """
    graph.append(
        Filter(
            command="cover_rect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="cover", value=cover), FilterOption(name="mode", value=mode)],
        )
    )
    return graph


def channelmap(graph: Stream, map: Optional[str] = None, channel_layout: Optional[str] = None):
    """Remap audio channels.

    :param str map: A comma-separated list of input channel numbers in output order.
    :param str channel_layout: Output channel layout.
    """
    graph.append(
        Filter(
            command="channelmap",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="map", value=map), FilterOption(name="channel_layout", value=channel_layout)],
        )
    )
    return graph


def dedot(
    graph: Stream,
    m: Optional[str] = None,
    lt: Optional[float] = None,
    tl: Optional[float] = None,
    tc: Optional[float] = None,
    ct: Optional[float] = None,
):
    """Reduce cross-luminance and cross-color.

    :param str m: set filtering mode
            possible values: dotcrawl, rainbows
    :param float lt: set spatial luma threshold
    :param float tl: set tolerance for temporal luma
    :param float tc: set tolerance for chroma temporal variation
    :param float ct: set temporal chroma threshold
    """
    graph.append(
        Filter(
            command="dedot",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="m", value=m),
                FilterOption(name="lt", value=lt),
                FilterOption(name="tl", value=tl),
                FilterOption(name="tc", value=tc),
                FilterOption(name="ct", value=ct),
            ],
        )
    )
    return graph


def adeclick(
    graph: Stream,
    window: Optional[float] = None,
    overlap: Optional[float] = None,
    arorder: Optional[float] = None,
    threshold: Optional[float] = None,
    burst: Optional[float] = None,
    method: Optional[str] = None,
):
    """Remove impulsive noise from input audio.

    :param float window: set window size
    :param float overlap: set window overlap
    :param float arorder: set autoregression order
    :param float threshold: set threshold
    :param float burst: set burst fusion
    :param str method: set overlap method
            possible values: add, a, save, s
    """
    graph.append(
        Filter(
            command="adeclick",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="window", value=window),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="arorder", value=arorder),
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="burst", value=burst),
                FilterOption(name="method", value=method),
            ],
        )
    )
    return graph


def adeclip(
    graph: Stream,
    window: Optional[float] = None,
    overlap: Optional[float] = None,
    arorder: Optional[float] = None,
    threshold: Optional[float] = None,
    hsize: Optional[int] = None,
    method: Optional[str] = None,
):
    """Remove clipping from input audio.

    :param float window: set window size
    :param float overlap: set window overlap
    :param float arorder: set autoregression order
    :param float threshold: set threshold
    :param int hsize: set histogram size
    :param str method: set overlap method
            possible values: add, a, save, s
    """
    graph.append(
        Filter(
            command="adeclip",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="window", value=window),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="arorder", value=arorder),
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="hsize", value=hsize),
                FilterOption(name="method", value=method),
            ],
        )
    )
    return graph


def settb(graph: Stream, expr: Optional[str] = None):
    """Set timebase for the video output link.

    :param str expr: set expression determining the output timebase
    """
    graph.append(Filter(command="settb", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="expr", value=expr)]))
    return graph


def asettb(graph: Stream, expr: Optional[str] = None):
    """Set timebase for the audio output link.

    :param str expr: set expression determining the output timebase
    """
    graph.append(Filter(command="asettb", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="expr", value=expr)]))
    return graph


def xfade_opencl(
    graph: Stream,
    transition: Optional[str] = None,
    source: Optional[str] = None,
    kernel: Optional[str] = None,
    duration: Optional[int] = None,
    offset: Optional[int] = None,
):
    """Cross fade one video with another video.

    :param str transition: set cross fade transition
            possible values: custom, fade, wipeleft, wiperight, wipeup, wipedown, slideleft, slideright, slideup, slidedown
    :param str source: set OpenCL program source file for custom transition
    :param str kernel: set kernel name in program file for custom transition
    :param int duration: set cross fade duration
    :param int offset: set cross fade start relative to first input stream
    """
    graph.append(
        Filter(
            command="xfade_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="transition", value=transition),
                FilterOption(name="source", value=source),
                FilterOption(name="kernel", value=kernel),
                FilterOption(name="duration", value=duration),
                FilterOption(name="offset", value=offset),
            ],
        )
    )
    return graph


def idet(
    graph: Stream,
    intl_thres: Optional[float] = None,
    prog_thres: Optional[float] = None,
    rep_thres: Optional[float] = None,
    half_life: Optional[float] = None,
    analyze_interlaced_flag: Optional[int] = None,
):
    """Interlace detect Filter.

    :param float intl_thres: set interlacing threshold
    :param float prog_thres: set progressive threshold
    :param float rep_thres: set repeat threshold
    :param float half_life: half life of cumulative statistics
    :param int analyze_interlaced_flag: set number of frames to use to determine if the interlace flag is accurate
    """
    graph.append(
        Filter(
            command="idet",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="intl_thres", value=intl_thres),
                FilterOption(name="prog_thres", value=prog_thres),
                FilterOption(name="rep_thres", value=rep_thres),
                FilterOption(name="half_life", value=half_life),
                FilterOption(name="analyze_interlaced_flag", value=analyze_interlaced_flag),
            ],
        )
    )
    return graph


def weave(graph: Stream, first_field: Optional[str] = None):
    """Weave input video fields into frames.

    :param str first_field: set first field
            possible values: top, t, bottom, b
    """
    graph.append(Filter(command="weave", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="first_field", value=first_field)]))
    return graph


def doubleweave(
    graph: Stream,
):
    """Weave input video fields into double number of frames."""
    graph.append(Filter(command="doubleweave", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def pad(
    graph: Stream,
    width: Optional[str] = None,
    height: Optional[str] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    eval: Optional[str] = None,
    aspect: Optional[int] = None,
):
    """Pad the input video.

    :param str width: set the pad area width expression
    :param str height: set the pad area height expression
    :param str x: set the x offset expression for the input image position
    :param str y: set the y offset expression for the input image position
    :param str color: set the color of the padded area border
    :param str eval: specify when to evaluate expressions
            possible values: init, frame
    :param int aspect: pad to fit an aspect instead of a resolution
    """
    graph.append(
        Filter(
            command="pad",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="width", value=width),
                FilterOption(name="height", value=height),
                FilterOption(name="x", value=x),
                FilterOption(name="y", value=y),
                FilterOption(name="color", value=color),
                FilterOption(name="eval", value=eval),
                FilterOption(name="aspect", value=aspect),
            ],
        )
    )
    return graph


def amultiply(
    graph: Stream,
):
    """Multiply two audio streams."""
    graph.append(Filter(command="amultiply", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def compand(
    graph: Stream,
    attacks: Optional[str] = None,
    decays: Optional[str] = None,
    points: Optional[str] = None,
    soft_knee: Optional[float] = None,
    gain: Optional[float] = None,
    volume: Optional[float] = None,
    delay: Optional[float] = None,
):
    """Compress or expand audio dynamic range.

    :param str attacks: set time over which increase of volume is determined
    :param str decays: set time over which decrease of volume is determined
    :param str points: set points of transfer function
    :param float soft-knee: set soft-knee
    :param float gain: set output gain
    :param float volume: set initial volume
    :param float delay: set delay for samples before sending them to volume adjuster
    """
    graph.append(
        Filter(
            command="compand",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="attacks", value=attacks),
                FilterOption(name="decays", value=decays),
                FilterOption(name="points", value=points),
                FilterOption(name="soft-knee", value=soft_knee),
                FilterOption(name="gain", value=gain),
                FilterOption(name="volume", value=volume),
                FilterOption(name="delay", value=delay),
            ],
        )
    )
    return graph


def life(
    graph: Stream,
    filename: Optional[str] = None,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    rule: Optional[str] = None,
    random_fill_ratio: Optional[float] = None,
    random_seed: Optional[int] = None,
    stitch: Optional[bool] = None,
    mold: Optional[int] = None,
    life_color: Optional[str] = None,
    death_color: Optional[str] = None,
    mold_color: Optional[str] = None,
):
    """Create life.

    :param str filename: set source file
    :param int size: set video size
    :param str rate: set video rate
    :param str rule: set rule
    :param float random_fill_ratio: set fill ratio for filling initial grid randomly
    :param int random_seed: set the seed for filling the initial grid randomly
    :param bool stitch: stitch boundaries
    :param int mold: set mold speed for dead cells
    :param str life_color: set life color
    :param str death_color: set death color
    :param str mold_color: set mold color
    """
    graph.append(
        Filter(
            command="life",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="filename", value=filename),
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="rule", value=rule),
                FilterOption(name="random_fill_ratio", value=random_fill_ratio),
                FilterOption(name="random_seed", value=random_seed),
                FilterOption(name="stitch", value=stitch),
                FilterOption(name="mold", value=mold),
                FilterOption(name="life_color", value=life_color),
                FilterOption(name="death_color", value=death_color),
                FilterOption(name="mold_color", value=mold_color),
            ],
        )
    )
    return graph


def speechnorm(
    graph: Stream,
    peak: Optional[float] = None,
    expansion: Optional[float] = None,
    compression: Optional[float] = None,
    threshold: Optional[float] = None,
    _raise: Optional[float] = None,
    fall: Optional[float] = None,
    channels: Optional[str] = None,
    invert: Optional[bool] = None,
    link: Optional[bool] = None,
    rms: Optional[float] = None,
):
    """Speech Normalizer.

    :param float peak: set the peak value
    :param float expansion: set the max expansion factor
    :param float compression: set the max compression factor
    :param float threshold: set the threshold value
    :param float raise: set the expansion raising amount
    :param float fall: set the compression raising amount
    :param str channels: set channels to filter
    :param bool invert: set inverted filtering
    :param bool link: set linked channels filtering
    :param float rms: set the RMS value
    """
    graph.append(
        Filter(
            command="speechnorm",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="peak", value=peak),
                FilterOption(name="expansion", value=expansion),
                FilterOption(name="compression", value=compression),
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="raise", value=_raise),
                FilterOption(name="fall", value=fall),
                FilterOption(name="channels", value=channels),
                FilterOption(name="invert", value=invert),
                FilterOption(name="link", value=link),
                FilterOption(name="rms", value=rms),
            ],
        )
    )
    return graph


def amerge(graph: Stream, inputs: Optional[int] = None):
    """Merge two or more audio streams into a single multi-channel stream.

    :param int inputs: specify the number of inputs
    """
    graph.append(Filter(command="amerge", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="inputs", value=inputs)]))
    return graph


def xmedian(graph: Stream, inputs: Optional[int] = None, planes: Optional[int] = None, percentile: Optional[float] = None):
    """Pick median pixels from several video inputs.

    :param int inputs: set number of inputs
    :param int planes: set planes to filter
    :param float percentile: set percentile
    """
    graph.append(
        Filter(
            command="xmedian",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="inputs", value=inputs),
                FilterOption(name="planes", value=planes),
                FilterOption(name="percentile", value=percentile),
            ],
        )
    )
    return graph


def tmedian(graph: Stream, radius: Optional[int] = None, planes: Optional[int] = None, percentile: Optional[float] = None):
    """Pick median pixels from successive frames.

    :param int radius: set median filter radius
    :param int planes: set planes to filter
    :param float percentile: set percentile
    """
    graph.append(
        Filter(
            command="tmedian",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="radius", value=radius),
                FilterOption(name="planes", value=planes),
                FilterOption(name="percentile", value=percentile),
            ],
        )
    )
    return graph


def detelecine(graph: Stream, first_field: Optional[str] = None, pattern: Optional[str] = None, start_frame: Optional[int] = None):
    """Apply an inverse telecine pattern.

    :param str first_field: select first field
            possible values: top, t, bottom, b
    :param str pattern: pattern that describe for how many fields a frame is to be displayed
    :param int start_frame: position of first frame with respect to the pattern if stream is cut
    """
    graph.append(
        Filter(
            command="detelecine",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="first_field", value=first_field),
                FilterOption(name="pattern", value=pattern),
                FilterOption(name="start_frame", value=start_frame),
            ],
        )
    )
    return graph


def framepack(graph: Stream, format: Optional[str] = None):
    """Generate a frame packed stereoscopic video.

    :param str format: Frame pack output format
            possible values: sbs, tab, frameseq, lines, columns
    """
    graph.append(Filter(command="framepack", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="format", value=format)]))
    return graph


def asetrate(graph: Stream, sample_rate: Optional[int] = None):
    """Change the sample rate without altering the data.

    :param int sample_rate: set the sample rate
    """
    graph.append(Filter(command="asetrate", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="sample_rate", value=sample_rate)]))
    return graph


def stereowiden(
    graph: Stream,
    delay: Optional[float] = None,
    feedback: Optional[float] = None,
    crossfeed: Optional[float] = None,
    drymix: Optional[float] = None,
):
    """Apply stereo widening effect.

    :param float delay: set delay time
    :param float feedback: set feedback gain
    :param float crossfeed: set cross feed
    :param float drymix: set dry-mix
    """
    graph.append(
        Filter(
            command="stereowiden",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="delay", value=delay),
                FilterOption(name="feedback", value=feedback),
                FilterOption(name="crossfeed", value=crossfeed),
                FilterOption(name="drymix", value=drymix),
            ],
        )
    )
    return graph


def shuffleplanes(
    graph: Stream, map0: Optional[int] = None, map1: Optional[int] = None, map2: Optional[int] = None, map3: Optional[int] = None
):
    """Shuffle video planes.

    :param int map0: Index of the input plane to be used as the first output plane
    :param int map1: Index of the input plane to be used as the second output plane
    :param int map2: Index of the input plane to be used as the third output plane
    :param int map3: Index of the input plane to be used as the fourth output plane
    """
    graph.append(
        Filter(
            command="shuffleplanes",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="map0", value=map0),
                FilterOption(name="map1", value=map1),
                FilterOption(name="map2", value=map2),
                FilterOption(name="map3", value=map3),
            ],
        )
    )
    return graph


def mptestsrc(
    graph: Stream, rate: Optional[str] = None, duration: Optional[int] = None, test: Optional[str] = None, max_frames: Optional[int] = None
):
    """Generate various test pattern.

    :param str rate: set video rate
    :param int duration: set video duration
    :param str test: set test to perform
            possible values: dc_luma, dc_chroma, freq_luma, freq_chroma, amp_luma, amp_chroma, cbp, mv, ring1, ring2, all
    :param int max_frames: Set the maximum number of frames generated for each test
    """
    graph.append(
        Filter(
            command="mptestsrc",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rate", value=rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="test", value=test),
                FilterOption(name="max_frames", value=max_frames),
            ],
        )
    )
    return graph


def xbr(graph: Stream, n: Optional[int] = None):
    """Scale the input using xBR algorithm.

    :param int n: set scale factor
    """
    graph.append(Filter(command="xbr", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="n", value=n)]))
    return graph


def anull(
    graph: Stream,
):
    """Pass the source unchanged to the output."""
    graph.append(Filter(command="anull", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def hstack(
    graph: Stream,
):
    """Stack video inputs horizontally."""
    graph.append(Filter(command="hstack", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def vstack(
    graph: Stream,
):
    """Stack video inputs vertically."""
    graph.append(Filter(command="vstack", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def xstack(
    graph: Stream,
    inputs: Optional[int] = None,
    layout: Optional[str] = None,
    grid: Optional[int] = None,
    shortest: Optional[bool] = None,
    fill: Optional[str] = None,
):
    """Stack video inputs into custom layout.

    :param int inputs: set number of inputs
    :param str layout: set custom layout
    :param int grid: set fixed size grid layout
    :param bool shortest: force termination when the shortest input terminates
    :param str fill: set the color for unused pixels
    """
    graph.append(
        Filter(
            command="xstack",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="inputs", value=inputs),
                FilterOption(name="layout", value=layout),
                FilterOption(name="grid", value=grid),
                FilterOption(name="shortest", value=shortest),
                FilterOption(name="fill", value=fill),
            ],
        )
    )
    return graph


def scale_vulkan(
    graph: Stream,
    w: Optional[str] = None,
    h: Optional[str] = None,
    scaler: Optional[str] = None,
    format: Optional[str] = None,
    out_range: Optional[str] = None,
):
    """Scale Vulkan frames

    :param str w: Output video width
    :param str h: Output video height
    :param str scaler: Scaler function
            possible values: bilinear, nearest
    :param str format: Output video format (software format of hardware frames)
    :param str out_range: Output colour range (from 0 to 2) (default 0)
            possible values: full, limited, jpeg, mpeg, tv, pc
    """
    graph.append(
        Filter(
            command="scale_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="scaler", value=scaler),
                FilterOption(name="format", value=format),
                FilterOption(name="out_range", value=out_range),
            ],
        )
    )
    return graph


def uspp(
    graph: Stream,
    quality: Optional[int] = None,
    qp: Optional[int] = None,
    use_bframe_qp: Optional[bool] = None,
    codec: Optional[str] = None,
):
    """Apply Ultra Simple / Slow Post-processing filter.

    :param int quality: set quality
    :param int qp: force a constant quantizer parameter
    :param bool use_bframe_qp: use B-frames' QP
    :param str codec: Codec name
    """
    graph.append(
        Filter(
            command="uspp",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="quality", value=quality),
                FilterOption(name="qp", value=qp),
                FilterOption(name="use_bframe_qp", value=use_bframe_qp),
                FilterOption(name="codec", value=codec),
            ],
        )
    )
    return graph


def acrusher(
    graph: Stream,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    bits: Optional[float] = None,
    mix: Optional[float] = None,
    mode: Optional[str] = None,
    dc: Optional[float] = None,
    aa: Optional[float] = None,
    samples: Optional[float] = None,
    lfo: Optional[bool] = None,
    lforange: Optional[float] = None,
    lforate: Optional[float] = None,
):
    """Reduce audio bit resolution.

    :param float level_in: set level in
    :param float level_out: set level out
    :param float bits: set bit reduction
    :param float mix: set mix
    :param str mode: set mode
            possible values: lin, log
    :param float dc: set DC
    :param float aa: set anti-aliasing
    :param float samples: set sample reduction
    :param bool lfo: enable LFO
    :param float lforange: set LFO depth
    :param float lforate: set LFO rate
    """
    graph.append(
        Filter(
            command="acrusher",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="bits", value=bits),
                FilterOption(name="mix", value=mix),
                FilterOption(name="mode", value=mode),
                FilterOption(name="dc", value=dc),
                FilterOption(name="aa", value=aa),
                FilterOption(name="samples", value=samples),
                FilterOption(name="lfo", value=lfo),
                FilterOption(name="lforange", value=lforange),
                FilterOption(name="lforate", value=lforate),
            ],
        )
    )
    return graph


def showcqt(
    graph: Stream,
    size: Optional[int] = None,
    fps: Optional[str] = None,
    bar_h: Optional[int] = None,
    axis_h: Optional[int] = None,
    sono_h: Optional[int] = None,
    fullhd: Optional[bool] = None,
    sono_v: Optional[str] = None,
    bar_v: Optional[str] = None,
    sono_g: Optional[float] = None,
    bar_g: Optional[float] = None,
    bar_t: Optional[float] = None,
    timeclamp: Optional[float] = None,
    attack: Optional[float] = None,
    basefreq: Optional[float] = None,
    endfreq: Optional[float] = None,
    coeffclamp: Optional[float] = None,
    tlength: Optional[str] = None,
    count: Optional[int] = None,
    fcount: Optional[int] = None,
    fontfile: Optional[str] = None,
    font: Optional[str] = None,
    fontcolor: Optional[str] = None,
    axisfile: Optional[str] = None,
    axis: Optional[bool] = None,
    csp: Optional[str] = None,
    cscheme: Optional[str] = None,
):
    """Convert input audio to a CQT (Constant/Clamped Q Transform) spectrum video output.

    :param int size: set video size
    :param str fps: set video rate
    :param int bar_h: set bargraph height
    :param int axis_h: set axis height
    :param int sono_h: set sonogram height
    :param bool fullhd: set fullhd size
    :param str sono_v: set sonogram volume
    :param str bar_v: set bargraph volume
    :param float sono_g: set sonogram gamma
    :param float bar_g: set bargraph gamma
    :param float bar_t: set bar transparency
    :param float timeclamp: set timeclamp
    :param float attack: set attack time
    :param float basefreq: set base frequency
    :param float endfreq: set end frequency
    :param float coeffclamp: set coeffclamp
    :param str tlength: set tlength
    :param int count: set transform count
    :param int fcount: set frequency count
    :param str fontfile: set axis font file
    :param str font: set axis font
    :param str fontcolor: set font color
    :param str axisfile: set axis image
    :param bool axis: draw axis
    :param str csp: set color space
            possible values: unspecified, bt709, fcc, bt470bg, smpte170m, smpte240m, bt2020ncl
    :param str cscheme: set color scheme
    """
    graph.append(
        Filter(
            command="showcqt",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="fps", value=fps),
                FilterOption(name="bar_h", value=bar_h),
                FilterOption(name="axis_h", value=axis_h),
                FilterOption(name="sono_h", value=sono_h),
                FilterOption(name="fullhd", value=fullhd),
                FilterOption(name="sono_v", value=sono_v),
                FilterOption(name="bar_v", value=bar_v),
                FilterOption(name="sono_g", value=sono_g),
                FilterOption(name="bar_g", value=bar_g),
                FilterOption(name="bar_t", value=bar_t),
                FilterOption(name="timeclamp", value=timeclamp),
                FilterOption(name="attack", value=attack),
                FilterOption(name="basefreq", value=basefreq),
                FilterOption(name="endfreq", value=endfreq),
                FilterOption(name="coeffclamp", value=coeffclamp),
                FilterOption(name="tlength", value=tlength),
                FilterOption(name="count", value=count),
                FilterOption(name="fcount", value=fcount),
                FilterOption(name="fontfile", value=fontfile),
                FilterOption(name="font", value=font),
                FilterOption(name="fontcolor", value=fontcolor),
                FilterOption(name="axisfile", value=axisfile),
                FilterOption(name="axis", value=axis),
                FilterOption(name="csp", value=csp),
                FilterOption(name="cscheme", value=cscheme),
            ],
        )
    )
    return graph


def hflip_vulkan(
    graph: Stream,
):
    """Horizontally flip the input video in Vulkan"""
    graph.append(Filter(command="hflip_vulkan", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def vflip_vulkan(
    graph: Stream,
):
    """Vertically flip the input video in Vulkan"""
    graph.append(Filter(command="vflip_vulkan", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def flip_vulkan(
    graph: Stream,
):
    """Flip both horizontally and vertically"""
    graph.append(Filter(command="flip_vulkan", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def adelay(graph: Stream, delays: Optional[str] = None, all: Optional[bool] = None):
    """Delay one or more audio channels.

    :param str delays: set list of delays for each channel
    :param bool all: use last available delay for remained channels
    """
    graph.append(
        Filter(
            command="adelay",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="delays", value=delays), FilterOption(name="all", value=all)],
        )
    )
    return graph


def aevalsrc(
    graph: Stream,
    exprs: Optional[str] = None,
    nb_samples: Optional[int] = None,
    sample_rate: Optional[str] = None,
    duration: Optional[int] = None,
    channel_layout: Optional[str] = None,
):
    """Generate an audio signal generated by an expression.

    :param str exprs: set the '|'-separated list of channels expressions
    :param int nb_samples: set the number of samples per requested frame
    :param str sample_rate: set the sample rate
    :param int duration: set audio duration
    :param str channel_layout: set channel layout
    """
    graph.append(
        Filter(
            command="aevalsrc",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="exprs", value=exprs),
                FilterOption(name="nb_samples", value=nb_samples),
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="duration", value=duration),
                FilterOption(name="channel_layout", value=channel_layout),
            ],
        )
    )
    return graph


def aeval(graph: Stream, exprs: Optional[str] = None, channel_layout: Optional[str] = None):
    """Filter audio signal according to a specified expression.

    :param str exprs: set the '|'-separated list of channels expressions
    :param str channel_layout: set channel layout
    """
    graph.append(
        Filter(
            command="aeval",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="exprs", value=exprs), FilterOption(name="channel_layout", value=channel_layout)],
        )
    )
    return graph


def greyedge(graph: Stream, difford: Optional[int] = None, minknorm: Optional[int] = None, sigma: Optional[float] = None):
    """Estimates scene illumination by grey edge assumption.

    :param int difford: set differentiation order
    :param int minknorm: set Minkowski norm
    :param float sigma: set sigma
    """
    graph.append(
        Filter(
            command="greyedge",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="difford", value=difford),
                FilterOption(name="minknorm", value=minknorm),
                FilterOption(name="sigma", value=sigma),
            ],
        )
    )
    return graph


def latency(
    graph: Stream,
):
    """Report video filtering latency."""
    graph.append(Filter(command="latency", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def alatency(
    graph: Stream,
):
    """Report audio filtering latency."""
    graph.append(Filter(command="alatency", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def kerndeint(
    graph: Stream,
    thresh: Optional[int] = None,
    map: Optional[bool] = None,
    order: Optional[bool] = None,
    sharp: Optional[bool] = None,
    twoway: Optional[bool] = None,
):
    """Apply kernel deinterlacing to the input.

    :param int thresh: set the threshold
    :param bool map: set the map
    :param bool order: set the order
    :param bool sharp: set sharpening
    :param bool twoway: set twoway
    """
    graph.append(
        Filter(
            command="kerndeint",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="thresh", value=thresh),
                FilterOption(name="map", value=map),
                FilterOption(name="order", value=order),
                FilterOption(name="sharp", value=sharp),
                FilterOption(name="twoway", value=twoway),
            ],
        )
    )
    return graph


def showpalette(graph: Stream, s: Optional[int] = None):
    """Display frame palette.

    :param int s: set pixel box size
    """
    graph.append(Filter(command="showpalette", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="s", value=s)]))
    return graph


def hwdownload(
    graph: Stream,
):
    """Download a hardware frame to a normal frame"""
    graph.append(Filter(command="hwdownload", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def tonemap_opencl(
    graph: Stream,
    tonemap: Optional[str] = None,
    transfer: Optional[str] = None,
    matrix: Optional[str] = None,
    primaries: Optional[str] = None,
    range: Optional[str] = None,
    format: Optional[str] = None,
    peak: Optional[float] = None,
    param: Optional[float] = None,
    desat: Optional[float] = None,
    threshold: Optional[float] = None,
):
    """Perform HDR to SDR conversion with tonemapping.

    :param str tonemap: tonemap algorithm selection
            possible values: none, linear, gamma, clip, reinhard, hable, mobius
    :param str transfer: set transfer characteristic
            possible values: bt709, bt2020
    :param str matrix: set colorspace matrix
            possible values: bt709, bt2020
    :param str primaries: set color primaries
            possible values: bt709, bt2020
    :param str range: set color range
            possible values: tv, pc, limited, full
    :param str format: output pixel format
    :param float peak: signal peak override
    :param float param: tonemap parameter
    :param float desat: desaturation parameter
    :param float threshold: scene detection threshold
    """
    graph.append(
        Filter(
            command="tonemap_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="tonemap", value=tonemap),
                FilterOption(name="transfer", value=transfer),
                FilterOption(name="matrix", value=matrix),
                FilterOption(name="primaries", value=primaries),
                FilterOption(name="range", value=range),
                FilterOption(name="format", value=format),
                FilterOption(name="peak", value=peak),
                FilterOption(name="param", value=param),
                FilterOption(name="desat", value=desat),
                FilterOption(name="threshold", value=threshold),
            ],
        )
    )
    return graph


def nnedi(
    graph: Stream,
    weights: Optional[str] = None,
    deint: Optional[str] = None,
    field: Optional[str] = None,
    planes: Optional[int] = None,
    nsize: Optional[str] = None,
    nns: Optional[str] = None,
    qual: Optional[str] = None,
    etype: Optional[str] = None,
    pscrn: Optional[str] = None,
):
    """Apply neural network edge directed interpolation intra-only deinterlacer.

    :param str weights: set weights file
    :param str deint: set which frames to deinterlace
            possible values: all, interlaced
    :param str field: set mode of operation
            possible values: af, a, t, b, tf, bf
    :param int planes: set which planes to process
    :param str nsize: set size of local neighborhood around each pixel, used by the predictor neural network
            possible values: s8x6, s16x6, s32x6, s48x6, s8x4, s16x4, s32x4
    :param str nns: set number of neurons in predictor neural network
            possible values: n16, n32, n64, n128, n256
    :param str qual: set quality
            possible values: fast, slow
    :param str etype: set which set of weights to use in the predictor
            possible values: a, abs, s, mse
    :param str pscrn: set prescreening
            possible values: none, original, new, new2, new3
    """
    graph.append(
        Filter(
            command="nnedi",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="weights", value=weights),
                FilterOption(name="deint", value=deint),
                FilterOption(name="field", value=field),
                FilterOption(name="planes", value=planes),
                FilterOption(name="nsize", value=nsize),
                FilterOption(name="nns", value=nns),
                FilterOption(name="qual", value=qual),
                FilterOption(name="etype", value=etype),
                FilterOption(name="pscrn", value=pscrn),
            ],
        )
    )
    return graph


def pullup(
    graph: Stream,
    jl: Optional[int] = None,
    jr: Optional[int] = None,
    jt: Optional[int] = None,
    jb: Optional[int] = None,
    sb: Optional[bool] = None,
    mp: Optional[str] = None,
):
    """Pullup from field sequence to frames.

    :param int jl: set left junk size
    :param int jr: set right junk size
    :param int jt: set top junk size
    :param int jb: set bottom junk size
    :param bool sb: set strict breaks
    :param str mp: set metric plane
            possible values: y, u, v
    """
    graph.append(
        Filter(
            command="pullup",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="jl", value=jl),
                FilterOption(name="jr", value=jr),
                FilterOption(name="jt", value=jt),
                FilterOption(name="jb", value=jb),
                FilterOption(name="sb", value=sb),
                FilterOption(name="mp", value=mp),
            ],
        )
    )
    return graph


def crystalizer(graph: Stream, i: Optional[float] = None, c: Optional[bool] = None):
    """Simple audio noise sharpening filter.

    :param float i: set intensity
    :param bool c: enable clipping
    """
    graph.append(
        Filter(
            command="crystalizer",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="i", value=i), FilterOption(name="c", value=c)],
        )
    )
    return graph


def flanger(
    graph: Stream,
    delay: Optional[float] = None,
    depth: Optional[float] = None,
    regen: Optional[float] = None,
    width: Optional[float] = None,
    speed: Optional[float] = None,
    shape: Optional[str] = None,
    phase: Optional[float] = None,
    interp: Optional[str] = None,
):
    """Apply a flanging effect to the audio.

    :param float delay: base delay in milliseconds
    :param float depth: added swept delay in milliseconds
    :param float regen: percentage regeneration (delayed signal feedback)
    :param float width: percentage of delayed signal mixed with original
    :param float speed: sweeps per second (Hz)
    :param str shape: swept wave shape
            possible values: triangular, t, sinusoidal, s
    :param float phase: swept wave percentage phase-shift for multi-channel
    :param str interp: delay-line interpolation
            possible values: linear, quadratic
    """
    graph.append(
        Filter(
            command="flanger",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="delay", value=delay),
                FilterOption(name="depth", value=depth),
                FilterOption(name="regen", value=regen),
                FilterOption(name="width", value=width),
                FilterOption(name="speed", value=speed),
                FilterOption(name="shape", value=shape),
                FilterOption(name="phase", value=phase),
                FilterOption(name="interp", value=interp),
            ],
        )
    )
    return graph


def owdenoise(graph: Stream, depth: Optional[int] = None, luma_strength: Optional[float] = None, chroma_strength: Optional[float] = None):
    """Denoise using wavelets.

    :param int depth: set depth
    :param float luma_strength: set luma strength
    :param float chroma_strength: set chroma strength
    """
    graph.append(
        Filter(
            command="owdenoise",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="depth", value=depth),
                FilterOption(name="luma_strength", value=luma_strength),
                FilterOption(name="chroma_strength", value=chroma_strength),
            ],
        )
    )
    return graph


def sr(
    graph: Stream,
    dnn_backend: Optional[int] = None,
    scale_factor: Optional[int] = None,
    model: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
):
    """Apply DNN-based image super resolution to the input.

    :param int dnn_backend: DNN backend used for model execution
    :param int scale_factor: scale factor for SRCNN model
    :param str model: path to model file specifying network architecture and its parameters
    :param str input: input name of the model
    :param str output: output name of the model
    """
    graph.append(
        Filter(
            command="sr",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="dnn_backend", value=dnn_backend),
                FilterOption(name="scale_factor", value=scale_factor),
                FilterOption(name="model", value=model),
                FilterOption(name="input", value=input),
                FilterOption(name="output", value=output),
            ],
        )
    )
    return graph


def overlay_opencl(graph: Stream, x: Optional[int] = None, y: Optional[int] = None):
    """Overlay one video on top of another

    :param int x: Overlay x position
    :param int y: Overlay y position
    """
    graph.append(
        Filter(
            command="overlay_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="x", value=x), FilterOption(name="y", value=y)],
        )
    )
    return graph


def extrastereo(graph: Stream, m: Optional[float] = None, c: Optional[bool] = None):
    """Increase difference between stereo audio channels.

    :param float m: set the difference coefficient
    :param bool c: enable clipping
    """
    graph.append(
        Filter(
            command="extrastereo",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="m", value=m), FilterOption(name="c", value=c)],
        )
    )
    return graph


def maskedclamp(graph: Stream, undershoot: Optional[int] = None, overshoot: Optional[int] = None, planes: Optional[int] = None):
    """Clamp first stream with second stream and third stream.

    :param int undershoot: set undershoot
    :param int overshoot: set overshoot
    :param int planes: set planes
    """
    graph.append(
        Filter(
            command="maskedclamp",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="undershoot", value=undershoot),
                FilterOption(name="overshoot", value=overshoot),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def pseudocolor(
    graph: Stream,
    c0: Optional[str] = None,
    c1: Optional[str] = None,
    c2: Optional[str] = None,
    c3: Optional[str] = None,
    index: Optional[int] = None,
    preset: Optional[str] = None,
    opacity: Optional[float] = None,
):
    """Make pseudocolored video frames.

    :param str c0: set component #0 expression
    :param str c1: set component #1 expression
    :param str c2: set component #2 expression
    :param str c3: set component #3 expression
    :param int index: set component as base
    :param str preset: set preset
            possible values: none, magma, inferno, plasma, viridis, turbo, cividis, range1, range2, shadows, highlights, solar, nominal, preferred, total, spectral, cool, heat, fiery, blues, green, helix
    :param float opacity: set pseudocolor opacity
    """
    graph.append(
        Filter(
            command="pseudocolor",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="c0", value=c0),
                FilterOption(name="c1", value=c1),
                FilterOption(name="c2", value=c2),
                FilterOption(name="c3", value=c3),
                FilterOption(name="index", value=index),
                FilterOption(name="preset", value=preset),
                FilterOption(name="opacity", value=opacity),
            ],
        )
    )
    return graph


def xfade(
    graph: Stream,
    transition: Optional[str] = None,
    duration: Optional[int] = None,
    offset: Optional[int] = None,
    expr: Optional[str] = None,
):
    """Cross fade one video with another video.

    :param str transition: set cross fade transition
            possible values: custom, fade, wipeleft, wiperight, wipeup, wipedown, slideleft, slideright, slideup, slidedown, circlecrop, rectcrop, distance, fadeblack, fadewhite, radial, smoothleft, smoothright, smoothup, smoothdown, circleopen, circleclose, vertopen, vertclose, horzopen, horzclose, dissolve, pixelize, diagtl, diagtr, diagbl, diagbr, hlslice, hrslice, vuslice, vdslice, hblur, fadegrays, wipetl, wipetr, wipebl, wipebr, squeezeh, squeezev, zoomin, fadefast, fadeslow, hlwind, hrwind, vuwind, vdwind, coverleft, coverright, coverup, coverdown, revealleft, revealright, revealup, revealdown
    :param int duration: set cross fade duration
    :param int offset: set cross fade start relative to first input stream
    :param str expr: set expression for custom transition
    """
    graph.append(
        Filter(
            command="xfade",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="transition", value=transition),
                FilterOption(name="duration", value=duration),
                FilterOption(name="offset", value=offset),
                FilterOption(name="expr", value=expr),
            ],
        )
    )
    return graph


def perspective(
    graph: Stream,
    x0: Optional[str] = None,
    y0: Optional[str] = None,
    x1: Optional[str] = None,
    y1: Optional[str] = None,
    x2: Optional[str] = None,
    y2: Optional[str] = None,
    x3: Optional[str] = None,
    y3: Optional[str] = None,
    interpolation: Optional[str] = None,
    sense: Optional[str] = None,
    eval: Optional[str] = None,
):
    """Correct the perspective of video.

    :param str x0: set top left x coordinate
    :param str y0: set top left y coordinate
    :param str x1: set top right x coordinate
    :param str y1: set top right y coordinate
    :param str x2: set bottom left x coordinate
    :param str y2: set bottom left y coordinate
    :param str x3: set bottom right x coordinate
    :param str y3: set bottom right y coordinate
    :param str interpolation: set interpolation
            possible values: linear, cubic
    :param str sense: specify the sense of the coordinates
            possible values: source, destination
    :param str eval: specify when to evaluate expressions
            possible values: init, frame
    """
    graph.append(
        Filter(
            command="perspective",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="x0", value=x0),
                FilterOption(name="y0", value=y0),
                FilterOption(name="x1", value=x1),
                FilterOption(name="y1", value=y1),
                FilterOption(name="x2", value=x2),
                FilterOption(name="y2", value=y2),
                FilterOption(name="x3", value=x3),
                FilterOption(name="y3", value=y3),
                FilterOption(name="interpolation", value=interpolation),
                FilterOption(name="sense", value=sense),
                FilterOption(name="eval", value=eval),
            ],
        )
    )
    return graph


def tremolo(graph: Stream, f: Optional[float] = None, d: Optional[float] = None):
    """Apply tremolo effect.

    :param float f: set frequency in hertz
    :param float d: set depth as percentage
    """
    graph.append(
        Filter(
            command="tremolo", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="f", value=f), FilterOption(name="d", value=d)]
        )
    )
    return graph


def adynamicsmooth(graph: Stream, sensitivity: Optional[float] = None, basefreq: Optional[float] = None):
    """Apply Dynamic Smoothing of input audio.

    :param float sensitivity: set smooth sensitivity
    :param float basefreq: set base frequency
    """
    graph.append(
        Filter(
            command="adynamicsmooth",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="sensitivity", value=sensitivity), FilterOption(name="basefreq", value=basefreq)],
        )
    )
    return graph


def bm3d(
    graph: Stream,
    sigma: Optional[float] = None,
    block: Optional[int] = None,
    bstep: Optional[int] = None,
    group: Optional[int] = None,
    range: Optional[int] = None,
    mstep: Optional[int] = None,
    thmse: Optional[float] = None,
    hdthr: Optional[float] = None,
    estim: Optional[str] = None,
    ref: Optional[bool] = None,
    planes: Optional[int] = None,
):
    """Block-Matching 3D denoiser.

    :param float sigma: set denoising strength
    :param int block: set size of local patch
    :param int bstep: set sliding step for processing blocks
    :param int group: set maximal number of similar blocks
    :param int range: set block matching range
    :param int mstep: set step for block matching
    :param float thmse: set threshold of mean square error for block matching
    :param float hdthr: set hard threshold for 3D transfer domain
    :param str estim: set filtering estimation mode
            possible values: basic, final
    :param bool ref: have reference stream
    :param int planes: set planes to filter
    """
    graph.append(
        Filter(
            command="bm3d",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="sigma", value=sigma),
                FilterOption(name="block", value=block),
                FilterOption(name="bstep", value=bstep),
                FilterOption(name="group", value=group),
                FilterOption(name="range", value=range),
                FilterOption(name="mstep", value=mstep),
                FilterOption(name="thmse", value=thmse),
                FilterOption(name="hdthr", value=hdthr),
                FilterOption(name="estim", value=estim),
                FilterOption(name="ref", value=ref),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def shuffleframes(graph: Stream, mapping: Optional[str] = None):
    """Shuffle video frames.

    :param str mapping: set destination indexes of input frames
    """
    graph.append(Filter(command="shuffleframes", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="mapping", value=mapping)]))
    return graph


def hqdn3d(
    graph: Stream,
    luma_spatial: Optional[float] = None,
    chroma_spatial: Optional[float] = None,
    luma_tmp: Optional[float] = None,
    chroma_tmp: Optional[float] = None,
):
    """Apply a High Quality 3D Denoiser.

    :param float luma_spatial: spatial luma strength
    :param float chroma_spatial: spatial chroma strength
    :param float luma_tmp: temporal luma strength
    :param float chroma_tmp: temporal chroma strength
    """
    graph.append(
        Filter(
            command="hqdn3d",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="luma_spatial", value=luma_spatial),
                FilterOption(name="chroma_spatial", value=chroma_spatial),
                FilterOption(name="luma_tmp", value=luma_tmp),
                FilterOption(name="chroma_tmp", value=chroma_tmp),
            ],
        )
    )
    return graph


def lut3d(
    graph: Stream,
):
    """Adjust colors using a 3D LUT."""
    graph.append(Filter(command="lut3d", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def haldclut(
    graph: Stream,
):
    """Adjust colors using a Hald CLUT."""
    graph.append(Filter(command="haldclut", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def lut1d(graph: Stream, file: Optional[str] = None, interp: Optional[str] = None):
    """Adjust colors using a 1D LUT.

    :param str file: set 1D LUT file name
    :param str interp: select interpolation mode
            possible values: nearest, linear, cosine, cubic, spline
    """
    graph.append(
        Filter(
            command="lut1d",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="file", value=file), FilterOption(name="interp", value=interp)],
        )
    )
    return graph


def elbg(
    graph: Stream,
    codebook_length: Optional[int] = None,
    nb_steps: Optional[int] = None,
    seed: Optional[int] = None,
    pal8: Optional[bool] = None,
    use_alpha: Optional[bool] = None,
):
    """Apply posterize effect, using the ELBG algorithm.

    :param int codebook_length: set codebook length
    :param int nb_steps: set max number of steps used to compute the mapping
    :param int seed: set the random seed
    :param bool pal8: set the pal8 output
    :param bool use_alpha: use alpha channel for mapping
    """
    graph.append(
        Filter(
            command="elbg",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="codebook_length", value=codebook_length),
                FilterOption(name="nb_steps", value=nb_steps),
                FilterOption(name="seed", value=seed),
                FilterOption(name="pal8", value=pal8),
                FilterOption(name="use_alpha", value=use_alpha),
            ],
        )
    )
    return graph


def blackdetect(
    graph: Stream, d: Optional[float] = None, picture_black_ratio_th: Optional[float] = None, pixel_black_th: Optional[float] = None
):
    """Detect video intervals that are (almost) black.

    :param float d: set minimum detected black duration in seconds
    :param float picture_black_ratio_th: set the picture black ratio threshold
    :param float pixel_black_th: set the pixel black threshold
    """
    graph.append(
        Filter(
            command="blackdetect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="d", value=d),
                FilterOption(name="picture_black_ratio_th", value=picture_black_ratio_th),
                FilterOption(name="pixel_black_th", value=pixel_black_th),
            ],
        )
    )
    return graph


def gradfun(graph: Stream, strength: Optional[float] = None, radius: Optional[int] = None):
    """Debands video quickly using gradients.

    :param float strength: The maximum amount by which the filter will change any one pixel.
    :param int radius: The neighborhood to fit the gradient to.
    """
    graph.append(
        Filter(
            command="gradfun",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="strength", value=strength), FilterOption(name="radius", value=radius)],
        )
    )
    return graph


def firequalizer(
    graph: Stream,
    gain: Optional[str] = None,
    gain_entry: Optional[str] = None,
    delay: Optional[float] = None,
    accuracy: Optional[float] = None,
    wfunc: Optional[str] = None,
    fixed: Optional[bool] = None,
    multi: Optional[bool] = None,
    zero_phase: Optional[bool] = None,
    scale: Optional[str] = None,
    dumpfile: Optional[str] = None,
    dumpscale: Optional[int] = None,
    fft2: Optional[bool] = None,
    min_phase: Optional[bool] = None,
):
    """Finite Impulse Response Equalizer.

    :param str gain: set gain curve
    :param str gain_entry: set gain entry
    :param float delay: set delay
    :param float accuracy: set accuracy
    :param str wfunc: set window function
            possible values: rectangular, hann, hamming, blackman, nuttall3, mnuttall3, nuttall, bnuttall, bharris, tukey
    :param bool fixed: set fixed frame samples
    :param bool multi: set multi channels mode
    :param bool zero_phase: set zero phase mode
    :param str scale: set gain scale
            possible values: linlin, linlog, loglin, loglog
    :param str dumpfile: set dump file
    :param int dumpscale: set dump scale
    :param bool fft2: set 2-channels fft
    :param bool min_phase: set minimum phase mode
    """
    graph.append(
        Filter(
            command="firequalizer",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="gain", value=gain),
                FilterOption(name="gain_entry", value=gain_entry),
                FilterOption(name="delay", value=delay),
                FilterOption(name="accuracy", value=accuracy),
                FilterOption(name="wfunc", value=wfunc),
                FilterOption(name="fixed", value=fixed),
                FilterOption(name="multi", value=multi),
                FilterOption(name="zero_phase", value=zero_phase),
                FilterOption(name="scale", value=scale),
                FilterOption(name="dumpfile", value=dumpfile),
                FilterOption(name="dumpscale", value=dumpscale),
                FilterOption(name="fft2", value=fft2),
                FilterOption(name="min_phase", value=min_phase),
            ],
        )
    )
    return graph


def interleave(graph: Stream, nb_inputs: Optional[int] = None, duration: Optional[str] = None):
    """Temporally interleave video inputs.

    :param int nb_inputs: set number of inputs
    :param str duration: how to determine the end-of-stream
            possible values: longest, shortest, first
    """
    graph.append(
        Filter(
            command="interleave",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="nb_inputs", value=nb_inputs), FilterOption(name="duration", value=duration)],
        )
    )
    return graph


def ainterleave(graph: Stream, nb_inputs: Optional[int] = None, duration: Optional[str] = None):
    """Temporally interleave audio inputs.

    :param int nb_inputs: set number of inputs
    :param str duration: how to determine the end-of-stream
            possible values: longest, shortest, first
    """
    graph.append(
        Filter(
            command="ainterleave",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="nb_inputs", value=nb_inputs), FilterOption(name="duration", value=duration)],
        )
    )
    return graph


def transpose_opencl(graph: Stream, dir: Optional[str] = None, passthrough: Optional[str] = None):
    """Transpose input video

    :param str dir: set transpose direction
            possible values: cclock_flip, clock, cclock, clock_flip
    :param str passthrough: do not apply transposition if the input matches the specified geometry
            possible values: none, portrait, landscape
    """
    graph.append(
        Filter(
            command="transpose_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="dir", value=dir), FilterOption(name="passthrough", value=passthrough)],
        )
    )
    return graph


def deshake_opencl(
    graph: Stream,
    tripod: Optional[bool] = None,
    debug: Optional[bool] = None,
    adaptive_crop: Optional[bool] = None,
    refine_features: Optional[bool] = None,
    smooth_strength: Optional[float] = None,
    smooth_window_multiplier: Optional[float] = None,
):
    """Feature-point based video stabilization filter

    :param bool tripod: simulates a tripod by preventing any camera movement whatsoever from the original frame
    :param bool debug: turn on additional debugging information
    :param bool adaptive_crop: attempt to subtly crop borders to reduce mirrored content
    :param bool refine_features: refine feature point locations at a sub-pixel level
    :param float smooth_strength: smoothing strength (0 attempts to adaptively determine optimal strength)
    :param float smooth_window_multiplier: multiplier for number of frames to buffer for motion data
    """
    graph.append(
        Filter(
            command="deshake_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="tripod", value=tripod),
                FilterOption(name="debug", value=debug),
                FilterOption(name="adaptive_crop", value=adaptive_crop),
                FilterOption(name="refine_features", value=refine_features),
                FilterOption(name="smooth_strength", value=smooth_strength),
                FilterOption(name="smooth_window_multiplier", value=smooth_window_multiplier),
            ],
        )
    )
    return graph


def unsharp_opencl(
    graph: Stream,
    luma_msize_x: Optional[float] = None,
    luma_msize_y: Optional[float] = None,
    luma_amount: Optional[float] = None,
    chroma_msize_x: Optional[float] = None,
    chroma_msize_y: Optional[float] = None,
    chroma_amount: Optional[float] = None,
):
    """Apply unsharp mask to input video

    :param float luma_msize_x: Set luma mask horizontal diameter (pixels)
    :param float luma_msize_y: Set luma mask vertical diameter (pixels)
    :param float luma_amount: Set luma amount (multiplier)
    :param float chroma_msize_x: Set chroma mask horizontal diameter (pixels after subsampling)
    :param float chroma_msize_y: Set chroma mask vertical diameter (pixels after subsampling)
    :param float chroma_amount: Set chroma amount (multiplier)
    """
    graph.append(
        Filter(
            command="unsharp_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="luma_msize_x", value=luma_msize_x),
                FilterOption(name="luma_msize_y", value=luma_msize_y),
                FilterOption(name="luma_amount", value=luma_amount),
                FilterOption(name="chroma_msize_x", value=chroma_msize_x),
                FilterOption(name="chroma_msize_y", value=chroma_msize_y),
                FilterOption(name="chroma_amount", value=chroma_amount),
            ],
        )
    )
    return graph


def colorlevels(
    graph: Stream,
    rimin: Optional[float] = None,
    gimin: Optional[float] = None,
    bimin: Optional[float] = None,
    aimin: Optional[float] = None,
    rimax: Optional[float] = None,
    gimax: Optional[float] = None,
    bimax: Optional[float] = None,
    aimax: Optional[float] = None,
    romin: Optional[float] = None,
    gomin: Optional[float] = None,
    bomin: Optional[float] = None,
    aomin: Optional[float] = None,
    romax: Optional[float] = None,
    gomax: Optional[float] = None,
    bomax: Optional[float] = None,
    aomax: Optional[float] = None,
    preserve: Optional[str] = None,
):
    """Adjust the color levels.

    :param float rimin: set input red black point
    :param float gimin: set input green black point
    :param float bimin: set input blue black point
    :param float aimin: set input alpha black point
    :param float rimax: set input red white point
    :param float gimax: set input green white point
    :param float bimax: set input blue white point
    :param float aimax: set input alpha white point
    :param float romin: set output red black point
    :param float gomin: set output green black point
    :param float bomin: set output blue black point
    :param float aomin: set output alpha black point
    :param float romax: set output red white point
    :param float gomax: set output green white point
    :param float bomax: set output blue white point
    :param float aomax: set output alpha white point
    :param str preserve: set preserve color mode
            possible values: none, lum, max, avg, sum, nrm, pwr
    """
    graph.append(
        Filter(
            command="colorlevels",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rimin", value=rimin),
                FilterOption(name="gimin", value=gimin),
                FilterOption(name="bimin", value=bimin),
                FilterOption(name="aimin", value=aimin),
                FilterOption(name="rimax", value=rimax),
                FilterOption(name="gimax", value=gimax),
                FilterOption(name="bimax", value=bimax),
                FilterOption(name="aimax", value=aimax),
                FilterOption(name="romin", value=romin),
                FilterOption(name="gomin", value=gomin),
                FilterOption(name="bomin", value=bomin),
                FilterOption(name="aomin", value=aomin),
                FilterOption(name="romax", value=romax),
                FilterOption(name="gomax", value=gomax),
                FilterOption(name="bomax", value=bomax),
                FilterOption(name="aomax", value=aomax),
                FilterOption(name="preserve", value=preserve),
            ],
        )
    )
    return graph


def super2xsai(
    graph: Stream,
):
    """Scale the input by 2x using the Super2xSaI pixel art algorithm."""
    graph.append(Filter(command="super2xsai", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def buffer(
    graph: Stream,
    width: Optional[int] = None,
    height: Optional[int] = None,
    pix_fmt: Optional[str] = None,
    sar: Optional[int] = None,
    time_base: Optional[int] = None,
    frame_rate: Optional[int] = None,
    colorspace: Optional[str] = None,
    range: Optional[str] = None,
):
    """Buffer video frames, and make them accessible to the filterchain.

    :param int sar: sample aspect ratio
    :param str colorspace: select colorspace
            possible values: gbr, bt709, unknown, fcc, bt470bg, smpte170m, smpte240m, ycgco, ycgco-re, ycgco-ro, bt2020nc, bt2020c, smpte2085, chroma-derived-nc, chroma-derived-c, ictcp, ipt-c2
    :param str range: select color range
            possible values: unspecified, unknown, limited, tv, mpeg, full, pc, jpeg
    """
    graph.append(
        Filter(
            command="buffer",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="width", value=width),
                FilterOption(name="height", value=height),
                FilterOption(name="pix_fmt", value=pix_fmt),
                FilterOption(name="sar", value=sar),
                FilterOption(name="time_base", value=time_base),
                FilterOption(name="frame_rate", value=frame_rate),
                FilterOption(name="colorspace", value=colorspace),
                FilterOption(name="range", value=range),
            ],
        )
    )
    return graph


def abuffer(
    graph: Stream,
    time_base: Optional[int] = None,
    sample_rate: Optional[int] = None,
    sample_fmt: Optional[str] = None,
    channel_layout: Optional[str] = None,
    channels: Optional[int] = None,
):
    """Buffer audio frames, and make them accessible to the filterchain."""
    graph.append(
        Filter(
            command="abuffer",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="time_base", value=time_base),
                FilterOption(name="sample_rate", value=sample_rate),
                FilterOption(name="sample_fmt", value=sample_fmt),
                FilterOption(name="channel_layout", value=channel_layout),
                FilterOption(name="channels", value=channels),
            ],
        )
    )
    return graph


def arnndn(graph: Stream, model: Optional[str] = None, mix: Optional[float] = None):
    """Reduce noise from speech using Recurrent Neural Networks.

    :param str model: set model name
    :param float mix: set output vs input mix
    """
    graph.append(
        Filter(
            command="arnndn",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="model", value=model), FilterOption(name="mix", value=mix)],
        )
    )
    return graph


def adecorrelate(graph: Stream, stages: Optional[int] = None, seed: Optional[int] = None):
    """Apply decorrelation to input audio.

    :param int stages: set filtering stages
    :param int seed: set random seed
    """
    graph.append(
        Filter(
            command="adecorrelate",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="stages", value=stages), FilterOption(name="seed", value=seed)],
        )
    )
    return graph


def join(graph: Stream, inputs: Optional[int] = None, channel_layout: Optional[str] = None, map: Optional[str] = None):
    """Join multiple audio streams into multi-channel output.

    :param int inputs: Number of input streams.
    :param str channel_layout: Channel layout of the output stream.
    :param str map: A comma-separated list of channels maps in the format 'input_stream.input_channel-output_channel.
    """
    graph.append(
        Filter(
            command="join",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="inputs", value=inputs),
                FilterOption(name="channel_layout", value=channel_layout),
                FilterOption(name="map", value=map),
            ],
        )
    )
    return graph


def pixdesctest(
    graph: Stream,
):
    """Test pixel format definitions."""
    graph.append(Filter(command="pixdesctest", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def dnn_processing(
    graph: Stream,
    dnn_backend: Optional[int] = None,
    model: Optional[str] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
    backend_configs: Optional[str] = None,
    options: Optional[str] = None,
    _async: Optional[bool] = None,
):
    """Apply DNN processing filter to the input.

    :param int dnn_backend: DNN backend
    :param str model: path to model file
    :param str input: input name of the model
    :param str output: output name of the model
    :param str backend_configs: backend configs
    :param str options: backend configs (deprecated, use backend_configs)
    :param bool async: use DNN async inference (ignored, use backend_configs='async=1')
    """
    graph.append(
        Filter(
            command="dnn_processing",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="dnn_backend", value=dnn_backend),
                FilterOption(name="model", value=model),
                FilterOption(name="input", value=input),
                FilterOption(name="output", value=output),
                FilterOption(name="backend_configs", value=backend_configs),
                FilterOption(name="options", value=options),
                FilterOption(name="async", value=_async),
            ],
        )
    )
    return graph


def hdcd(
    graph: Stream,
    disable_autoconvert: Optional[bool] = None,
    process_stereo: Optional[bool] = None,
    cdt_ms: Optional[int] = None,
    force_pe: Optional[bool] = None,
    analyze_mode: Optional[str] = None,
    bits_per_sample: Optional[str] = None,
):
    """Apply High Definition Compatible Digital (HDCD) decoding.

    :param bool disable_autoconvert: Disable any format conversion or resampling in the filter graph.
    :param bool process_stereo: Process stereo channels together. Only apply target_gain when both channels match.
    :param int cdt_ms: Code detect timer period in ms.
    :param bool force_pe: Always extend peaks above -3dBFS even when PE is not signaled.
    :param str analyze_mode: Replace audio with solid tone and signal some processing aspect in the amplitude.
            possible values: off, lle, pe, cdt, tgm
    :param str bits_per_sample: Valid bits per sample (location of the true LSB).
            possible values: 16, 20, 24
    """
    graph.append(
        Filter(
            command="hdcd",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="disable_autoconvert", value=disable_autoconvert),
                FilterOption(name="process_stereo", value=process_stereo),
                FilterOption(name="cdt_ms", value=cdt_ms),
                FilterOption(name="force_pe", value=force_pe),
                FilterOption(name="analyze_mode", value=analyze_mode),
                FilterOption(name="bits_per_sample", value=bits_per_sample),
            ],
        )
    )
    return graph


def avectorscope(
    graph: Stream,
    mode: Optional[str] = None,
    rate: Optional[str] = None,
    size: Optional[int] = None,
    rc: Optional[int] = None,
    gc: Optional[int] = None,
    bc: Optional[int] = None,
    ac: Optional[int] = None,
    rf: Optional[int] = None,
    gf: Optional[int] = None,
    bf: Optional[int] = None,
    af: Optional[int] = None,
    zoom: Optional[float] = None,
    draw: Optional[str] = None,
    scale: Optional[str] = None,
    swap: Optional[bool] = None,
    mirror: Optional[str] = None,
):
    """Convert input audio to vectorscope video output.

    :param str mode: set mode
            possible values: lissajous, lissajous_xy, polar
    :param str rate: set video rate
    :param int size: set video size
    :param int rc: set red contrast
    :param int gc: set green contrast
    :param int bc: set blue contrast
    :param int ac: set alpha contrast
    :param int rf: set red fade
    :param int gf: set green fade
    :param int bf: set blue fade
    :param int af: set alpha fade
    :param float zoom: set zoom factor
    :param str draw: set draw mode
            possible values: dot, line, aaline
    :param str scale: set amplitude scale mode
            possible values: lin, sqrt, cbrt, log
    :param bool swap: swap x axis with y axis
    :param str mirror: mirror axis
            possible values: none, x, y, xy
    """
    graph.append(
        Filter(
            command="avectorscope",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mode", value=mode),
                FilterOption(name="rate", value=rate),
                FilterOption(name="size", value=size),
                FilterOption(name="rc", value=rc),
                FilterOption(name="gc", value=gc),
                FilterOption(name="bc", value=bc),
                FilterOption(name="ac", value=ac),
                FilterOption(name="rf", value=rf),
                FilterOption(name="gf", value=gf),
                FilterOption(name="bf", value=bf),
                FilterOption(name="af", value=af),
                FilterOption(name="zoom", value=zoom),
                FilterOption(name="draw", value=draw),
                FilterOption(name="scale", value=scale),
                FilterOption(name="swap", value=swap),
                FilterOption(name="mirror", value=mirror),
            ],
        )
    )
    return graph


def vibrance(
    graph: Stream,
    intensity: Optional[float] = None,
    rbal: Optional[float] = None,
    gbal: Optional[float] = None,
    bbal: Optional[float] = None,
    rlum: Optional[float] = None,
    glum: Optional[float] = None,
    blum: Optional[float] = None,
    alternate: Optional[bool] = None,
):
    """Boost or alter saturation.

    :param float intensity: set the intensity value
    :param float rbal: set the red balance value
    :param float gbal: set the green balance value
    :param float bbal: set the blue balance value
    :param float rlum: set the red luma coefficient
    :param float glum: set the green luma coefficient
    :param float blum: set the blue luma coefficient
    :param bool alternate: use alternate colors
    """
    graph.append(
        Filter(
            command="vibrance",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="intensity", value=intensity),
                FilterOption(name="rbal", value=rbal),
                FilterOption(name="gbal", value=gbal),
                FilterOption(name="bbal", value=bbal),
                FilterOption(name="rlum", value=rlum),
                FilterOption(name="glum", value=glum),
                FilterOption(name="blum", value=blum),
                FilterOption(name="alternate", value=alternate),
            ],
        )
    )
    return graph


def showspectrum(
    graph: Stream,
    size: Optional[int] = None,
    slide: Optional[str] = None,
    mode: Optional[str] = None,
    color: Optional[str] = None,
    scale: Optional[str] = None,
    fscale: Optional[str] = None,
    saturation: Optional[float] = None,
    win_func: Optional[str] = None,
    orientation: Optional[str] = None,
    overlap: Optional[float] = None,
    gain: Optional[float] = None,
    data: Optional[str] = None,
    rotation: Optional[float] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    fps: Optional[str] = None,
    legend: Optional[bool] = None,
    drange: Optional[float] = None,
    limit: Optional[float] = None,
    opacity: Optional[float] = None,
):
    """Convert input audio to a spectrum video output.

    :param int size: set video size
    :param str slide: set sliding mode
            possible values: replace, scroll, fullframe, rscroll, lreplace
    :param str mode: set channel display mode
            possible values: combined, separate
    :param str color: set channel coloring
            possible values: channel, intensity, rainbow, moreland, nebulae, fire, fiery, fruit, cool, magma, green, viridis, plasma, cividis, terrain
    :param str scale: set display scale
            possible values: lin, sqrt, cbrt, log, 4thrt, 5thrt
    :param str fscale: set frequency scale
            possible values: lin, log
    :param float saturation: color saturation multiplier
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    :param str orientation: set orientation
            possible values: vertical, horizontal
    :param float overlap: set window overlap
    :param float gain: set scale gain
    :param str data: set data mode
            possible values: magnitude, phase, uphase
    :param float rotation: color rotation
    :param int start: start frequency
    :param int stop: stop frequency
    :param str fps: set video rate
    :param bool legend: draw legend
    :param float drange: set dynamic range in dBFS
    :param float limit: set upper limit in dBFS
    :param float opacity: set opacity strength
    """
    graph.append(
        Filter(
            command="showspectrum",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="slide", value=slide),
                FilterOption(name="mode", value=mode),
                FilterOption(name="color", value=color),
                FilterOption(name="scale", value=scale),
                FilterOption(name="fscale", value=fscale),
                FilterOption(name="saturation", value=saturation),
                FilterOption(name="win_func", value=win_func),
                FilterOption(name="orientation", value=orientation),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="gain", value=gain),
                FilterOption(name="data", value=data),
                FilterOption(name="rotation", value=rotation),
                FilterOption(name="start", value=start),
                FilterOption(name="stop", value=stop),
                FilterOption(name="fps", value=fps),
                FilterOption(name="legend", value=legend),
                FilterOption(name="drange", value=drange),
                FilterOption(name="limit", value=limit),
                FilterOption(name="opacity", value=opacity),
            ],
        )
    )
    return graph


def showspectrumpic(
    graph: Stream,
    size: Optional[int] = None,
    mode: Optional[str] = None,
    color: Optional[str] = None,
    scale: Optional[str] = None,
    fscale: Optional[str] = None,
    saturation: Optional[float] = None,
    win_func: Optional[str] = None,
    orientation: Optional[str] = None,
    gain: Optional[float] = None,
    legend: Optional[bool] = None,
    rotation: Optional[float] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    drange: Optional[float] = None,
    limit: Optional[float] = None,
    opacity: Optional[float] = None,
):
    """Convert input audio to a spectrum video output single picture.

    :param int size: set video size
    :param str mode: set channel display mode
            possible values: combined, separate
    :param str color: set channel coloring
            possible values: channel, intensity, rainbow, moreland, nebulae, fire, fiery, fruit, cool, magma, green, viridis, plasma, cividis, terrain
    :param str scale: set display scale
            possible values: lin, sqrt, cbrt, log, 4thrt, 5thrt
    :param str fscale: set frequency scale
            possible values: lin, log
    :param float saturation: color saturation multiplier
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    :param str orientation: set orientation
            possible values: vertical, horizontal
    :param float gain: set scale gain
    :param bool legend: draw legend
    :param float rotation: color rotation
    :param int start: start frequency
    :param int stop: stop frequency
    :param float drange: set dynamic range in dBFS
    :param float limit: set upper limit in dBFS
    :param float opacity: set opacity strength
    """
    graph.append(
        Filter(
            command="showspectrumpic",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="mode", value=mode),
                FilterOption(name="color", value=color),
                FilterOption(name="scale", value=scale),
                FilterOption(name="fscale", value=fscale),
                FilterOption(name="saturation", value=saturation),
                FilterOption(name="win_func", value=win_func),
                FilterOption(name="orientation", value=orientation),
                FilterOption(name="gain", value=gain),
                FilterOption(name="legend", value=legend),
                FilterOption(name="rotation", value=rotation),
                FilterOption(name="start", value=start),
                FilterOption(name="stop", value=stop),
                FilterOption(name="drange", value=drange),
                FilterOption(name="limit", value=limit),
                FilterOption(name="opacity", value=opacity),
            ],
        )
    )
    return graph


def setpts(graph: Stream, expr: Optional[str] = None):
    """Set PTS for the output video frame.

    :param str expr: Expression determining the frame timestamp
    """
    graph.append(Filter(command="setpts", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="expr", value=expr)]))
    return graph


def asetpts(graph: Stream, expr: Optional[str] = None):
    """Set PTS for the output audio frame.

    :param str expr: Expression determining the frame timestamp
    """
    graph.append(Filter(command="asetpts", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="expr", value=expr)]))
    return graph


def volumedetect(
    graph: Stream,
):
    """Detect audio volume."""
    graph.append(Filter(command="volumedetect", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def stereo3d(graph: Stream, _in: Optional[str] = None, out: Optional[str] = None):
    """Convert video stereoscopic 3D view.

    :param str in: set input format
            possible values: ab2l, tb2l, ab2r, tb2r, abl, tbl, abr, tbr, al, ar, sbs2l, sbs2r, sbsl, sbsr, irl, irr, icl, icr
    :param str out: set output format
            possible values: ab2l, tb2l, ab2r, tb2r, abl, tbl, abr, tbr, agmc, agmd, agmg, agmh, al, ar, arbg, arcc, arcd, arcg, arch, argg, aybc, aybd, aybg, aybh, irl, irr, ml, mr, sbs2l, sbs2r, sbsl, sbsr, chl, chr, icl, icr, hdmi
    """
    graph.append(
        Filter(
            command="stereo3d",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="in", value=_in), FilterOption(name="out", value=out)],
        )
    )
    return graph


def rotate(
    graph: Stream,
    angle: Optional[str] = None,
    out_w: Optional[str] = None,
    out_h: Optional[str] = None,
    fillcolor: Optional[str] = None,
    bilinear: Optional[bool] = None,
):
    """Rotate the input image.

    :param str angle: set angle (in radians)
    :param str out_w: set output width expression
    :param str out_h: set output height expression
    :param str fillcolor: set background fill color
    :param bool bilinear: use bilinear interpolation
    """
    graph.append(
        Filter(
            command="rotate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="angle", value=angle),
                FilterOption(name="out_w", value=out_w),
                FilterOption(name="out_h", value=out_h),
                FilterOption(name="fillcolor", value=fillcolor),
                FilterOption(name="bilinear", value=bilinear),
            ],
        )
    )
    return graph


def apsyclip(
    graph: Stream,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    clip: Optional[float] = None,
    diff: Optional[bool] = None,
    adaptive: Optional[float] = None,
    iterations: Optional[int] = None,
    level: Optional[bool] = None,
):
    """Audio Psychoacoustic Clipper.

    :param float level_in: set input level
    :param float level_out: set output level
    :param float clip: set clip level
    :param bool diff: enable difference
    :param float adaptive: set adaptive distortion
    :param int iterations: set iterations
    :param bool level: set auto level
    """
    graph.append(
        Filter(
            command="apsyclip",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="clip", value=clip),
                FilterOption(name="diff", value=diff),
                FilterOption(name="adaptive", value=adaptive),
                FilterOption(name="iterations", value=iterations),
                FilterOption(name="level", value=level),
            ],
        )
    )
    return graph


def dcshift(graph: Stream, shift: Optional[float] = None, limitergain: Optional[float] = None):
    """Apply a DC shift to the audio.

    :param float shift: set DC shift
    :param float limitergain: set limiter gain
    """
    graph.append(
        Filter(
            command="dcshift",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="shift", value=shift), FilterOption(name="limitergain", value=limitergain)],
        )
    )
    return graph


def varblur(graph: Stream, min_r: Optional[int] = None, max_r: Optional[int] = None, planes: Optional[int] = None):
    """Apply Variable Blur filter.

    :param int min_r: set min blur radius
    :param int max_r: set max blur radius
    :param int planes: set planes to filter
    """
    graph.append(
        Filter(
            command="varblur",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="min_r", value=min_r),
                FilterOption(name="max_r", value=max_r),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def threshold(graph: Stream, planes: Optional[int] = None):
    """Threshold first video stream using other video streams.

    :param int planes: set planes to filter
    """
    graph.append(Filter(command="threshold", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="planes", value=planes)]))
    return graph


def replaygain(graph: Stream, track_gain: Optional[float] = None, track_peak: Optional[float] = None):
    """ReplayGain scanner.

    :param float track_gain: track gain (dB)
    :param float track_peak: track peak
    """
    graph.append(
        Filter(
            command="replaygain",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="track_gain", value=track_gain), FilterOption(name="track_peak", value=track_peak)],
        )
    )
    return graph


def xfade_vulkan(graph: Stream, transition: Optional[str] = None, duration: Optional[int] = None, offset: Optional[int] = None):
    """Cross fade one video with another video.

    :param str transition: set cross fade transition
            possible values: fade, wipeleft, wiperight, wipeup, wipedown, slidedown, slideup, slideleft, slideright, circleopen, circleclose, dissolve, pixelize, wipetl, wipetr, wipebl, wipebr
    :param int duration: set cross fade duration
    :param int offset: set cross fade start relative to first input stream
    """
    graph.append(
        Filter(
            command="xfade_vulkan",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="transition", value=transition),
                FilterOption(name="duration", value=duration),
                FilterOption(name="offset", value=offset),
            ],
        )
    )
    return graph


def selectivecolor(
    graph: Stream,
    correction_method: Optional[str] = None,
    reds: Optional[str] = None,
    yellows: Optional[str] = None,
    greens: Optional[str] = None,
    cyans: Optional[str] = None,
    blues: Optional[str] = None,
    magentas: Optional[str] = None,
    whites: Optional[str] = None,
    neutrals: Optional[str] = None,
    blacks: Optional[str] = None,
    psfile: Optional[str] = None,
):
    """Apply CMYK adjustments to specific color ranges.

    :param str correction_method: select correction method
            possible values: absolute, relative
    :param str reds: adjust red regions
    :param str yellows: adjust yellow regions
    :param str greens: adjust green regions
    :param str cyans: adjust cyan regions
    :param str blues: adjust blue regions
    :param str magentas: adjust magenta regions
    :param str whites: adjust white regions
    :param str neutrals: adjust neutral regions
    :param str blacks: adjust black regions
    :param str psfile: set Photoshop selectivecolor file name
    """
    graph.append(
        Filter(
            command="selectivecolor",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="correction_method", value=correction_method),
                FilterOption(name="reds", value=reds),
                FilterOption(name="yellows", value=yellows),
                FilterOption(name="greens", value=greens),
                FilterOption(name="cyans", value=cyans),
                FilterOption(name="blues", value=blues),
                FilterOption(name="magentas", value=magentas),
                FilterOption(name="whites", value=whites),
                FilterOption(name="neutrals", value=neutrals),
                FilterOption(name="blacks", value=blacks),
                FilterOption(name="psfile", value=psfile),
            ],
        )
    )
    return graph


def apad(
    graph: Stream,
    packet_size: Optional[int] = None,
    pad_len: Optional[int] = None,
    whole_len: Optional[int] = None,
    pad_dur: Optional[int] = None,
    whole_dur: Optional[int] = None,
):
    """Pad audio with silence.

    :param int packet_size: set silence packet size
    :param int pad_len: set number of samples of silence to add
    :param int whole_len: set minimum target number of samples in the audio stream
    :param int pad_dur: set duration of silence to add
    :param int whole_dur: set minimum target duration in the audio stream
    """
    graph.append(
        Filter(
            command="apad",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="packet_size", value=packet_size),
                FilterOption(name="pad_len", value=pad_len),
                FilterOption(name="whole_len", value=whole_len),
                FilterOption(name="pad_dur", value=pad_dur),
                FilterOption(name="whole_dur", value=whole_dur),
            ],
        )
    )
    return graph


def fieldhint(graph: Stream, hint: Optional[str] = None, mode: Optional[str] = None):
    """Field matching using hints.

    :param str hint: set hint file
    :param str mode: set hint mode
            possible values: absolute, relative, pattern
    """
    graph.append(
        Filter(
            command="fieldhint",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="hint", value=hint), FilterOption(name="mode", value=mode)],
        )
    )
    return graph


def fieldmatch(
    graph: Stream,
    order: Optional[str] = None,
    mode: Optional[str] = None,
    ppsrc: Optional[bool] = None,
    field: Optional[str] = None,
    mchroma: Optional[bool] = None,
    y0: Optional[int] = None,
    y1: Optional[int] = None,
    scthresh: Optional[float] = None,
    combmatch: Optional[str] = None,
    combdbg: Optional[str] = None,
    cthresh: Optional[int] = None,
    chroma: Optional[bool] = None,
    blockx: Optional[int] = None,
    blocky: Optional[int] = None,
    combpel: Optional[int] = None,
):
    """Field matching for inverse telecine.

    :param str order: specify the assumed field order
            possible values: auto, bff, tff
    :param str mode: set the matching mode or strategy to use
            possible values: pc, pc_n, pc_u, pc_n_ub, pcn, pcn_ub
    :param bool ppsrc: mark main input as a pre-processed input and activate clean source input stream
    :param str field: set the field to match from
            possible values: auto, bottom, top
    :param bool mchroma: set whether or not chroma is included during the match comparisons
    :param int y0: define an exclusion band which excludes the lines between y0 and y1 from the field matching decision
    :param int y1: define an exclusion band which excludes the lines between y0 and y1 from the field matching decision
    :param float scthresh: set scene change detection threshold
    :param str combmatch: set combmatching mode
            possible values: none, sc, full
    :param str combdbg: enable comb debug
            possible values: none, pcn, pcnub
    :param int cthresh: set the area combing threshold used for combed frame detection
    :param bool chroma: set whether or not chroma is considered in the combed frame decision
    :param int blockx: set the x-axis size of the window used during combed frame detection
    :param int blocky: set the y-axis size of the window used during combed frame detection
    :param int combpel: set the number of combed pixels inside any of the blocky by blockx size blocks on the frame for the frame to be detected as combed
    """
    graph.append(
        Filter(
            command="fieldmatch",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="order", value=order),
                FilterOption(name="mode", value=mode),
                FilterOption(name="ppsrc", value=ppsrc),
                FilterOption(name="field", value=field),
                FilterOption(name="mchroma", value=mchroma),
                FilterOption(name="y0", value=y0),
                FilterOption(name="y1", value=y1),
                FilterOption(name="scthresh", value=scthresh),
                FilterOption(name="combmatch", value=combmatch),
                FilterOption(name="combdbg", value=combdbg),
                FilterOption(name="cthresh", value=cthresh),
                FilterOption(name="chroma", value=chroma),
                FilterOption(name="blockx", value=blockx),
                FilterOption(name="blocky", value=blocky),
                FilterOption(name="combpel", value=combpel),
            ],
        )
    )
    return graph


def bench(graph: Stream, action: Optional[str] = None):
    """Benchmark part of a filtergraph.

    :param str action: set action
            possible values: start, stop
    """
    graph.append(Filter(command="bench", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="action", value=action)]))
    return graph


def abench(graph: Stream, action: Optional[str] = None):
    """Benchmark part of a filtergraph.

    :param str action: set action
            possible values: start, stop
    """
    graph.append(Filter(command="abench", filter_type="AVMEDIA_TYPE_AUDIO", params=[FilterOption(name="action", value=action)]))
    return graph


def aloop(graph: Stream, loop: Optional[int] = None, size: Optional[int] = None, start: Optional[int] = None, time: Optional[int] = None):
    """Loop audio samples.

    :param int loop: number of loops
    :param int size: max number of samples to loop
    :param int start: set the loop start sample
    :param int time: set the loop start time
    """
    graph.append(
        Filter(
            command="aloop",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="loop", value=loop),
                FilterOption(name="size", value=size),
                FilterOption(name="start", value=start),
                FilterOption(name="time", value=time),
            ],
        )
    )
    return graph


def loop(graph: Stream, loop: Optional[int] = None, size: Optional[int] = None, start: Optional[int] = None, time: Optional[int] = None):
    """Loop video frames.

    :param int loop: number of loops
    :param int size: max number of frames to loop
    :param int start: set the loop start frame
    :param int time: set the loop start time
    """
    graph.append(
        Filter(
            command="loop",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="loop", value=loop),
                FilterOption(name="size", value=size),
                FilterOption(name="start", value=start),
                FilterOption(name="time", value=time),
            ],
        )
    )
    return graph


def hue(graph: Stream, h: Optional[str] = None, s: Optional[str] = None, H: Optional[str] = None, b: Optional[str] = None):
    """Adjust the hue and saturation of the input video.

    :param str h: set the hue angle degrees expression
    :param str s: set the saturation expression
    :param str H: set the hue angle radians expression
    :param str b: set the brightness expression
    """
    graph.append(
        Filter(
            command="hue",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="h", value=h),
                FilterOption(name="s", value=s),
                FilterOption(name="H", value=H),
                FilterOption(name="b", value=b),
            ],
        )
    )
    return graph


def remap_opencl(graph: Stream, interp: Optional[str] = None, fill: Optional[str] = None):
    """Remap pixels using OpenCL.

    :param str interp: set interpolation method
            possible values: near, linear
    :param str fill: set the color of the unmapped pixels
    """
    graph.append(
        Filter(
            command="remap_opencl",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="interp", value=interp), FilterOption(name="fill", value=fill)],
        )
    )
    return graph


def field(graph: Stream, type: Optional[str] = None):
    """Extract a field from the input video.

    :param str type: set field type (top or bottom)
            possible values: top, bottom
    """
    graph.append(Filter(command="field", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="type", value=type)]))
    return graph


def aspectralstats(
    graph: Stream,
    win_size: Optional[int] = None,
    win_func: Optional[str] = None,
    overlap: Optional[float] = None,
    measure: Optional[str] = None,
):
    """Show frequency domain statistics about audio frames.

    :param int win_size: set the window size
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    :param float overlap: set window overlap
    :param str measure: select the parameters which are measured
            possible values: none, all, mean, variance, centroid, spread, skewness, kurtosis, entropy, flatness, crest, flux, slope, decrease, rolloff
    """
    graph.append(
        Filter(
            command="aspectralstats",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="win_size", value=win_size),
                FilterOption(name="win_func", value=win_func),
                FilterOption(name="overlap", value=overlap),
                FilterOption(name="measure", value=measure),
            ],
        )
    )
    return graph


def axcorrelate(graph: Stream, size: Optional[int] = None, algo: Optional[str] = None):
    """Cross-correlate two audio streams.

    :param int size: set the segment size
    :param str algo: set the algorithm
            possible values: slow, fast, best
    """
    graph.append(
        Filter(
            command="axcorrelate",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="size", value=size), FilterOption(name="algo", value=algo)],
        )
    )
    return graph


def realtime(
    graph: Stream,
):
    """Slow down filtering to match realtime."""
    graph.append(Filter(command="realtime", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def arealtime(
    graph: Stream,
):
    """Slow down filtering to match realtime."""
    graph.append(Filter(command="arealtime", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def aperms(
    graph: Stream,
):
    """Set permissions for the output audio frame."""
    graph.append(Filter(command="aperms", filter_type="AVMEDIA_TYPE_AUDIO", params=[]))
    return graph


def perms(
    graph: Stream,
):
    """Set permissions for the output video frame."""
    graph.append(Filter(command="perms", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def setparams(
    graph: Stream,
    field_mode: Optional[str] = None,
    range: Optional[str] = None,
    color_primaries: Optional[str] = None,
    color_trc: Optional[str] = None,
    colorspace: Optional[str] = None,
):
    """Force field, or color property for the output video frame.

    :param str field_mode: select interlace mode
            possible values: auto, bff, tff, prog
    :param str range: select color range
            possible values: auto, unspecified, unknown, limited, tv, mpeg, full, pc, jpeg
    :param str color_primaries: select color primaries
            possible values: auto, bt709, unknown, bt470m, bt470bg, smpte170m, smpte240m, film, bt2020, smpte428, smpte431, smpte432, jedec-p22, ebu3213
    :param str color_trc: select color transfer
            possible values: auto, bt709, unknown, bt470m, bt470bg, smpte170m, smpte240m, linear, log100, log316, iec61966-2-4, bt1361e, iec61966-2-1, bt2020-10, bt2020-12, smpte2084, smpte428, arib-std-b67
    :param str colorspace: select colorspace
            possible values: auto, gbr, bt709, unknown, fcc, bt470bg, smpte170m, smpte240m, ycgco, ycgco-re, ycgco-ro, bt2020nc, bt2020c, smpte2085, chroma-derived-nc, chroma-derived-c, ictcp, ipt-c2
    """
    graph.append(
        Filter(
            command="setparams",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="field_mode", value=field_mode),
                FilterOption(name="range", value=range),
                FilterOption(name="color_primaries", value=color_primaries),
                FilterOption(name="color_trc", value=color_trc),
                FilterOption(name="colorspace", value=colorspace),
            ],
        )
    )
    return graph


def setrange(graph: Stream, range: Optional[str] = None):
    """Force color range for the output video frame.

    :param str range: select color range
            possible values: auto, unspecified, unknown, limited, tv, mpeg, full, pc, jpeg
    """
    graph.append(Filter(command="setrange", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="range", value=range)]))
    return graph


def setfield(graph: Stream, mode: Optional[str] = None):
    """Force field for the output video frame.

    :param str mode: select interlace mode
            possible values: auto, bff, tff, prog
    """
    graph.append(Filter(command="setfield", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="mode", value=mode)]))
    return graph


def setdar(graph: Stream, dar: Optional[str] = None, max: Optional[int] = None):
    """Set the frame display aspect ratio.

    :param str dar: set display aspect ratio
    :param int max: set max value for nominator or denominator in the ratio
    """
    graph.append(
        Filter(
            command="setdar",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="dar", value=dar), FilterOption(name="max", value=max)],
        )
    )
    return graph


def setsar(graph: Stream, sar: Optional[str] = None, max: Optional[int] = None):
    """Set the pixel sample aspect ratio.

    :param str sar: set sample (pixel) aspect ratio
    :param int max: set max value for nominator or denominator in the ratio
    """
    graph.append(
        Filter(
            command="setsar",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="sar", value=sar), FilterOption(name="max", value=max)],
        )
    )
    return graph


def lumakey(graph: Stream, threshold: Optional[float] = None, tolerance: Optional[float] = None, softness: Optional[float] = None):
    """Turns a certain luma into transparency.

    :param float threshold: set the threshold value
    :param float tolerance: set the tolerance value
    :param float softness: set the softness value
    """
    graph.append(
        Filter(
            command="lumakey",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="tolerance", value=tolerance),
                FilterOption(name="softness", value=softness),
            ],
        )
    )
    return graph


def cas(graph: Stream, strength: Optional[float] = None, planes: Optional[str] = None):
    """Contrast Adaptive Sharpen.

    :param float strength: set the sharpening strength
    :param str planes: set what planes to filter
    """
    graph.append(
        Filter(
            command="cas",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="strength", value=strength), FilterOption(name="planes", value=planes)],
        )
    )
    return graph


def spp(
    graph: Stream, quality: Optional[int] = None, qp: Optional[int] = None, mode: Optional[str] = None, use_bframe_qp: Optional[bool] = None
):
    """Apply a simple post processing filter.

    :param int quality: set quality
    :param int qp: force a constant quantizer parameter
    :param str mode: set thresholding mode
            possible values: hard, soft
    :param bool use_bframe_qp: use B-frames' QP
    """
    graph.append(
        Filter(
            command="spp",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="quality", value=quality),
                FilterOption(name="qp", value=qp),
                FilterOption(name="mode", value=mode),
                FilterOption(name="use_bframe_qp", value=use_bframe_qp),
            ],
        )
    )
    return graph


def multiply(graph: Stream, scale: Optional[float] = None, offset: Optional[float] = None, planes: Optional[str] = None):
    """Multiply first video stream with second video stream.

    :param float scale: set scale
    :param float offset: set offset
    :param str planes: set planes
    """
    graph.append(
        Filter(
            command="multiply",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="scale", value=scale),
                FilterOption(name="offset", value=offset),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def vignette(
    graph: Stream,
    angle: Optional[str] = None,
    x0: Optional[str] = None,
    y0: Optional[str] = None,
    mode: Optional[str] = None,
    eval: Optional[str] = None,
    dither: Optional[bool] = None,
    aspect: Optional[int] = None,
):
    """Make or reverse a vignette effect.

    :param str angle: set lens angle
    :param str x0: set circle center position on x-axis
    :param str y0: set circle center position on y-axis
    :param str mode: set forward/backward mode
            possible values: forward, backward
    :param str eval: specify when to evaluate expressions
            possible values: init, frame
    :param bool dither: set dithering
    :param int aspect: set aspect ratio
    """
    graph.append(
        Filter(
            command="vignette",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="angle", value=angle),
                FilterOption(name="x0", value=x0),
                FilterOption(name="y0", value=y0),
                FilterOption(name="mode", value=mode),
                FilterOption(name="eval", value=eval),
                FilterOption(name="dither", value=dither),
                FilterOption(name="aspect", value=aspect),
            ],
        )
    )
    return graph


def silenceremove(
    graph: Stream,
    start_periods: Optional[int] = None,
    start_duration: Optional[int] = None,
    start_threshold: Optional[float] = None,
    start_silence: Optional[int] = None,
    start_mode: Optional[str] = None,
    stop_periods: Optional[int] = None,
    stop_duration: Optional[int] = None,
    stop_threshold: Optional[float] = None,
    stop_silence: Optional[int] = None,
    stop_mode: Optional[int] = None,
    detection: Optional[str] = None,
    window: Optional[int] = None,
    timestamp: Optional[str] = None,
):
    """Remove silence.

    :param int start_periods: set periods of silence parts to skip from start
    :param int start_duration: set start duration of non-silence part
    :param float start_threshold: set threshold for start silence detection
    :param int start_silence: set start duration of silence part to keep
    :param str start_mode: set which channel will trigger trimming from start
            possible values: any, all
    :param int stop_periods: set periods of silence parts to skip from end
    :param int stop_duration: set stop duration of silence part
    :param float stop_threshold: set threshold for stop silence detection
    :param int stop_silence: set stop duration of silence part to keep
    :param int stop_mode: set which channel will trigger trimming from end
    :param str detection: set how silence is detected
            possible values: avg, rms, peak, median, ptp, dev
    :param int window: set duration of window for silence detection
    :param str timestamp: set how every output frame timestamp is processed
            possible values: write, copy
    """
    graph.append(
        Filter(
            command="silenceremove",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="start_periods", value=start_periods),
                FilterOption(name="start_duration", value=start_duration),
                FilterOption(name="start_threshold", value=start_threshold),
                FilterOption(name="start_silence", value=start_silence),
                FilterOption(name="start_mode", value=start_mode),
                FilterOption(name="stop_periods", value=stop_periods),
                FilterOption(name="stop_duration", value=stop_duration),
                FilterOption(name="stop_threshold", value=stop_threshold),
                FilterOption(name="stop_silence", value=stop_silence),
                FilterOption(name="stop_mode", value=stop_mode),
                FilterOption(name="detection", value=detection),
                FilterOption(name="window", value=window),
                FilterOption(name="timestamp", value=timestamp),
            ],
        )
    )
    return graph


def deesser(graph: Stream, i: Optional[float] = None, m: Optional[float] = None, f: Optional[float] = None, s: Optional[str] = None):
    """Apply de-essing to the audio.

    :param float i: set intensity
    :param float m: set max deessing
    :param float f: set frequency
    :param str s: set output mode
            possible values: i, o, e
    """
    graph.append(
        Filter(
            command="deesser",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="i", value=i),
                FilterOption(name="m", value=m),
                FilterOption(name="f", value=f),
                FilterOption(name="s", value=s),
            ],
        )
    )
    return graph


def nlmeans(
    graph: Stream,
    s: Optional[float] = None,
    p: Optional[int] = None,
    pc: Optional[int] = None,
    r: Optional[int] = None,
    rc: Optional[int] = None,
):
    """Non-local means denoiser.

    :param float s: denoising strength
    :param int p: patch size
    :param int pc: patch size for chroma planes
    :param int r: research window
    :param int rc: research window for chroma planes
    """
    graph.append(
        Filter(
            command="nlmeans",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="s", value=s),
                FilterOption(name="p", value=p),
                FilterOption(name="pc", value=pc),
                FilterOption(name="r", value=r),
                FilterOption(name="rc", value=rc),
            ],
        )
    )
    return graph


def mestimate(graph: Stream, method: Optional[str] = None, mb_size: Optional[int] = None, search_param: Optional[int] = None):
    """Generate motion vectors.

    :param str method: motion estimation method
            possible values: esa, tss, tdls, ntss, fss, ds, hexbs, epzs, umh
    :param int mb_size: macroblock size
    :param int search_param: search parameter
    """
    graph.append(
        Filter(
            command="mestimate",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="method", value=method),
                FilterOption(name="mb_size", value=mb_size),
                FilterOption(name="search_param", value=search_param),
            ],
        )
    )
    return graph


def edgedetect(
    graph: Stream, high: Optional[float] = None, low: Optional[float] = None, mode: Optional[str] = None, planes: Optional[str] = None
):
    """Detect and draw edge.

    :param float high: set high threshold
    :param float low: set low threshold
    :param str mode: set mode
            possible values: wires, colormix, canny
    :param str planes: set planes to filter
            possible values: y, u, v, r, g, b
    """
    graph.append(
        Filter(
            command="edgedetect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="high", value=high),
                FilterOption(name="low", value=low),
                FilterOption(name="mode", value=mode),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def removegrain(graph: Stream, m0: Optional[int] = None, m1: Optional[int] = None, m2: Optional[int] = None, m3: Optional[int] = None):
    """Remove grain.

    :param int m0: set mode for 1st plane
    :param int m1: set mode for 2nd plane
    :param int m2: set mode for 3rd plane
    :param int m3: set mode for 4th plane
    """
    graph.append(
        Filter(
            command="removegrain",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="m0", value=m0),
                FilterOption(name="m1", value=m1),
                FilterOption(name="m2", value=m2),
                FilterOption(name="m3", value=m3),
            ],
        )
    )
    return graph


def sierpinski(
    graph: Stream,
    size: Optional[int] = None,
    rate: Optional[str] = None,
    seed: Optional[int] = None,
    jump: Optional[int] = None,
    type: Optional[str] = None,
):
    """Render a Sierpinski fractal.

    :param int size: set frame size
    :param str rate: set frame rate
    :param int seed: set the seed
    :param int jump: set the jump
    :param str type: set fractal type
            possible values: carpet, triangle
    """
    graph.append(
        Filter(
            command="sierpinski",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="size", value=size),
                FilterOption(name="rate", value=rate),
                FilterOption(name="seed", value=seed),
                FilterOption(name="jump", value=jump),
                FilterOption(name="type", value=type),
            ],
        )
    )
    return graph


def swaprect(
    graph: Stream,
    w: Optional[str] = None,
    h: Optional[str] = None,
    x1: Optional[str] = None,
    y1: Optional[str] = None,
    x2: Optional[str] = None,
    y2: Optional[str] = None,
):
    """Swap 2 rectangular objects in video.

    :param str w: set rect width
    :param str h: set rect height
    :param str x1: set 1st rect x top left coordinate
    :param str y1: set 1st rect y top left coordinate
    :param str x2: set 2nd rect x top left coordinate
    :param str y2: set 2nd rect y top left coordinate
    """
    graph.append(
        Filter(
            command="swaprect",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="w", value=w),
                FilterOption(name="h", value=h),
                FilterOption(name="x1", value=x1),
                FilterOption(name="y1", value=y1),
                FilterOption(name="x2", value=x2),
                FilterOption(name="y2", value=y2),
            ],
        )
    )
    return graph


def abitscope(
    graph: Stream, rate: Optional[str] = None, size: Optional[int] = None, colors: Optional[str] = None, mode: Optional[str] = None
):
    """Convert input audio to audio bit scope video output.

    :param str rate: set video rate
    :param int size: set video size
    :param str colors: set channels colors
    :param str mode: set output mode
            possible values: bars, trace
    """
    graph.append(
        Filter(
            command="abitscope",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="rate", value=rate),
                FilterOption(name="size", value=size),
                FilterOption(name="colors", value=colors),
                FilterOption(name="mode", value=mode),
            ],
        )
    )
    return graph


def lenscorrection(
    graph: Stream,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    k1: Optional[float] = None,
    k2: Optional[float] = None,
    i: Optional[str] = None,
    fc: Optional[str] = None,
):
    """Rectify the image by correcting for lens distortion.

    :param float cx: set relative center x
    :param float cy: set relative center y
    :param float k1: set quadratic distortion factor
    :param float k2: set double quadratic distortion factor
    :param str i: set interpolation type
            possible values: nearest, bilinear
    :param str fc: set the color of the unmapped pixels
    """
    graph.append(
        Filter(
            command="lenscorrection",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="cx", value=cx),
                FilterOption(name="cy", value=cy),
                FilterOption(name="k1", value=k1),
                FilterOption(name="k2", value=k2),
                FilterOption(name="i", value=i),
                FilterOption(name="fc", value=fc),
            ],
        )
    )
    return graph


def vfrdet(
    graph: Stream,
):
    """Variable frame rate detect filter."""
    graph.append(Filter(command="vfrdet", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def atilt(
    graph: Stream,
    freq: Optional[float] = None,
    slope: Optional[float] = None,
    width: Optional[float] = None,
    order: Optional[int] = None,
    level: Optional[float] = None,
):
    """Apply spectral tilt to audio.

    :param float freq: set central frequency
    :param float slope: set filter slope
    :param float width: set filter width
    :param int order: set filter order
    :param float level: set input level
    """
    graph.append(
        Filter(
            command="atilt",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="freq", value=freq),
                FilterOption(name="slope", value=slope),
                FilterOption(name="width", value=width),
                FilterOption(name="order", value=order),
                FilterOption(name="level", value=level),
            ],
        )
    )
    return graph


def remap(graph: Stream, format: Optional[str] = None, fill: Optional[str] = None):
    """Remap pixels.

    :param str format: set output format
            possible values: color, gray
    :param str fill: set the color of the unmapped pixels
    """
    graph.append(
        Filter(
            command="remap",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="format", value=format), FilterOption(name="fill", value=fill)],
        )
    )
    return graph


def bilateral(graph: Stream, sigmaS: Optional[float] = None, sigmaR: Optional[float] = None, planes: Optional[int] = None):
    """Apply Bilateral filter.

    :param float sigmaS: set spatial sigma
    :param float sigmaR: set range sigma
    :param int planes: set planes to filter
    """
    graph.append(
        Filter(
            command="bilateral",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="sigmaS", value=sigmaS),
                FilterOption(name="sigmaR", value=sigmaR),
                FilterOption(name="planes", value=planes),
            ],
        )
    )
    return graph


def huesaturation(
    graph: Stream,
    hue: Optional[float] = None,
    saturation: Optional[float] = None,
    intensity: Optional[float] = None,
    colors: Optional[str] = None,
    strength: Optional[float] = None,
    rw: Optional[float] = None,
    gw: Optional[float] = None,
    bw: Optional[float] = None,
    lightness: Optional[bool] = None,
):
    """Apply hue-saturation-intensity adjustments.

    :param float hue: set the hue shift
    :param float saturation: set the saturation shift
    :param float intensity: set the intensity shift
    :param str colors: set colors range
            possible values: r, y, g, c, b, m, a
    :param float strength: set the filtering strength
    :param float rw: set the red weight
    :param float gw: set the green weight
    :param float bw: set the blue weight
    :param bool lightness: set the preserve lightness
    """
    graph.append(
        Filter(
            command="huesaturation",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="hue", value=hue),
                FilterOption(name="saturation", value=saturation),
                FilterOption(name="intensity", value=intensity),
                FilterOption(name="colors", value=colors),
                FilterOption(name="strength", value=strength),
                FilterOption(name="rw", value=rw),
                FilterOption(name="gw", value=gw),
                FilterOption(name="bw", value=bw),
                FilterOption(name="lightness", value=lightness),
            ],
        )
    )
    return graph


def epx(graph: Stream, n: Optional[int] = None):
    """Scale the input using EPX algorithm.

    :param int n: set scale factor
    """
    graph.append(Filter(command="epx", filter_type="AVMEDIA_TYPE_VIDEO", params=[FilterOption(name="n", value=n)]))
    return graph


def lagfun(graph: Stream, decay: Optional[float] = None, planes: Optional[str] = None):
    """Slowly update darker pixels.

    :param float decay: set decay
    :param str planes: set what planes to filter
    """
    graph.append(
        Filter(
            command="lagfun",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="decay", value=decay), FilterOption(name="planes", value=planes)],
        )
    )
    return graph


def alphamerge(
    graph: Stream,
):
    """Copy the luma value of the second input into the alpha channel of the first input."""
    graph.append(Filter(command="alphamerge", filter_type="AVMEDIA_TYPE_VIDEO", params=[]))
    return graph


def signalstats(graph: Stream, stat: Optional[str] = None, out: Optional[str] = None, c: Optional[str] = None):
    """Generate statistics from video analysis.

    :param str stat: set statistics filters
            possible values: tout, vrep, brng
    :param str out: set video filter
            possible values: tout, vrep, brng
    :param str c: set highlight color
    """
    graph.append(
        Filter(
            command="signalstats",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="stat", value=stat), FilterOption(name="out", value=out), FilterOption(name="c", value=c)],
        )
    )
    return graph


def asetnsamples(graph: Stream, nb_out_samples: Optional[int] = None, pad: Optional[bool] = None):
    """Set the number of samples for each output audio frames.

    :param int nb_out_samples: set the number of per-frame output samples
    :param bool pad: pad last frame with zeros
    """
    graph.append(
        Filter(
            command="asetnsamples",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="nb_out_samples", value=nb_out_samples), FilterOption(name="pad", value=pad)],
        )
    )
    return graph


def adenorm(graph: Stream, level: Optional[float] = None, type: Optional[str] = None):
    """Remedy denormals by adding extremely low-level noise.

    :param float level: set level
    :param str type: set type
            possible values: dc, ac, square, pulse
    """
    graph.append(
        Filter(
            command="adenorm",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[FilterOption(name="level", value=level), FilterOption(name="type", value=type)],
        )
    )
    return graph


def surround(
    graph: Stream,
    chl_out: Optional[str] = None,
    chl_in: Optional[str] = None,
    level_in: Optional[float] = None,
    level_out: Optional[float] = None,
    lfe: Optional[bool] = None,
    lfe_low: Optional[int] = None,
    lfe_high: Optional[int] = None,
    lfe_mode: Optional[str] = None,
    smooth: Optional[float] = None,
    angle: Optional[float] = None,
    focus: Optional[float] = None,
    fc_in: Optional[float] = None,
    fc_out: Optional[float] = None,
    fl_in: Optional[float] = None,
    fl_out: Optional[float] = None,
    fr_in: Optional[float] = None,
    fr_out: Optional[float] = None,
    sl_in: Optional[float] = None,
    sl_out: Optional[float] = None,
    sr_in: Optional[float] = None,
    sr_out: Optional[float] = None,
    bl_in: Optional[float] = None,
    bl_out: Optional[float] = None,
    br_in: Optional[float] = None,
    br_out: Optional[float] = None,
    bc_in: Optional[float] = None,
    bc_out: Optional[float] = None,
    lfe_in: Optional[float] = None,
    lfe_out: Optional[float] = None,
    allx: Optional[float] = None,
    ally: Optional[float] = None,
    fcx: Optional[float] = None,
    flx: Optional[float] = None,
    frx: Optional[float] = None,
    blx: Optional[float] = None,
    brx: Optional[float] = None,
    slx: Optional[float] = None,
    srx: Optional[float] = None,
    bcx: Optional[float] = None,
    fcy: Optional[float] = None,
    fly: Optional[float] = None,
    fry: Optional[float] = None,
    bly: Optional[float] = None,
    bry: Optional[float] = None,
    sly: Optional[float] = None,
    sry: Optional[float] = None,
    bcy: Optional[float] = None,
    win_size: Optional[int] = None,
    win_func: Optional[str] = None,
    overlap: Optional[float] = None,
):
    """Apply audio surround upmix filter.

    :param str chl_out: set output channel layout
    :param str chl_in: set input channel layout
    :param float level_in: set input level
    :param float level_out: set output level
    :param bool lfe: output LFE
    :param int lfe_low: LFE low cut off
    :param int lfe_high: LFE high cut off
    :param str lfe_mode: set LFE channel mode
            possible values: add, sub
    :param float smooth: set temporal smoothness strength
    :param float angle: set soundfield transform angle
    :param float focus: set soundfield transform focus
    :param float fc_in: set front center channel input level
    :param float fc_out: set front center channel output level
    :param float fl_in: set front left channel input level
    :param float fl_out: set front left channel output level
    :param float fr_in: set front right channel input level
    :param float fr_out: set front right channel output level
    :param float sl_in: set side left channel input level
    :param float sl_out: set side left channel output level
    :param float sr_in: set side right channel input level
    :param float sr_out: set side right channel output level
    :param float bl_in: set back left channel input level
    :param float bl_out: set back left channel output level
    :param float br_in: set back right channel input level
    :param float br_out: set back right channel output level
    :param float bc_in: set back center channel input level
    :param float bc_out: set back center channel output level
    :param float lfe_in: set lfe channel input level
    :param float lfe_out: set lfe channel output level
    :param float allx: set all channel's x spread
    :param float ally: set all channel's y spread
    :param float fcx: set front center channel x spread
    :param float flx: set front left channel x spread
    :param float frx: set front right channel x spread
    :param float blx: set back left channel x spread
    :param float brx: set back right channel x spread
    :param float slx: set side left channel x spread
    :param float srx: set side right channel x spread
    :param float bcx: set back center channel x spread
    :param float fcy: set front center channel y spread
    :param float fly: set front left channel y spread
    :param float fry: set front right channel y spread
    :param float bly: set back left channel y spread
    :param float bry: set back right channel y spread
    :param float sly: set side left channel y spread
    :param float sry: set side right channel y spread
    :param float bcy: set back center channel y spread
    :param int win_size: set window size
    :param str win_func: set window function
            possible values: rect, bartlett, hann, hanning, hamming, blackman, welch, flattop, bharris, bnuttall, bhann, sine, nuttall, lanczos, gauss, tukey, dolph, cauchy, parzen, poisson, bohman, kaiser
    :param float overlap: set window overlap
    """
    graph.append(
        Filter(
            command="surround",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="chl_out", value=chl_out),
                FilterOption(name="chl_in", value=chl_in),
                FilterOption(name="level_in", value=level_in),
                FilterOption(name="level_out", value=level_out),
                FilterOption(name="lfe", value=lfe),
                FilterOption(name="lfe_low", value=lfe_low),
                FilterOption(name="lfe_high", value=lfe_high),
                FilterOption(name="lfe_mode", value=lfe_mode),
                FilterOption(name="smooth", value=smooth),
                FilterOption(name="angle", value=angle),
                FilterOption(name="focus", value=focus),
                FilterOption(name="fc_in", value=fc_in),
                FilterOption(name="fc_out", value=fc_out),
                FilterOption(name="fl_in", value=fl_in),
                FilterOption(name="fl_out", value=fl_out),
                FilterOption(name="fr_in", value=fr_in),
                FilterOption(name="fr_out", value=fr_out),
                FilterOption(name="sl_in", value=sl_in),
                FilterOption(name="sl_out", value=sl_out),
                FilterOption(name="sr_in", value=sr_in),
                FilterOption(name="sr_out", value=sr_out),
                FilterOption(name="bl_in", value=bl_in),
                FilterOption(name="bl_out", value=bl_out),
                FilterOption(name="br_in", value=br_in),
                FilterOption(name="br_out", value=br_out),
                FilterOption(name="bc_in", value=bc_in),
                FilterOption(name="bc_out", value=bc_out),
                FilterOption(name="lfe_in", value=lfe_in),
                FilterOption(name="lfe_out", value=lfe_out),
                FilterOption(name="allx", value=allx),
                FilterOption(name="ally", value=ally),
                FilterOption(name="fcx", value=fcx),
                FilterOption(name="flx", value=flx),
                FilterOption(name="frx", value=frx),
                FilterOption(name="blx", value=blx),
                FilterOption(name="brx", value=brx),
                FilterOption(name="slx", value=slx),
                FilterOption(name="srx", value=srx),
                FilterOption(name="bcx", value=bcx),
                FilterOption(name="fcy", value=fcy),
                FilterOption(name="fly", value=fly),
                FilterOption(name="fry", value=fry),
                FilterOption(name="bly", value=bly),
                FilterOption(name="bry", value=bry),
                FilterOption(name="sly", value=sly),
                FilterOption(name="sry", value=sry),
                FilterOption(name="bcy", value=bcy),
                FilterOption(name="win_size", value=win_size),
                FilterOption(name="win_func", value=win_func),
                FilterOption(name="overlap", value=overlap),
            ],
        )
    )
    return graph


def noise(
    graph: Stream,
    all_seed: Optional[int] = None,
    all_strength: Optional[int] = None,
    alls: Optional[int] = None,
    all_flags: Optional[str] = None,
    allf: Optional[str] = None,
    c0_seed: Optional[int] = None,
    c0_strength: Optional[int] = None,
    c0s: Optional[int] = None,
    c0_flags: Optional[str] = None,
    c0f: Optional[str] = None,
    c1_seed: Optional[int] = None,
    c1_strength: Optional[int] = None,
    c1s: Optional[int] = None,
    c1_flags: Optional[str] = None,
    c1f: Optional[str] = None,
    c2_seed: Optional[int] = None,
    c2_strength: Optional[int] = None,
    c2s: Optional[int] = None,
    c2_flags: Optional[str] = None,
    c2f: Optional[str] = None,
    c3_seed: Optional[int] = None,
    c3_strength: Optional[int] = None,
    c3s: Optional[int] = None,
    c3_flags: Optional[str] = None,
    c3f: Optional[str] = None,
):
    """Add noise.

    :param int all_seed: set component #0 noise seed
    :param int all_strength: set component #0 strength
    :param int alls: set component #0 strength
    :param str all_flags: set component #0 flags
            possible values: a, p, t, u
    :param str allf: set component #0 flags
            possible values: a, p, t, u
    :param int c0_seed: set component #0 noise seed
    :param int c0_strength: set component #0 strength
    :param int c0s: set component #0 strength
    :param str c0_flags: set component #0 flags
            possible values: a, p, t, u
    :param str c0f: set component #0 flags
            possible values: a, p, t, u
    :param int c1_seed: set component #1 noise seed
    :param int c1_strength: set component #1 strength
    :param int c1s: set component #1 strength
    :param str c1_flags: set component #1 flags
            possible values: a, p, t, u
    :param str c1f: set component #1 flags
            possible values: a, p, t, u
    :param int c2_seed: set component #2 noise seed
    :param int c2_strength: set component #2 strength
    :param int c2s: set component #2 strength
    :param str c2_flags: set component #2 flags
            possible values: a, p, t, u
    :param str c2f: set component #2 flags
            possible values: a, p, t, u
    :param int c3_seed: set component #3 noise seed
    :param int c3_strength: set component #3 strength
    :param int c3s: set component #3 strength
    :param str c3_flags: set component #3 flags
            possible values: a, p, t, u
    :param str c3f: set component #3 flags
            possible values: a, p, t, u
    """
    graph.append(
        Filter(
            command="noise",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="all_seed", value=all_seed),
                FilterOption(name="all_strength", value=all_strength),
                FilterOption(name="alls", value=alls),
                FilterOption(name="all_flags", value=all_flags),
                FilterOption(name="allf", value=allf),
                FilterOption(name="c0_seed", value=c0_seed),
                FilterOption(name="c0_strength", value=c0_strength),
                FilterOption(name="c0s", value=c0s),
                FilterOption(name="c0_flags", value=c0_flags),
                FilterOption(name="c0f", value=c0f),
                FilterOption(name="c1_seed", value=c1_seed),
                FilterOption(name="c1_strength", value=c1_strength),
                FilterOption(name="c1s", value=c1s),
                FilterOption(name="c1_flags", value=c1_flags),
                FilterOption(name="c1f", value=c1f),
                FilterOption(name="c2_seed", value=c2_seed),
                FilterOption(name="c2_strength", value=c2_strength),
                FilterOption(name="c2s", value=c2s),
                FilterOption(name="c2_flags", value=c2_flags),
                FilterOption(name="c2f", value=c2f),
                FilterOption(name="c3_seed", value=c3_seed),
                FilterOption(name="c3_strength", value=c3_strength),
                FilterOption(name="c3s", value=c3s),
                FilterOption(name="c3_flags", value=c3_flags),
                FilterOption(name="c3f", value=c3f),
            ],
        )
    )
    return graph


def pp7(graph: Stream, qp: Optional[int] = None, mode: Optional[str] = None):
    """Apply Postprocessing 7 filter.

    :param int qp: force a constant quantizer parameter
    :param str mode: set thresholding mode
            possible values: hard, soft, medium
    """
    graph.append(
        Filter(
            command="pp7",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="qp", value=qp), FilterOption(name="mode", value=mode)],
        )
    )
    return graph


def showinfo(graph: Stream, checksum: Optional[bool] = None, udu_sei_as_ascii: Optional[bool] = None):
    """Show textual information for each video frame.

    :param bool checksum: calculate checksums
    :param bool udu_sei_as_ascii: try to print user data unregistered SEI as ascii character when possible
    """
    graph.append(
        Filter(
            command="showinfo",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[FilterOption(name="checksum", value=checksum), FilterOption(name="udu_sei_as_ascii", value=udu_sei_as_ascii)],
        )
    )
    return graph


def mergeplanes(
    graph: Stream,
    mapping: Optional[int] = None,
    format: Optional[str] = None,
    map0s: Optional[int] = None,
    map0p: Optional[int] = None,
    map1s: Optional[int] = None,
    map1p: Optional[int] = None,
    map2s: Optional[int] = None,
    map2p: Optional[int] = None,
    map3s: Optional[int] = None,
    map3p: Optional[int] = None,
):
    """Merge planes.

    :param int mapping: set input to output plane mapping
    :param str format: set output pixel format
    :param int map0s: set 1st input to output stream mapping
    :param int map0p: set 1st input to output plane mapping
    :param int map1s: set 2nd input to output stream mapping
    :param int map1p: set 2nd input to output plane mapping
    :param int map2s: set 3rd input to output stream mapping
    :param int map2p: set 3rd input to output plane mapping
    :param int map3s: set 4th input to output stream mapping
    :param int map3p: set 4th input to output plane mapping
    """
    graph.append(
        Filter(
            command="mergeplanes",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="mapping", value=mapping),
                FilterOption(name="format", value=format),
                FilterOption(name="map0s", value=map0s),
                FilterOption(name="map0p", value=map0p),
                FilterOption(name="map1s", value=map1s),
                FilterOption(name="map1p", value=map1p),
                FilterOption(name="map2s", value=map2s),
                FilterOption(name="map2p", value=map2p),
                FilterOption(name="map3s", value=map3s),
                FilterOption(name="map3p", value=map3p),
            ],
        )
    )
    return graph


def afwtdn(
    graph: Stream,
    sigma: Optional[float] = None,
    levels: Optional[int] = None,
    wavet: Optional[str] = None,
    percent: Optional[float] = None,
    profile: Optional[bool] = None,
    adaptive: Optional[bool] = None,
    samples: Optional[int] = None,
    softness: Optional[float] = None,
):
    """Denoise audio stream using Wavelets.

    :param float sigma: set noise sigma
    :param int levels: set number of wavelet levels
    :param str wavet: set wavelet type
            possible values: sym2, sym4, rbior68, deb10, sym10, coif5, bl3
    :param float percent: set percent of full denoising
    :param bool profile: profile noise
    :param bool adaptive: adaptive profiling of noise
    :param int samples: set frame size in number of samples
    :param float softness: set thresholding softness
    """
    graph.append(
        Filter(
            command="afwtdn",
            filter_type="AVMEDIA_TYPE_AUDIO",
            params=[
                FilterOption(name="sigma", value=sigma),
                FilterOption(name="levels", value=levels),
                FilterOption(name="wavet", value=wavet),
                FilterOption(name="percent", value=percent),
                FilterOption(name="profile", value=profile),
                FilterOption(name="adaptive", value=adaptive),
                FilterOption(name="samples", value=samples),
                FilterOption(name="softness", value=softness),
            ],
        )
    )
    return graph


def hsvkey(
    graph: Stream,
    hue: Optional[float] = None,
    sat: Optional[float] = None,
    val: Optional[float] = None,
    similarity: Optional[float] = None,
    blend: Optional[float] = None,
):
    """Turns a certain HSV range into transparency. Operates on YUV colors.

    :param float hue: set the hue value
    :param float sat: set the saturation value
    :param float val: set the value value
    :param float similarity: set the hsvkey similarity value
    :param float blend: set the hsvkey blend value
    """
    graph.append(
        Filter(
            command="hsvkey",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="hue", value=hue),
                FilterOption(name="sat", value=sat),
                FilterOption(name="val", value=val),
                FilterOption(name="similarity", value=similarity),
                FilterOption(name="blend", value=blend),
            ],
        )
    )
    return graph


def hsvhold(
    graph: Stream,
    hue: Optional[float] = None,
    sat: Optional[float] = None,
    val: Optional[float] = None,
    similarity: Optional[float] = None,
    blend: Optional[float] = None,
):
    """Turns a certain HSV range into gray.

    :param float hue: set the hue value
    :param float sat: set the saturation value
    :param float val: set the value value
    :param float similarity: set the hsvhold similarity value
    :param float blend: set the hsvhold blend value
    """
    graph.append(
        Filter(
            command="hsvhold",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="hue", value=hue),
                FilterOption(name="sat", value=sat),
                FilterOption(name="val", value=val),
                FilterOption(name="similarity", value=similarity),
                FilterOption(name="blend", value=blend),
            ],
        )
    )
    return graph


def vaguedenoiser(
    graph: Stream,
    threshold: Optional[float] = None,
    method: Optional[str] = None,
    nsteps: Optional[int] = None,
    percent: Optional[float] = None,
    planes: Optional[int] = None,
    type: Optional[str] = None,
):
    """Apply a Wavelet based Denoiser.

    :param float threshold: set filtering strength
    :param str method: set filtering method
            possible values: hard, soft, garrote
    :param int nsteps: set number of steps
    :param float percent: set percent of full denoising
    :param int planes: set planes to filter
    :param str type: set threshold type
            possible values: universal, bayes
    """
    graph.append(
        Filter(
            command="vaguedenoiser",
            filter_type="AVMEDIA_TYPE_VIDEO",
            params=[
                FilterOption(name="threshold", value=threshold),
                FilterOption(name="method", value=method),
                FilterOption(name="nsteps", value=nsteps),
                FilterOption(name="percent", value=percent),
                FilterOption(name="planes", value=planes),
                FilterOption(name="type", value=type),
            ],
        )
    )
    return graph
