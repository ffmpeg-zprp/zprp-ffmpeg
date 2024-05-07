from zprp_ffmpeg.filter_classes import FilterOption
from zprp_ffmpeg.FilterGraph import Filter
from zprp_ffmpeg.FilterGraph import Stream


def hwmap(graph: Stream, mode: str, derive_device: str, reverse: int):
    """Map hardware frames
    :param str mode: Frame mapping mode
    :param str derive_device: Derive a new device of this type
    :param int reverse: Map in reverse (create and allocate in the sink)"""
    filter = Filter(
        command="hwmap",
        params=[FilterOption(name="mode", type=str), FilterOption(name="derive_device", type=str), FilterOption(name="reverse", type=int)],
    )
    graph.append(filter)
    return graph


def vibrato(graph: Stream, f: float, d: float):
    """Apply vibrato effect.
    :param float f: set frequency in hertz
    :param float d: set depth as percentage"""
    filter = Filter(command="vibrato", params=[FilterOption(name="f", type=float), FilterOption(name="d", type=float)])
    graph.append(filter)
    return graph


def sidechaincompress(
    graph: Stream,
):
    """Sidechain compressor."""
    filter = Filter(command="sidechaincompress", params=[])
    graph.append(filter)
    return graph


def acompressor(
    graph: Stream,
):
    """Audio compressor."""
    filter = Filter(command="acompressor", params=[])
    graph.append(filter)
    return graph


def colormap(graph: Stream, patch_size: int, nb_patches: int, type: int, kernel: int):
    """Apply custom Color Maps to video stream.
    :param int patch_size: set patch size
    :param int nb_patches: set number of patches
    :param int type: set the target type used
    :param int kernel: set the kernel used for measuring color difference"""
    filter = Filter(
        command="colormap",
        params=[
            FilterOption(name="patch_size", type=int),
            FilterOption(name="nb_patches", type=int),
            FilterOption(name="type", type=int),
            FilterOption(name="kernel", type=int),
        ],
    )
    graph.append(filter)
    return graph


def separatefields(
    graph: Stream,
):
    """Split input video frames into fields."""
    filter = Filter(command="separatefields", params=[])
    graph.append(filter)
    return graph


def tiltandshift(graph: Stream, tilt: int, start: int, end: int, hold: int, pad: int):
    """Generate a tilt-and-shift'd video.
    :param int tilt: Tilt the video horizontally while shifting
    :param int start: Action at the start of input
    :param int end: Action at the end of input
    :param int hold: Number of columns to hold at the start of the video
    :param int pad: Number of columns to pad at the end of the video"""
    filter = Filter(
        command="tiltandshift",
        params=[
            FilterOption(name="tilt", type=int),
            FilterOption(name="start", type=int),
            FilterOption(name="end", type=int),
            FilterOption(name="hold", type=int),
            FilterOption(name="pad", type=int),
        ],
    )
    graph.append(filter)
    return graph


def w3fdif(graph: Stream, filter: int, mode: int, parity: int, deint: int):
    """Apply Martin Weston three field deinterlace.
    :param int filter: specify the filter
    :param int mode: specify the interlacing mode
    :param int parity: specify the assumed picture field parity
    :param int deint: specify which frames to deinterlace"""
    filter = Filter(
        command="w3fdif",
        params=[
            FilterOption(name="filter", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="parity", type=int),
            FilterOption(name="deint", type=int),
        ],
    )
    graph.append(filter)
    return graph


def aap(graph: Stream, order: int, projection: int, mu: float, delta: float, out_mode: int, precision: int):
    """Apply Affine Projection algorithm to first audio stream.
    :param int order: set the filter order
    :param int projection: set the filter projection
    :param float mu: set the filter mu
    :param float delta: set the filter delta
    :param int out_mode: set output mode
    :param int precision: set processing precision"""
    filter = Filter(
        command="aap",
        params=[
            FilterOption(name="order", type=int),
            FilterOption(name="projection", type=int),
            FilterOption(name="mu", type=float),
            FilterOption(name="delta", type=float),
            FilterOption(name="out_mode", type=int),
            FilterOption(name="precision", type=int),
        ],
    )
    graph.append(filter)
    return graph


def drawbox(graph: Stream, x: str, y: str, width: str, height: str, color: str, thickness: str, replace: bool, box_source: str):
    """Draw a colored box on the input video.
    :param str x: set horizontal position of the left box edge
    :param str y: set vertical position of the top box edge
    :param str width: set width of the box
    :param str height: set height of the box
    :param str color: set color of the box
    :param str thickness: set the box thickness
    :param bool replace: replace color & alpha
    :param str box_source: use datas from bounding box in side data"""
    filter = Filter(
        command="drawbox",
        params=[
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="width", type=str),
            FilterOption(name="height", type=str),
            FilterOption(name="color", type=str),
            FilterOption(name="thickness", type=str),
            FilterOption(name="replace", type=bool),
            FilterOption(name="box_source", type=str),
        ],
    )
    graph.append(filter)
    return graph


def drawgrid(graph: Stream, x: str, y: str, width: str, height: str, color: str, thickness: str, replace: bool):
    """Draw a colored grid on the input video.
    :param str x: set horizontal offset
    :param str y: set vertical offset
    :param str width: set width of grid cell
    :param str height: set height of grid cell
    :param str color: set color of the grid
    :param str thickness: set grid line thickness
    :param bool replace: replace color & alpha"""
    filter = Filter(
        command="drawgrid",
        params=[
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="width", type=str),
            FilterOption(name="height", type=str),
            FilterOption(name="color", type=str),
            FilterOption(name="thickness", type=str),
            FilterOption(name="replace", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def earwax(
    graph: Stream,
):
    """Widen the stereo image."""
    filter = Filter(command="earwax", params=[])
    graph.append(filter)
    return graph


def bbox(graph: Stream, min_val: int):
    """Compute bounding box for each frame.
    :param int min_val: set minimum luminance value for bounding box"""
    filter = Filter(command="bbox", params=[FilterOption(name="min_val", type=int)])
    graph.append(filter)
    return graph


def mandelbrot(
    graph: Stream,
    size: int,
    rate: str,
    maxiter: int,
    start_x: float,
    start_y: float,
    start_scale: float,
    end_scale: float,
    end_pts: float,
    bailout: float,
    morphxf: float,
    morphyf: float,
    morphamp: float,
    outer: int,
    inner: int,
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
    :param int outer: set outer coloring mode
    :param int inner: set inner coloring mode"""
    filter = Filter(
        command="mandelbrot",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="maxiter", type=int),
            FilterOption(name="start_x", type=float),
            FilterOption(name="start_y", type=float),
            FilterOption(name="start_scale", type=float),
            FilterOption(name="end_scale", type=float),
            FilterOption(name="end_pts", type=float),
            FilterOption(name="bailout", type=float),
            FilterOption(name="morphxf", type=float),
            FilterOption(name="morphyf", type=float),
            FilterOption(name="morphamp", type=float),
            FilterOption(name="outer", type=int),
            FilterOption(name="inner", type=int),
        ],
    )
    graph.append(filter)
    return graph


def haas(
    graph: Stream,
    level_in: float,
    level_out: float,
    side_gain: float,
    middle_source: int,
    middle_phase: bool,
    left_delay: float,
    left_balance: float,
    left_gain: float,
    left_phase: bool,
    right_delay: float,
    right_balance: float,
    right_gain: float,
    right_phase: bool,
):
    """Apply Haas Stereo Enhancer.
    :param float level_in: set level in
    :param float level_out: set level out
    :param float side_gain: set side gain
    :param int middle_source: set middle source
    :param bool middle_phase: set middle phase
    :param float left_delay: set left delay
    :param float left_balance: set left balance
    :param float left_gain: set left gain
    :param bool left_phase: set left phase
    :param float right_delay: set right delay
    :param float right_balance: set right balance
    :param float right_gain: set right gain
    :param bool right_phase: set right phase"""
    filter = Filter(
        command="haas",
        params=[
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="side_gain", type=float),
            FilterOption(name="middle_source", type=int),
            FilterOption(name="middle_phase", type=bool),
            FilterOption(name="left_delay", type=float),
            FilterOption(name="left_balance", type=float),
            FilterOption(name="left_gain", type=float),
            FilterOption(name="left_phase", type=bool),
            FilterOption(name="right_delay", type=float),
            FilterOption(name="right_balance", type=float),
            FilterOption(name="right_gain", type=float),
            FilterOption(name="right_phase", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def random(graph: Stream, frames: int, seed: int):
    """Return random frames.
    :param int frames: set number of frames in cache
    :param int seed: set the seed"""
    filter = Filter(command="random", params=[FilterOption(name="frames", type=int), FilterOption(name="seed", type=int)])
    graph.append(filter)
    return graph


def aphaser(graph: Stream, in_gain: float, out_gain: float, delay: float, decay: float, speed: float, type: int):
    """Add a phasing effect to the audio.
    :param float in_gain: set input gain
    :param float out_gain: set output gain
    :param float delay: set delay in milliseconds
    :param float decay: set decay
    :param float speed: set modulation speed
    :param int type: set modulation type"""
    filter = Filter(
        command="aphaser",
        params=[
            FilterOption(name="in_gain", type=float),
            FilterOption(name="out_gain", type=float),
            FilterOption(name="delay", type=float),
            FilterOption(name="decay", type=float),
            FilterOption(name="speed", type=float),
            FilterOption(name="type", type=int),
        ],
    )
    graph.append(filter)
    return graph


def feedback(graph: Stream, x: int, y: int, w: int, h: int):
    """Apply feedback video filter.
    :param int x: set top left crop position
    :param int y: set top left crop position
    :param int w: set crop size
    :param int h: set crop size"""
    filter = Filter(
        command="feedback",
        params=[
            FilterOption(name="x", type=int),
            FilterOption(name="y", type=int),
            FilterOption(name="w", type=int),
            FilterOption(name="h", type=int),
        ],
    )
    graph.append(filter)
    return graph


def aexciter(
    graph: Stream, level_in: float, level_out: float, amount: float, drive: float, blend: float, freq: float, ceil: float, listen: bool
):
    """Enhance high frequency part of audio.
    :param float level_in: set level in
    :param float level_out: set level out
    :param float amount: set amount
    :param float drive: set harmonics
    :param float blend: set blend harmonics
    :param float freq: set scope
    :param float ceil: set ceiling
    :param bool listen: enable listen mode"""
    filter = Filter(
        command="aexciter",
        params=[
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="amount", type=float),
            FilterOption(name="drive", type=float),
            FilterOption(name="blend", type=float),
            FilterOption(name="freq", type=float),
            FilterOption(name="ceil", type=float),
            FilterOption(name="listen", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def aderivative(
    graph: Stream,
):
    """Compute derivative of input audio."""
    filter = Filter(command="aderivative", params=[])
    graph.append(filter)
    return graph


def aintegral(
    graph: Stream,
):
    """Compute integral of input audio."""
    filter = Filter(command="aintegral", params=[])
    graph.append(filter)
    return graph


def palettegen(graph: Stream, max_colors: int, reserve_transparent: bool, transparency_color: str, stats_mode: int):
    """Find the optimal palette for a given stream.
    :param int max_colors: set the maximum number of colors to use in the palette
    :param bool reserve_transparent: reserve a palette entry for transparency
    :param str transparency_color: set a background color for transparency
    :param int stats_mode: set statistics mode"""
    filter = Filter(
        command="palettegen",
        params=[
            FilterOption(name="max_colors", type=int),
            FilterOption(name="reserve_transparent", type=bool),
            FilterOption(name="transparency_color", type=str),
            FilterOption(name="stats_mode", type=int),
        ],
    )
    graph.append(filter)
    return graph


def adrc(graph: Stream, transfer: str, attack: float, release: float, channels: str):
    """Audio Spectral Dynamic Range Controller.
    :param str transfer: set the transfer expression
    :param float attack: set the attack
    :param float release: set the release
    :param str channels: set channels to filter"""
    filter = Filter(
        command="adrc",
        params=[
            FilterOption(name="transfer", type=str),
            FilterOption(name="attack", type=float),
            FilterOption(name="release", type=float),
            FilterOption(name="channels", type=str),
        ],
    )
    graph.append(filter)
    return graph


def loudnorm(
    graph: Stream,
    I: float,
    LRA: float,
    TP: float,
    measured_I: float,
    measured_LRA: float,
    measured_TP: float,
    measured_thresh: float,
    offset: float,
    linear: bool,
    dual_mono: bool,
    print_format: int,
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
    :param int print_format: set print format for stats"""
    filter = Filter(
        command="loudnorm",
        params=[
            FilterOption(name="I", type=float),
            FilterOption(name="LRA", type=float),
            FilterOption(name="TP", type=float),
            FilterOption(name="measured_I", type=float),
            FilterOption(name="measured_LRA", type=float),
            FilterOption(name="measured_TP", type=float),
            FilterOption(name="measured_thresh", type=float),
            FilterOption(name="offset", type=float),
            FilterOption(name="linear", type=bool),
            FilterOption(name="dual_mono", type=bool),
            FilterOption(name="print_format", type=int),
        ],
    )
    graph.append(filter)
    return graph


def colorcontrast(graph: Stream, rc: float, gm: float, by: float, rcw: float, gmw: float, byw: float, pl: float):
    """Adjust color contrast between RGB components.
    :param float rc: set the red-cyan contrast
    :param float gm: set the green-magenta contrast
    :param float by: set the blue-yellow contrast
    :param float rcw: set the red-cyan weight
    :param float gmw: set the green-magenta weight
    :param float byw: set the blue-yellow weight
    :param float pl: set the amount of preserving lightness"""
    filter = Filter(
        command="colorcontrast",
        params=[
            FilterOption(name="rc", type=float),
            FilterOption(name="gm", type=float),
            FilterOption(name="by", type=float),
            FilterOption(name="rcw", type=float),
            FilterOption(name="gmw", type=float),
            FilterOption(name="byw", type=float),
            FilterOption(name="pl", type=float),
        ],
    )
    graph.append(filter)
    return graph


def scroll(graph: Stream, horizontal: float, vertical: float, hpos: float, vpos: float):
    """Scroll input video.
    :param float horizontal: set the horizontal scrolling speed
    :param float vertical: set the vertical scrolling speed
    :param float hpos: set initial horizontal position
    :param float vpos: set initial vertical position"""
    filter = Filter(
        command="scroll",
        params=[
            FilterOption(name="horizontal", type=float),
            FilterOption(name="vertical", type=float),
            FilterOption(name="hpos", type=float),
            FilterOption(name="vpos", type=float),
        ],
    )
    graph.append(filter)
    return graph


def shufflepixels(graph: Stream, direction: int, mode: int, width: int, height: int, seed: int):
    """Shuffle video pixels.
    :param int direction: set shuffle direction
    :param int mode: set shuffle mode
    :param int width: set block width
    :param int height: set block height
    :param int seed: set random seed"""
    filter = Filter(
        command="shufflepixels",
        params=[
            FilterOption(name="direction", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="width", type=int),
            FilterOption(name="height", type=int),
            FilterOption(name="seed", type=int),
        ],
    )
    graph.append(filter)
    return graph


def sinc(
    graph: Stream,
    sample_rate: int,
    nb_samples: int,
    hp: float,
    lp: float,
    phase: float,
    beta: float,
    att: float,
    round: bool,
    hptaps: int,
    lptaps: int,
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
    :param int lptaps: set number of taps for low-pass filter"""
    filter = Filter(
        command="sinc",
        params=[
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="hp", type=float),
            FilterOption(name="lp", type=float),
            FilterOption(name="phase", type=float),
            FilterOption(name="beta", type=float),
            FilterOption(name="att", type=float),
            FilterOption(name="round", type=bool),
            FilterOption(name="hptaps", type=int),
            FilterOption(name="lptaps", type=int),
        ],
    )
    graph.append(filter)
    return graph


def hilbert(graph: Stream, sample_rate: int, taps: int, nb_samples: int, win_func: int):
    """Generate a Hilbert transform FIR coefficients.
    :param int sample_rate: set sample rate
    :param int taps: set number of taps
    :param int nb_samples: set the number of samples per requested frame
    :param int win_func: set window function"""
    filter = Filter(
        command="hilbert",
        params=[
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="taps", type=int),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="win_func", type=int),
        ],
    )
    graph.append(filter)
    return graph


def exposure(graph: Stream, exposure: float, black: float):
    """Adjust exposure of the video stream.
    :param float exposure: set the exposure correction
    :param float black: set the black level correction"""
    filter = Filter(command="exposure", params=[FilterOption(name="exposure", type=float), FilterOption(name="black", type=float)])
    graph.append(filter)
    return graph


def boxblur(
    graph: Stream,
    luma_radius: str,
    lr: str,
    luma_power: int,
    lp: int,
    chroma_radius: str,
    cr: str,
    chroma_power: int,
    cp: int,
    alpha_radius: str,
    ar: str,
    alpha_power: int,
    ap: int,
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
    :param int ap: How many times should the boxblur be applied to alpha"""
    filter = Filter(
        command="boxblur",
        params=[
            FilterOption(name="luma_radius", type=str),
            FilterOption(name="lr", type=str),
            FilterOption(name="luma_power", type=int),
            FilterOption(name="lp", type=int),
            FilterOption(name="chroma_radius", type=str),
            FilterOption(name="cr", type=str),
            FilterOption(name="chroma_power", type=int),
            FilterOption(name="cp", type=int),
            FilterOption(name="alpha_radius", type=str),
            FilterOption(name="ar", type=str),
            FilterOption(name="alpha_power", type=int),
            FilterOption(name="ap", type=int),
        ],
    )
    graph.append(filter)
    return graph


def copy(
    graph: Stream,
):
    """Copy the input video unchanged to the output."""
    filter = Filter(command="copy", params=[])
    graph.append(filter)
    return graph


def addroi(graph: Stream, x: str, y: str, w: str, h: str, qoffset: int, clear: bool):
    """Add region of interest to frame.
    :param str x: Region distance from left edge of frame.
    :param str y: Region distance from top edge of frame.
    :param str w: Region width.
    :param str h: Region height.
    :param int qoffset: Quantisation offset to apply in the region.
    :param bool clear: Remove any existing regions of interest before adding the new one."""
    filter = Filter(
        command="addroi",
        params=[
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="w", type=str),
            FilterOption(name="h", type=str),
            FilterOption(name="qoffset", type=int),
            FilterOption(name="clear", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def afftfilt(graph: Stream, real: str, imag: str, win_size: int, win_func: int, overlap: float):
    """Apply arbitrary expressions to samples in frequency domain.
    :param str real: set channels real expressions
    :param str imag: set channels imaginary expressions
    :param int win_size: set window size
    :param int win_func: set window function
    :param float overlap: set window overlap"""
    filter = Filter(
        command="afftfilt",
        params=[
            FilterOption(name="real", type=str),
            FilterOption(name="imag", type=str),
            FilterOption(name="win_size", type=int),
            FilterOption(name="win_func", type=int),
            FilterOption(name="overlap", type=float),
        ],
    )
    graph.append(filter)
    return graph


def colortemperature(graph: Stream, temperature: float, mix: float, pl: float):
    """Adjust color temperature of video.
    :param float temperature: set the temperature in Kelvin
    :param float mix: set the mix with filtered output
    :param float pl: set the amount of preserving lightness"""
    filter = Filter(
        command="colortemperature",
        params=[FilterOption(name="temperature", type=float), FilterOption(name="mix", type=float), FilterOption(name="pl", type=float)],
    )
    graph.append(filter)
    return graph


def colorcorrect(graph: Stream, rl: float, bl: float, rh: float, bh: float, saturation: float, analyze: int):
    """Adjust color white balance selectively for blacks and whites.
    :param float rl: set the red shadow spot
    :param float bl: set the blue shadow spot
    :param float rh: set the red highlight spot
    :param float bh: set the blue highlight spot
    :param float saturation: set the amount of saturation
    :param int analyze: set the analyze mode"""
    filter = Filter(
        command="colorcorrect",
        params=[
            FilterOption(name="rl", type=float),
            FilterOption(name="bl", type=float),
            FilterOption(name="rh", type=float),
            FilterOption(name="bh", type=float),
            FilterOption(name="saturation", type=float),
            FilterOption(name="analyze", type=int),
        ],
    )
    graph.append(filter)
    return graph


def maskfun(graph: Stream, low: int, high: int, planes: int, fill: int, sum: int):
    """Create Mask.
    :param int low: set low threshold
    :param int high: set high threshold
    :param int planes: set planes
    :param int fill: set fill value
    :param int sum: set sum value"""
    filter = Filter(
        command="maskfun",
        params=[
            FilterOption(name="low", type=int),
            FilterOption(name="high", type=int),
            FilterOption(name="planes", type=int),
            FilterOption(name="fill", type=int),
            FilterOption(name="sum", type=int),
        ],
    )
    graph.append(filter)
    return graph


def scale_vaapi(
    graph: Stream,
    w: str,
    h: str,
    format: str,
    mode: int,
    out_color_matrix: str,
    out_range: int,
    out_color_primaries: str,
    out_color_transfer: str,
    out_chroma_location: str,
    force_original_aspect_ratio: int,
    force_divisible_by: int,
):
    """Scale to/from VAAPI surfaces.
    :param str w: Output video width
    :param str h: Output video height
    :param str format: Output video format (software format of hardware frames)
    :param int mode: Scaling mode
    :param str out_color_matrix: Output colour matrix coefficient set
    :param int out_range: Output colour range
    :param str out_color_primaries: Output colour primaries
    :param str out_color_transfer: Output colour transfer characteristics
    :param str out_chroma_location: Output chroma sample location
    :param int force_original_aspect_ratio: decrease or increase w/h if necessary to keep the original AR
    :param int force_divisible_by: enforce that the output resolution is divisible by a defined integer when force_original_aspect_ratio is used
    """
    filter = Filter(
        command="scale_vaapi",
        params=[
            FilterOption(name="w", type=str),
            FilterOption(name="h", type=str),
            FilterOption(name="format", type=str),
            FilterOption(name="mode", type=int),
            FilterOption(name="out_color_matrix", type=str),
            FilterOption(name="out_range", type=int),
            FilterOption(name="out_color_primaries", type=str),
            FilterOption(name="out_color_transfer", type=str),
            FilterOption(name="out_chroma_location", type=str),
            FilterOption(name="force_original_aspect_ratio", type=int),
            FilterOption(name="force_divisible_by", type=int),
        ],
    )
    graph.append(filter)
    return graph


def afirsrc(graph: Stream, taps: int, frequency: str, magnitude: str, phase: str, sample_rate: int, nb_samples: int, win_func: int):
    """Generate a FIR coefficients audio stream.
    :param int taps: set number of taps
    :param str frequency: set frequency points
    :param str magnitude: set magnitude values
    :param str phase: set phase values
    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param int win_func: set window function"""
    filter = Filter(
        command="afirsrc",
        params=[
            FilterOption(name="taps", type=int),
            FilterOption(name="frequency", type=str),
            FilterOption(name="magnitude", type=str),
            FilterOption(name="phase", type=str),
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="win_func", type=int),
        ],
    )
    graph.append(filter)
    return graph


def afireqsrc(graph: Stream, preset: int, gains: str, bands: str, taps: int, sample_rate: int, nb_samples: int, interp: int, phase: int):
    """Generate a FIR equalizer coefficients audio stream.
    :param int preset: set equalizer preset
    :param str gains: set gain values per band
    :param str bands: set central frequency values per band
    :param int taps: set number of taps
    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param int interp: set the interpolation
    :param int phase: set the phase"""
    filter = Filter(
        command="afireqsrc",
        params=[
            FilterOption(name="preset", type=int),
            FilterOption(name="gains", type=str),
            FilterOption(name="bands", type=str),
            FilterOption(name="taps", type=int),
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="interp", type=int),
            FilterOption(name="phase", type=int),
        ],
    )
    graph.append(filter)
    return graph


def crop(graph: Stream, out_w: str, out_h: str, x: str, y: str, keep_aspect: bool, exact: bool):
    """Crop the input video.
    :param str out_w: set the width crop area expression
    :param str out_h: set the height crop area expression
    :param str x: set the x crop area expression
    :param str y: set the y crop area expression
    :param bool keep_aspect: keep aspect ratio
    :param bool exact: do exact cropping"""
    filter = Filter(
        command="crop",
        params=[
            FilterOption(name="out_w", type=str),
            FilterOption(name="out_h", type=str),
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="keep_aspect", type=bool),
            FilterOption(name="exact", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def acopy(
    graph: Stream,
):
    """Copy the input audio unchanged to the output."""
    filter = Filter(command="acopy", params=[])
    graph.append(filter)
    return graph


def concat(graph: Stream, n: int, v: int, a: int, unsafe: bool):
    """Concatenate audio and video streams.
    :param int n: specify the number of segments
    :param int v: specify the number of video streams
    :param int a: specify the number of audio streams
    :param bool unsafe: enable unsafe mode"""
    filter = Filter(
        command="concat",
        params=[
            FilterOption(name="n", type=int),
            FilterOption(name="v", type=int),
            FilterOption(name="a", type=int),
            FilterOption(name="unsafe", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def readeia608(graph: Stream, scan_min: int, scan_max: int, spw: float, chp: bool, lp: bool):
    """Read EIA-608 Closed Caption codes from input video and write them to frame metadata.
    :param int scan_min: set from which line to scan for codes
    :param int scan_max: set to which line to scan for codes
    :param float spw: set ratio of width reserved for sync code detection
    :param bool chp: check and apply parity bit
    :param bool lp: lowpass line prior to processing"""
    filter = Filter(
        command="readeia608",
        params=[
            FilterOption(name="scan_min", type=int),
            FilterOption(name="scan_max", type=int),
            FilterOption(name="spw", type=float),
            FilterOption(name="chp", type=bool),
            FilterOption(name="lp", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def zoompan(graph: Stream, zoom: str, x: str, y: str, d: str, s: int, fps: str):
    """Apply Zoom & Pan effect.
    :param str zoom: set the zoom expression
    :param str x: set the x expression
    :param str y: set the y expression
    :param str d: set the duration expression
    :param int s: set the output image size
    :param str fps: set the output framerate"""
    filter = Filter(
        command="zoompan",
        params=[
            FilterOption(name="zoom", type=str),
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="d", type=str),
            FilterOption(name="s", type=int),
            FilterOption(name="fps", type=str),
        ],
    )
    graph.append(filter)
    return graph


def colorbalance(
    graph: Stream, rs: float, gs: float, bs: float, rm: float, gm: float, bm: float, rh: float, gh: float, bh: float, pl: bool
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
    :param bool pl: preserve lightness"""
    filter = Filter(
        command="colorbalance",
        params=[
            FilterOption(name="rs", type=float),
            FilterOption(name="gs", type=float),
            FilterOption(name="bs", type=float),
            FilterOption(name="rm", type=float),
            FilterOption(name="gm", type=float),
            FilterOption(name="bm", type=float),
            FilterOption(name="rh", type=float),
            FilterOption(name="gh", type=float),
            FilterOption(name="bh", type=float),
            FilterOption(name="pl", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def premultiply(
    graph: Stream,
):
    """PreMultiply first stream with first plane of second stream."""
    filter = Filter(command="premultiply", params=[])
    graph.append(filter)
    return graph


def unpremultiply(
    graph: Stream,
):
    """UnPreMultiply first stream with first plane of second stream."""
    filter = Filter(command="unpremultiply", params=[])
    graph.append(filter)
    return graph


def channelsplit(graph: Stream, channel_layout: str, channels: str):
    """Split audio into per-channel streams.
    :param str channel_layout: Input channel layout.
    :param str channels: Channels to extract."""
    filter = Filter(command="channelsplit", params=[FilterOption(name="channel_layout", type=str), FilterOption(name="channels", type=str)])
    graph.append(filter)
    return graph


def graphmonitor(graph: Stream, size: int, opacity: float, mode: str, flags: str, rate: str):
    """Show various filtergraph stats.
    :param int size: set monitor size
    :param float opacity: set video opacity
    :param str mode: set mode
    :param str flags: set flags
    :param str rate: set video rate"""
    filter = Filter(
        command="graphmonitor",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="opacity", type=float),
            FilterOption(name="mode", type=str),
            FilterOption(name="flags", type=str),
            FilterOption(name="rate", type=str),
        ],
    )
    graph.append(filter)
    return graph


def agraphmonitor(
    graph: Stream,
):
    """Show various filtergraph stats."""
    filter = Filter(command="agraphmonitor", params=[])
    graph.append(filter)
    return graph


def untile(graph: Stream, layout: int):
    """Untile a frame into a sequence of frames.
    :param int layout: set grid size"""
    filter = Filter(command="untile", params=[FilterOption(name="layout", type=int)])
    graph.append(filter)
    return graph


def vmafmotion(graph: Stream, stats_file: str):
    """Calculate the VMAF Motion score.
    :param str stats_file: Set file where to store per-frame difference information"""
    filter = Filter(command="vmafmotion", params=[FilterOption(name="stats_file", type=str)])
    graph.append(filter)
    return graph


def blockdetect(graph: Stream, period_min: int, period_max: int, planes: int):
    """Blockdetect filter.
    :param int period_min: Minimum period to search for
    :param int period_max: Maximum period to search for
    :param int planes: set planes to filter"""
    filter = Filter(
        command="blockdetect",
        params=[
            FilterOption(name="period_min", type=int),
            FilterOption(name="period_max", type=int),
            FilterOption(name="planes", type=int),
        ],
    )
    graph.append(filter)
    return graph


def tmidequalizer(graph: Stream, radius: int, sigma: float, planes: int):
    """Apply Temporal Midway Equalization.
    :param int radius: set radius
    :param float sigma: set sigma
    :param int planes: set planes"""
    filter = Filter(
        command="tmidequalizer",
        params=[FilterOption(name="radius", type=int), FilterOption(name="sigma", type=float), FilterOption(name="planes", type=int)],
    )
    graph.append(filter)
    return graph


def overlay_vulkan(graph: Stream, x: int, y: int):
    """Overlay a source on top of another
    :param int x: Set horizontal offset
    :param int y: Set vertical offset"""
    filter = Filter(command="overlay_vulkan", params=[FilterOption(name="x", type=int), FilterOption(name="y", type=int)])
    graph.append(filter)
    return graph


def despill(graph: Stream, type: int, mix: float, expand: float, red: float, green: float, blue: float, brightness: float, alpha: bool):
    """Despill video.
    :param int type: set the screen type
    :param float mix: set the spillmap mix
    :param float expand: set the spillmap expand
    :param float red: set red scale
    :param float green: set green scale
    :param float blue: set blue scale
    :param float brightness: set brightness
    :param bool alpha: change alpha component"""
    filter = Filter(
        command="despill",
        params=[
            FilterOption(name="type", type=int),
            FilterOption(name="mix", type=float),
            FilterOption(name="expand", type=float),
            FilterOption(name="red", type=float),
            FilterOption(name="green", type=float),
            FilterOption(name="blue", type=float),
            FilterOption(name="brightness", type=float),
            FilterOption(name="alpha", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def volume(graph: Stream, volume: str, precision: int, eval: int, replaygain: int, replaygain_preamp: float, replaygain_noclip: bool):
    """Change input volume.
    :param str volume: set volume adjustment expression
    :param int precision: select mathematical precision
    :param int eval: specify when to evaluate expressions
    :param int replaygain: Apply replaygain side data when present
    :param float replaygain_preamp: Apply replaygain pre-amplification
    :param bool replaygain_noclip: Apply replaygain clipping prevention"""
    filter = Filter(
        command="volume",
        params=[
            FilterOption(name="volume", type=str),
            FilterOption(name="precision", type=int),
            FilterOption(name="eval", type=int),
            FilterOption(name="replaygain", type=int),
            FilterOption(name="replaygain_preamp", type=float),
            FilterOption(name="replaygain_noclip", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def unsharp(
    graph: Stream,
    luma_msize_x: int,
    luma_msize_y: int,
    luma_amount: float,
    chroma_msize_x: int,
    chroma_msize_y: int,
    chroma_amount: float,
    alpha_msize_x: int,
    alpha_msize_y: int,
    alpha_amount: float,
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
    :param float alpha_amount: set alpha effect strength"""
    filter = Filter(
        command="unsharp",
        params=[
            FilterOption(name="luma_msize_x", type=int),
            FilterOption(name="luma_msize_y", type=int),
            FilterOption(name="luma_amount", type=float),
            FilterOption(name="chroma_msize_x", type=int),
            FilterOption(name="chroma_msize_y", type=int),
            FilterOption(name="chroma_amount", type=float),
            FilterOption(name="alpha_msize_x", type=int),
            FilterOption(name="alpha_msize_y", type=int),
            FilterOption(name="alpha_amount", type=float),
        ],
    )
    graph.append(filter)
    return graph


def swapuv(
    graph: Stream,
):
    """Swap U and V components."""
    filter = Filter(command="swapuv", params=[])
    graph.append(filter)
    return graph


def avgblur_vulkan(graph: Stream, sizeX: int, sizeY: int, planes: int):
    """Apply avgblur mask to input video
    :param int sizeX: Set horizontal radius
    :param int sizeY: Set vertical radius
    :param int planes: Set planes to filter (bitmask)"""
    filter = Filter(
        command="avgblur_vulkan",
        params=[FilterOption(name="sizeX", type=int), FilterOption(name="sizeY", type=int), FilterOption(name="planes", type=int)],
    )
    graph.append(filter)
    return graph


def pad_opencl(graph: Stream, width: str, height: str, x: str, y: str, color: str, aspect: int):
    """Pad the input video.
    :param str width: set the pad area width
    :param str height: set the pad area height
    :param str x: set the x offset for the input image position
    :param str y: set the y offset for the input image position
    :param str color: set the color of the padded area border
    :param int aspect: pad to fit an aspect instead of a resolution"""
    filter = Filter(
        command="pad_opencl",
        params=[
            FilterOption(name="width", type=str),
            FilterOption(name="height", type=str),
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="color", type=str),
            FilterOption(name="aspect", type=int),
        ],
    )
    graph.append(filter)
    return graph


def limiter(graph: Stream, min: int, max: int, planes: int):
    """Limit pixels components to the specified range.
    :param int min: set min value
    :param int max: set max value
    :param int planes: set planes"""
    filter = Filter(
        command="limiter",
        params=[FilterOption(name="min", type=int), FilterOption(name="max", type=int), FilterOption(name="planes", type=int)],
    )
    graph.append(filter)
    return graph


def apulsator(
    graph: Stream,
    level_in: float,
    level_out: float,
    mode: int,
    amount: float,
    offset_l: float,
    offset_r: float,
    width: float,
    timing: int,
    bpm: float,
    ms: int,
    hz: float,
):
    """Audio pulsator.
    :param float level_in: set input gain
    :param float level_out: set output gain
    :param int mode: set mode
    :param float amount: set modulation
    :param float offset_l: set offset L
    :param float offset_r: set offset R
    :param float width: set pulse width
    :param int timing: set timing
    :param float bpm: set BPM
    :param int ms: set ms
    :param float hz: set frequency"""
    filter = Filter(
        command="apulsator",
        params=[
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="mode", type=int),
            FilterOption(name="amount", type=float),
            FilterOption(name="offset_l", type=float),
            FilterOption(name="offset_r", type=float),
            FilterOption(name="width", type=float),
            FilterOption(name="timing", type=int),
            FilterOption(name="bpm", type=float),
            FilterOption(name="ms", type=int),
            FilterOption(name="hz", type=float),
        ],
    )
    graph.append(filter)
    return graph


def il(graph: Stream, luma_mode: int, chroma_mode: int, alpha_mode: int, luma_swap: bool, chroma_swap: bool, alpha_swap: bool):
    """Deinterleave or interleave fields.
    :param int luma_mode: select luma mode
    :param int chroma_mode: select chroma mode
    :param int alpha_mode: select alpha mode
    :param bool luma_swap: swap luma fields
    :param bool chroma_swap: swap chroma fields
    :param bool alpha_swap: swap alpha fields"""
    filter = Filter(
        command="il",
        params=[
            FilterOption(name="luma_mode", type=int),
            FilterOption(name="chroma_mode", type=int),
            FilterOption(name="alpha_mode", type=int),
            FilterOption(name="luma_swap", type=bool),
            FilterOption(name="chroma_swap", type=bool),
            FilterOption(name="alpha_swap", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def morpho(graph: Stream, mode: int, planes: int, structure: int):
    """Apply Morphological filter.
    :param int mode: set morphological transform
    :param int planes: set planes to filter
    :param int structure: when to process structures"""
    filter = Filter(
        command="morpho",
        params=[FilterOption(name="mode", type=int), FilterOption(name="planes", type=int), FilterOption(name="structure", type=int)],
    )
    graph.append(filter)
    return graph


def lut(
    graph: Stream,
):
    """Compute and apply a lookup table to the RGB/YUV input video."""
    filter = Filter(command="lut", params=[])
    graph.append(filter)
    return graph


def lutyuv(
    graph: Stream,
):
    """Compute and apply a lookup table to the YUV input video."""
    filter = Filter(command="lutyuv", params=[])
    graph.append(filter)
    return graph


def lutrgb(
    graph: Stream,
):
    """Compute and apply a lookup table to the RGB input video."""
    filter = Filter(command="lutrgb", params=[])
    graph.append(filter)
    return graph


def negate(graph: Stream, components: str, negate_alpha: bool):
    """Negate input video.
    :param str components: set components to negate"""
    filter = Filter(command="negate", params=[FilterOption(name="components", type=str), FilterOption(name="negate_alpha", type=bool)])
    graph.append(filter)
    return graph


def hwupload(graph: Stream, derive_device: str):
    """Upload a normal frame to a hardware frame
    :param str derive_device: Derive a new device of this type"""
    filter = Filter(command="hwupload", params=[FilterOption(name="derive_device", type=str)])
    graph.append(filter)
    return graph


def dejudder(graph: Stream, cycle: int):
    """Remove judder produced by pullup.
    :param int cycle: set the length of the cycle to use for dejuddering"""
    filter = Filter(command="dejudder", params=[FilterOption(name="cycle", type=int)])
    graph.append(filter)
    return graph


def fsync(graph: Stream, file: str):
    """Synchronize video frames from external source.
    :param str file: set the file name to use for frame sync"""
    filter = Filter(command="fsync", params=[FilterOption(name="file", type=str)])
    graph.append(filter)
    return graph


def tonemap(graph: Stream, tonemap: int, param: float, desat: float, peak: float):
    """Conversion to/from different dynamic ranges.
    :param int tonemap: tonemap algorithm selection
    :param float param: tonemap parameter
    :param float desat: desaturation strength
    :param float peak: signal peak override"""
    filter = Filter(
        command="tonemap",
        params=[
            FilterOption(name="tonemap", type=int),
            FilterOption(name="param", type=float),
            FilterOption(name="desat", type=float),
            FilterOption(name="peak", type=float),
        ],
    )
    graph.append(filter)
    return graph


def superequalizer(
    graph: Stream,
    _1b: float,
    _2b: float,
    _3b: float,
    _4b: float,
    _5b: float,
    _6b: float,
    _7b: float,
    _8b: float,
    _9b: float,
    _10b: float,
    _11b: float,
    _12b: float,
    _13b: float,
    _14b: float,
    _15b: float,
    _16b: float,
    _17b: float,
    _18b: float,
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
    :param float 18b: set 20000Hz band gain"""
    filter = Filter(
        command="superequalizer",
        params=[
            FilterOption(name="1b", type=float),
            FilterOption(name="2b", type=float),
            FilterOption(name="3b", type=float),
            FilterOption(name="4b", type=float),
            FilterOption(name="5b", type=float),
            FilterOption(name="6b", type=float),
            FilterOption(name="7b", type=float),
            FilterOption(name="8b", type=float),
            FilterOption(name="9b", type=float),
            FilterOption(name="10b", type=float),
            FilterOption(name="11b", type=float),
            FilterOption(name="12b", type=float),
            FilterOption(name="13b", type=float),
            FilterOption(name="14b", type=float),
            FilterOption(name="15b", type=float),
            FilterOption(name="16b", type=float),
            FilterOption(name="17b", type=float),
            FilterOption(name="18b", type=float),
        ],
    )
    graph.append(filter)
    return graph


def asubboost(
    graph: Stream,
    dry: float,
    wet: float,
    boost: float,
    decay: float,
    feedback: float,
    cutoff: float,
    slope: float,
    delay: float,
    channels: str,
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
    :param str channels: set channels to filter"""
    filter = Filter(
        command="asubboost",
        params=[
            FilterOption(name="dry", type=float),
            FilterOption(name="wet", type=float),
            FilterOption(name="boost", type=float),
            FilterOption(name="decay", type=float),
            FilterOption(name="feedback", type=float),
            FilterOption(name="cutoff", type=float),
            FilterOption(name="slope", type=float),
            FilterOption(name="delay", type=float),
            FilterOption(name="channels", type=str),
        ],
    )
    graph.append(filter)
    return graph


def afade(
    graph: Stream, type: int, start_sample: int, nb_samples: int, start_time: int, duration: int, curve: int, silence: float, unity: float
):
    """Fade in/out input audio.
    :param int type: set the fade direction
    :param int start_sample: set number of first sample to start fading
    :param int nb_samples: set number of samples for fade duration
    :param int start_time: set time to start fading
    :param int duration: set fade duration
    :param int curve: set fade curve type
    :param float silence: set the silence gain
    :param float unity: set the unity gain"""
    filter = Filter(
        command="afade",
        params=[
            FilterOption(name="type", type=int),
            FilterOption(name="start_sample", type=int),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="start_time", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="curve", type=int),
            FilterOption(name="silence", type=float),
            FilterOption(name="unity", type=float),
        ],
    )
    graph.append(filter)
    return graph


def acrossfade(graph: Stream, nb_samples: int, duration: int, overlap: bool, curve1: int, curve2: int):
    """Cross fade two input audio streams.
    :param int nb_samples: set number of samples for cross fade duration
    :param int duration: set cross fade duration
    :param bool overlap: overlap 1st stream end with 2nd stream start
    :param int curve1: set fade curve type for 1st stream
    :param int curve2: set fade curve type for 2nd stream"""
    filter = Filter(
        command="acrossfade",
        params=[
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="overlap", type=bool),
            FilterOption(name="curve1", type=int),
            FilterOption(name="curve2", type=int),
        ],
    )
    graph.append(filter)
    return graph


def pp(graph: Stream, subfilters: str):
    """Filter video using libpostproc.
    :param str subfilters: set postprocess subfilters"""
    filter = Filter(command="pp", params=[FilterOption(name="subfilters", type=str)])
    graph.append(filter)
    return graph


def histeq(graph: Stream, strength: float, intensity: float, antibanding: int):
    """Apply global color histogram equalization.
    :param float strength: set the strength
    :param float intensity: set the intensity
    :param int antibanding: set the antibanding level"""
    filter = Filter(
        command="histeq",
        params=[
            FilterOption(name="strength", type=float),
            FilterOption(name="intensity", type=float),
            FilterOption(name="antibanding", type=int),
        ],
    )
    graph.append(filter)
    return graph


def ashowinfo(
    graph: Stream,
):
    """Show textual information for each audio frame."""
    filter = Filter(command="ashowinfo", params=[])
    graph.append(filter)
    return graph


def afftdn(
    graph: Stream,
    noise_reduction: float,
    noise_floor: float,
    noise_type: int,
    band_noise: str,
    residual_floor: float,
    track_noise: bool,
    track_residual: bool,
    output_mode: int,
    adaptivity: float,
    floor_offset: float,
    noise_link: int,
    band_multiplier: float,
    sample_noise: int,
    gain_smooth: int,
):
    """Denoise audio samples using FFT.
    :param float noise_reduction: set the noise reduction
    :param float noise_floor: set the noise floor
    :param int noise_type: set the noise type
    :param str band_noise: set the custom bands noise
    :param float residual_floor: set the residual floor
    :param bool track_noise: track noise
    :param bool track_residual: track residual
    :param int output_mode: set output mode
    :param float adaptivity: set adaptivity factor
    :param float floor_offset: set noise floor offset factor
    :param int noise_link: set the noise floor link
    :param float band_multiplier: set band multiplier
    :param int sample_noise: set sample noise mode
    :param int gain_smooth: set gain smooth radius"""
    filter = Filter(
        command="afftdn",
        params=[
            FilterOption(name="noise_reduction", type=float),
            FilterOption(name="noise_floor", type=float),
            FilterOption(name="noise_type", type=int),
            FilterOption(name="band_noise", type=str),
            FilterOption(name="residual_floor", type=float),
            FilterOption(name="track_noise", type=bool),
            FilterOption(name="track_residual", type=bool),
            FilterOption(name="output_mode", type=int),
            FilterOption(name="adaptivity", type=float),
            FilterOption(name="floor_offset", type=float),
            FilterOption(name="noise_link", type=int),
            FilterOption(name="band_multiplier", type=float),
            FilterOption(name="sample_noise", type=int),
            FilterOption(name="gain_smooth", type=int),
        ],
    )
    graph.append(filter)
    return graph


def signature(
    graph: Stream, detectmode: int, nb_inputs: int, filename: str, format: int, th_d: int, th_dc: int, th_xh: int, th_di: int, th_it: float
):
    """Calculate the MPEG-7 video signature
    :param int detectmode: set the detectmode
    :param int nb_inputs: number of inputs
    :param str filename: filename for output files
    :param int format: set output format
    :param int th_d: threshold to detect one word as similar
    :param int th_dc: threshold to detect all words as similar
    :param int th_xh: threshold to detect frames as similar
    :param int th_di: minimum length of matching sequence in frames
    :param float th_it: threshold for relation of good to all frames"""
    filter = Filter(
        command="signature",
        params=[
            FilterOption(name="detectmode", type=int),
            FilterOption(name="nb_inputs", type=int),
            FilterOption(name="filename", type=str),
            FilterOption(name="format", type=int),
            FilterOption(name="th_d", type=int),
            FilterOption(name="th_dc", type=int),
            FilterOption(name="th_xh", type=int),
            FilterOption(name="th_di", type=int),
            FilterOption(name="th_it", type=float),
        ],
    )
    graph.append(filter)
    return graph


def adynamicequalizer(
    graph: Stream,
    threshold: float,
    dfrequency: float,
    dqfactor: float,
    tfrequency: float,
    tqfactor: float,
    attack: float,
    release: float,
    ratio: float,
    makeup: float,
    range: float,
    mode: int,
    dftype: int,
    tftype: int,
    auto: int,
    precision: int,
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
    :param int mode: set mode
    :param int dftype: set detection filter type
    :param int tftype: set target filter type
    :param int auto: set auto threshold
    :param int precision: set processing precision"""
    filter = Filter(
        command="adynamicequalizer",
        params=[
            FilterOption(name="threshold", type=float),
            FilterOption(name="dfrequency", type=float),
            FilterOption(name="dqfactor", type=float),
            FilterOption(name="tfrequency", type=float),
            FilterOption(name="tqfactor", type=float),
            FilterOption(name="attack", type=float),
            FilterOption(name="release", type=float),
            FilterOption(name="ratio", type=float),
            FilterOption(name="makeup", type=float),
            FilterOption(name="range", type=float),
            FilterOption(name="mode", type=int),
            FilterOption(name="dftype", type=int),
            FilterOption(name="tftype", type=int),
            FilterOption(name="auto", type=int),
            FilterOption(name="precision", type=int),
        ],
    )
    graph.append(filter)
    return graph


def sine(graph: Stream, frequency: float, beep_factor: float, sample_rate: int, duration: int, samples_per_frame: str):
    """Generate sine wave audio signal.
    :param float frequency: set the sine frequency
    :param float beep_factor: set the beep frequency factor
    :param int sample_rate: set the sample rate
    :param int duration: set the audio duration
    :param str samples_per_frame: set the number of samples per frame"""
    filter = Filter(
        command="sine",
        params=[
            FilterOption(name="frequency", type=float),
            FilterOption(name="beep_factor", type=float),
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="samples_per_frame", type=str),
        ],
    )
    graph.append(filter)
    return graph


def anlms(graph: Stream, order: int, mu: float, eps: float, leakage: float, out_mode: int, precision: int):
    """Apply Normalized Least-Mean-Squares algorithm to first audio stream.
    :param int order: set the filter order
    :param float mu: set the filter mu
    :param float eps: set the filter eps
    :param float leakage: set the filter leakage
    :param int out_mode: set output mode
    :param int precision: set processing precision"""
    filter = Filter(
        command="anlms",
        params=[
            FilterOption(name="order", type=int),
            FilterOption(name="mu", type=float),
            FilterOption(name="eps", type=float),
            FilterOption(name="leakage", type=float),
            FilterOption(name="out_mode", type=int),
            FilterOption(name="precision", type=int),
        ],
    )
    graph.append(filter)
    return graph


def anlmf(
    graph: Stream,
):
    """Apply Normalized Least-Mean-Fourth algorithm to first audio stream."""
    filter = Filter(command="anlmf", params=[])
    graph.append(filter)
    return graph


def maskedthreshold(graph: Stream, threshold: int, planes: int, mode: int):
    """Pick pixels comparing absolute difference of two streams with threshold.
    :param int threshold: set threshold
    :param int planes: set planes
    :param int mode: set mode"""
    filter = Filter(
        command="maskedthreshold",
        params=[FilterOption(name="threshold", type=int), FilterOption(name="planes", type=int), FilterOption(name="mode", type=int)],
    )
    graph.append(filter)
    return graph


def alimiter(
    graph: Stream,
    level_in: float,
    level_out: float,
    limit: float,
    attack: float,
    release: float,
    asc: bool,
    asc_level: float,
    level: bool,
    latency: bool,
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
    :param bool latency: compensate delay"""
    filter = Filter(
        command="alimiter",
        params=[
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="limit", type=float),
            FilterOption(name="attack", type=float),
            FilterOption(name="release", type=float),
            FilterOption(name="asc", type=bool),
            FilterOption(name="asc_level", type=float),
            FilterOption(name="level", type=bool),
            FilterOption(name="latency", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def asidedata(graph: Stream, mode: int, type: int):
    """Manipulate audio frame side data.
    :param int mode: set a mode of operation
    :param int type: set side data type"""
    filter = Filter(command="asidedata", params=[FilterOption(name="mode", type=int), FilterOption(name="type", type=int)])
    graph.append(filter)
    return graph


def sidedata(graph: Stream, mode: int, type: int):
    """Manipulate video frame side data.
    :param int mode: set a mode of operation
    :param int type: set side data type"""
    filter = Filter(command="sidedata", params=[FilterOption(name="mode", type=int), FilterOption(name="type", type=int)])
    graph.append(filter)
    return graph


def siti(graph: Stream, print_summary: bool):
    """Calculate spatial information (SI) and temporal information (TI).
    :param bool print_summary: Print summary showing average values"""
    filter = Filter(command="siti", params=[FilterOption(name="print_summary", type=bool)])
    graph.append(filter)
    return graph


def sab(
    graph: Stream,
    luma_radius: float,
    lr: float,
    luma_pre_filter_radius: float,
    lpfr: float,
    luma_strength: float,
    ls: float,
    chroma_radius: float,
    cr: float,
    chroma_pre_filter_radius: float,
    cpfr: float,
    chroma_strength: float,
    cs: float,
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
    :param float cs: set chroma strength"""
    filter = Filter(
        command="sab",
        params=[
            FilterOption(name="luma_radius", type=float),
            FilterOption(name="lr", type=float),
            FilterOption(name="luma_pre_filter_radius", type=float),
            FilterOption(name="lpfr", type=float),
            FilterOption(name="luma_strength", type=float),
            FilterOption(name="ls", type=float),
            FilterOption(name="chroma_radius", type=float),
            FilterOption(name="cr", type=float),
            FilterOption(name="chroma_pre_filter_radius", type=float),
            FilterOption(name="cpfr", type=float),
            FilterOption(name="chroma_strength", type=float),
            FilterOption(name="cs", type=float),
        ],
    )
    graph.append(filter)
    return graph


def agate(
    graph: Stream,
):
    """Audio gate."""
    filter = Filter(command="agate", params=[])
    graph.append(filter)
    return graph


def sidechaingate(
    graph: Stream,
):
    """Audio sidechain gate."""
    filter = Filter(command="sidechaingate", params=[])
    graph.append(filter)
    return graph


def lut2(
    graph: Stream,
):
    """Compute and apply a lookup table from two video inputs."""
    filter = Filter(command="lut2", params=[])
    graph.append(filter)
    return graph


def tlut2(graph: Stream, c0: str, c1: str, c2: str, c3: str):
    """Compute and apply a lookup table from two successive frames.
    :param str c0: set component #0 expression
    :param str c1: set component #1 expression
    :param str c2: set component #2 expression
    :param str c3: set component #3 expression"""
    filter = Filter(
        command="tlut2",
        params=[
            FilterOption(name="c0", type=str),
            FilterOption(name="c1", type=str),
            FilterOption(name="c2", type=str),
            FilterOption(name="c3", type=str),
        ],
    )
    graph.append(filter)
    return graph


def deband(graph: Stream, _1thr: float, _2thr: float, _3thr: float, _4thr: float, range: int, direction: float, blur: bool, coupling: bool):
    """Debands video.
    :param float 1thr: set 1st plane threshold
    :param float 2thr: set 2nd plane threshold
    :param float 3thr: set 3rd plane threshold
    :param float 4thr: set 4th plane threshold
    :param int range: set range
    :param float direction: set direction
    :param bool blur: set blur
    :param bool coupling: set plane coupling"""
    filter = Filter(
        command="deband",
        params=[
            FilterOption(name="1thr", type=float),
            FilterOption(name="2thr", type=float),
            FilterOption(name="3thr", type=float),
            FilterOption(name="4thr", type=float),
            FilterOption(name="range", type=int),
            FilterOption(name="direction", type=float),
            FilterOption(name="blur", type=bool),
            FilterOption(name="coupling", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def median(graph: Stream, radius: int, planes: int, radiusV: int, percentile: float):
    """Apply Median filter.
    :param int radius: set median radius
    :param int planes: set planes to filter
    :param int radiusV: set median vertical radius
    :param float percentile: set median percentile"""
    filter = Filter(
        command="median",
        params=[
            FilterOption(name="radius", type=int),
            FilterOption(name="planes", type=int),
            FilterOption(name="radiusV", type=int),
            FilterOption(name="percentile", type=float),
        ],
    )
    graph.append(filter)
    return graph


def paletteuse(graph: Stream, dither: int, bayer_scale: int, diff_mode: int, new: bool, alpha_threshold: int, debug_kdtree: str):
    """Use a palette to downsample an input video stream.
    :param int dither: select dithering mode
    :param int bayer_scale: set scale for bayer dithering
    :param int diff_mode: set frame difference mode
    :param bool new: take new palette for each output frame
    :param int alpha_threshold: set the alpha threshold for transparency
    :param str debug_kdtree: save Graphviz graph of the kdtree in specified file"""
    filter = Filter(
        command="paletteuse",
        params=[
            FilterOption(name="dither", type=int),
            FilterOption(name="bayer_scale", type=int),
            FilterOption(name="diff_mode", type=int),
            FilterOption(name="new", type=bool),
            FilterOption(name="alpha_threshold", type=int),
            FilterOption(name="debug_kdtree", type=str),
        ],
    )
    graph.append(filter)
    return graph


def amplify(graph: Stream, radius: int, factor: float, threshold: float, tolerance: float, low: float, high: float, planes: str):
    """Amplify changes between successive video frames.
    :param int radius: set radius
    :param float factor: set factor
    :param float threshold: set threshold
    :param float tolerance: set tolerance
    :param float low: set low limit for amplification
    :param float high: set high limit for amplification
    :param str planes: set what planes to filter"""
    filter = Filter(
        command="amplify",
        params=[
            FilterOption(name="radius", type=int),
            FilterOption(name="factor", type=float),
            FilterOption(name="threshold", type=float),
            FilterOption(name="tolerance", type=float),
            FilterOption(name="low", type=float),
            FilterOption(name="high", type=float),
            FilterOption(name="planes", type=str),
        ],
    )
    graph.append(filter)
    return graph


def asupercut(graph: Stream, cutoff: float, order: int, level: float):
    """Cut super frequencies.
    :param float cutoff: set cutoff frequency
    :param int order: set filter order
    :param float level: set input level"""
    filter = Filter(
        command="asupercut",
        params=[FilterOption(name="cutoff", type=float), FilterOption(name="order", type=int), FilterOption(name="level", type=float)],
    )
    graph.append(filter)
    return graph


def asubcut(graph: Stream, cutoff: float, order: int, level: float):
    """Cut subwoofer frequencies.
    :param float cutoff: set cutoff frequency
    :param int order: set filter order
    :param float level: set input level"""
    filter = Filter(
        command="asubcut",
        params=[FilterOption(name="cutoff", type=float), FilterOption(name="order", type=int), FilterOption(name="level", type=float)],
    )
    graph.append(filter)
    return graph


def asuperpass(
    graph: Stream,
):
    """Apply high order Butterworth band-pass filter."""
    filter = Filter(command="asuperpass", params=[])
    graph.append(filter)
    return graph


def asuperstop(
    graph: Stream,
):
    """Apply high order Butterworth band-stop filter."""
    filter = Filter(command="asuperstop", params=[])
    graph.append(filter)
    return graph


def dctdnoiz(graph: Stream, sigma: float, overlap: int, expr: str, n: int):
    """Denoise frames using 2D DCT.
    :param float sigma: set noise sigma constant
    :param int overlap: set number of block overlapping pixels
    :param str expr: set coefficient factor expression
    :param int n: set the block size, expressed in bits"""
    filter = Filter(
        command="dctdnoiz",
        params=[
            FilterOption(name="sigma", type=float),
            FilterOption(name="overlap", type=int),
            FilterOption(name="expr", type=str),
            FilterOption(name="n", type=int),
        ],
    )
    graph.append(filter)
    return graph


def ebur128(
    graph: Stream,
    video: bool,
    size: int,
    meter: int,
    framelog: int,
    metadata: bool,
    peak: str,
    dualmono: bool,
    panlaw: float,
    target: int,
    gauge: int,
    scale: int,
    integrated: float,
    range: float,
    lra_low: float,
    lra_high: float,
    sample_peak: float,
    true_peak: float,
):
    """EBU R128 scanner.
    :param bool video: set video output
    :param int size: set video size
    :param int meter: set scale meter (+9 to +18)
    :param int framelog: force frame logging level
    :param bool metadata: inject metadata in the filtergraph
    :param str peak: set peak mode
    :param bool dualmono: treat mono input files as dual-mono
    :param float panlaw: set a specific pan law for dual-mono files
    :param int target: set a specific target level in LUFS (-23 to 0)
    :param int gauge: set gauge display type
    :param int scale: sets display method for the stats
    :param float integrated: integrated loudness (LUFS)
    :param float range: loudness range (LU)
    :param float lra_low: LRA low (LUFS)
    :param float lra_high: LRA high (LUFS)
    :param float sample_peak: sample peak (dBFS)
    :param float true_peak: true peak (dBFS)"""
    filter = Filter(
        command="ebur128",
        params=[
            FilterOption(name="video", type=bool),
            FilterOption(name="size", type=int),
            FilterOption(name="meter", type=int),
            FilterOption(name="framelog", type=int),
            FilterOption(name="metadata", type=bool),
            FilterOption(name="peak", type=str),
            FilterOption(name="dualmono", type=bool),
            FilterOption(name="panlaw", type=float),
            FilterOption(name="target", type=int),
            FilterOption(name="gauge", type=int),
            FilterOption(name="scale", type=int),
            FilterOption(name="integrated", type=float),
            FilterOption(name="range", type=float),
            FilterOption(name="lra_low", type=float),
            FilterOption(name="lra_high", type=float),
            FilterOption(name="sample_peak", type=float),
            FilterOption(name="true_peak", type=float),
        ],
    )
    graph.append(filter)
    return graph


def chromanr(
    graph: Stream, thres: float, sizew: int, sizeh: int, stepw: int, steph: int, threy: float, threu: float, threv: float, distance: int
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
    :param int distance: set distance type"""
    filter = Filter(
        command="chromanr",
        params=[
            FilterOption(name="thres", type=float),
            FilterOption(name="sizew", type=int),
            FilterOption(name="sizeh", type=int),
            FilterOption(name="stepw", type=int),
            FilterOption(name="steph", type=int),
            FilterOption(name="threy", type=float),
            FilterOption(name="threu", type=float),
            FilterOption(name="threv", type=float),
            FilterOption(name="distance", type=int),
        ],
    )
    graph.append(filter)
    return graph


def colorkey(graph: Stream, color: str, similarity: float, blend: float):
    """Turns a certain color into transparency. Operates on RGB colors.
    :param str color: set the colorkey key color
    :param float similarity: set the colorkey similarity value
    :param float blend: set the colorkey key blend value"""
    filter = Filter(
        command="colorkey",
        params=[FilterOption(name="color", type=str), FilterOption(name="similarity", type=float), FilterOption(name="blend", type=float)],
    )
    graph.append(filter)
    return graph


def colorhold(graph: Stream, color: str, similarity: float, blend: float):
    """Turns a certain color range into gray. Operates on RGB colors.
    :param str color: set the colorhold key color
    :param float similarity: set the colorhold similarity value
    :param float blend: set the colorhold blend value"""
    filter = Filter(
        command="colorhold",
        params=[FilterOption(name="color", type=str), FilterOption(name="similarity", type=float), FilterOption(name="blend", type=float)],
    )
    graph.append(filter)
    return graph


def corr(
    graph: Stream,
):
    """Calculate the correlation between two video streams."""
    filter = Filter(command="corr", params=[])
    graph.append(filter)
    return graph


def aresample(
    graph: Stream,
):
    """Resample audio data."""
    filter = Filter(command="aresample", params=[])
    graph.append(filter)
    return graph


def removelogo(graph: Stream, filename: str):
    """Remove a TV logo based on a mask image.
    :param str filename: set bitmap filename"""
    filter = Filter(command="removelogo", params=[FilterOption(name="filename", type=str)])
    graph.append(filter)
    return graph


def showfreqs(
    graph: Stream,
    size: int,
    rate: str,
    mode: int,
    ascale: int,
    fscale: int,
    win_size: int,
    win_func: int,
    overlap: float,
    averaging: int,
    colors: str,
    cmode: int,
    minamp: float,
    data: int,
    channels: str,
):
    """Convert input audio to a frequencies video output.
    :param int size: set video size
    :param str rate: set video rate
    :param int mode: set display mode
    :param int ascale: set amplitude scale
    :param int fscale: set frequency scale
    :param int win_size: set window size
    :param int win_func: set window function
    :param float overlap: set window overlap
    :param int averaging: set time averaging
    :param str colors: set channels colors
    :param int cmode: set channel mode
    :param float minamp: set minimum amplitude
    :param int data: set data mode
    :param str channels: set channels to draw"""
    filter = Filter(
        command="showfreqs",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="mode", type=int),
            FilterOption(name="ascale", type=int),
            FilterOption(name="fscale", type=int),
            FilterOption(name="win_size", type=int),
            FilterOption(name="win_func", type=int),
            FilterOption(name="overlap", type=float),
            FilterOption(name="averaging", type=int),
            FilterOption(name="colors", type=str),
            FilterOption(name="cmode", type=int),
            FilterOption(name="minamp", type=float),
            FilterOption(name="data", type=int),
            FilterOption(name="channels", type=str),
        ],
    )
    graph.append(filter)
    return graph


def trim(graph: Stream, start: int, end: int, start_pts: int, end_pts: int, duration: int, start_frame: int, end_frame: int):
    """Pick one continuous section from the input, drop the rest.
    :param int start: Timestamp of the first frame that should be passed
    :param int end: Timestamp of the first frame that should be dropped again
    :param int start_pts: Timestamp of the first frame that should be  passed
    :param int end_pts: Timestamp of the first frame that should be dropped again
    :param int duration: Maximum duration of the output
    :param int start_frame: Number of the first frame that should be passed to the output
    :param int end_frame: Number of the first frame that should be dropped again"""
    filter = Filter(
        command="trim",
        params=[
            FilterOption(name="start", type=int),
            FilterOption(name="end", type=int),
            FilterOption(name="start_pts", type=int),
            FilterOption(name="end_pts", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="start_frame", type=int),
            FilterOption(name="end_frame", type=int),
        ],
    )
    graph.append(filter)
    return graph


def atrim(graph: Stream, start: int, end: int, start_pts: int, end_pts: int, duration: int, start_sample: int, end_sample: int):
    """Pick one continuous section from the input, drop the rest.
    :param int start: Timestamp of the first frame that should be passed
    :param int end: Timestamp of the first frame that should be dropped again
    :param int start_pts: Timestamp of the first frame that should be  passed
    :param int end_pts: Timestamp of the first frame that should be dropped again
    :param int duration: Maximum duration of the output
    :param int start_sample: Number of the first audio sample that should be passed to the output
    :param int end_sample: Number of the first audio sample that should be dropped again"""
    filter = Filter(
        command="atrim",
        params=[
            FilterOption(name="start", type=int),
            FilterOption(name="end", type=int),
            FilterOption(name="start_pts", type=int),
            FilterOption(name="end_pts", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="start_sample", type=int),
            FilterOption(name="end_sample", type=int),
        ],
    )
    graph.append(filter)
    return graph


def anullsink(
    graph: Stream,
):
    """Do absolutely nothing with the input audio."""
    filter = Filter(command="anullsink", params=[])
    graph.append(filter)
    return graph


def sendcmd(
    graph: Stream,
):
    """Send commands to filters."""
    filter = Filter(command="sendcmd", params=[])
    graph.append(filter)
    return graph


def asendcmd(
    graph: Stream,
):
    """Send commands to filters."""
    filter = Filter(command="asendcmd", params=[])
    graph.append(filter)
    return graph


def scdet(graph: Stream, threshold: float, sc_pass: bool):
    """Detect video scene change
    :param float threshold: set scene change detect threshold
    :param bool sc_pass: Set the flag to pass scene change frames"""
    filter = Filter(command="scdet", params=[FilterOption(name="threshold", type=float), FilterOption(name="sc_pass", type=bool)])
    graph.append(filter)
    return graph


def blend(
    graph: Stream,
    c0_mode: int,
    c1_mode: int,
    c2_mode: int,
    c3_mode: int,
    all_mode: int,
    c0_expr: str,
    c1_expr: str,
    c2_expr: str,
    c3_expr: str,
    all_expr: str,
    c0_opacity: float,
    c1_opacity: float,
    c2_opacity: float,
    c3_opacity: float,
    all_opacity: float,
):
    """Blend two video frames into each other.
    :param int c0_mode: set component #0 blend mode
    :param int c1_mode: set component #1 blend mode
    :param int c2_mode: set component #2 blend mode
    :param int c3_mode: set component #3 blend mode
    :param int all_mode: set blend mode for all components
    :param str c0_expr: set color component #0 expression
    :param str c1_expr: set color component #1 expression
    :param str c2_expr: set color component #2 expression
    :param str c3_expr: set color component #3 expression
    :param str all_expr: set expression for all color components
    :param float c0_opacity: set color component #0 opacity
    :param float c1_opacity: set color component #1 opacity
    :param float c2_opacity: set color component #2 opacity
    :param float c3_opacity: set color component #3 opacity
    :param float all_opacity: set opacity for all color components"""
    filter = Filter(
        command="blend",
        params=[
            FilterOption(name="c0_mode", type=int),
            FilterOption(name="c1_mode", type=int),
            FilterOption(name="c2_mode", type=int),
            FilterOption(name="c3_mode", type=int),
            FilterOption(name="all_mode", type=int),
            FilterOption(name="c0_expr", type=str),
            FilterOption(name="c1_expr", type=str),
            FilterOption(name="c2_expr", type=str),
            FilterOption(name="c3_expr", type=str),
            FilterOption(name="all_expr", type=str),
            FilterOption(name="c0_opacity", type=float),
            FilterOption(name="c1_opacity", type=float),
            FilterOption(name="c2_opacity", type=float),
            FilterOption(name="c3_opacity", type=float),
            FilterOption(name="all_opacity", type=float),
        ],
    )
    graph.append(filter)
    return graph


def tblend(
    graph: Stream,
):
    """Blend successive frames."""
    filter = Filter(command="tblend", params=[])
    graph.append(filter)
    return graph


def ciescope(
    graph: Stream,
    system: int,
    cie: int,
    gamuts: str,
    size: int,
    intensity: float,
    contrast: float,
    corrgamma: bool,
    showwhite: bool,
    gamma: float,
    fill: bool,
):
    """Video CIE scope.
    :param int system: set color system
    :param int cie: set cie system
    :param str gamuts: set what gamuts to draw
    :param int size: set ciescope size
    :param float intensity: set ciescope intensity
    :param bool fill: fill with CIE colors"""
    filter = Filter(
        command="ciescope",
        params=[
            FilterOption(name="system", type=int),
            FilterOption(name="cie", type=int),
            FilterOption(name="gamuts", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="intensity", type=float),
            FilterOption(name="contrast", type=float),
            FilterOption(name="corrgamma", type=bool),
            FilterOption(name="showwhite", type=bool),
            FilterOption(name="gamma", type=float),
            FilterOption(name="fill", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def aphasemeter(
    graph: Stream,
    rate: str,
    size: int,
    rc: int,
    gc: int,
    bc: int,
    mpc: str,
    video: bool,
    phasing: bool,
    tolerance: float,
    angle: float,
    duration: int,
):
    """Convert input audio to phase meter video output.
    :param str rate: set video rate
    :param int size: set video size
    :param int rc: set red contrast
    :param int gc: set green contrast
    :param int bc: set blue contrast
    :param str mpc: set median phase color
    :param bool video: set video output
    :param bool phasing: set mono and out-of-phase detection output
    :param float tolerance: set phase tolerance for mono detection
    :param float angle: set angle threshold for out-of-phase detection
    :param int duration: set minimum mono or out-of-phase duration in seconds"""
    filter = Filter(
        command="aphasemeter",
        params=[
            FilterOption(name="rate", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="rc", type=int),
            FilterOption(name="gc", type=int),
            FilterOption(name="bc", type=int),
            FilterOption(name="mpc", type=str),
            FilterOption(name="video", type=bool),
            FilterOption(name="phasing", type=bool),
            FilterOption(name="tolerance", type=float),
            FilterOption(name="angle", type=float),
            FilterOption(name="duration", type=int),
        ],
    )
    graph.append(filter)
    return graph


def find_rect(graph: Stream, object: str, threshold: float, mipmaps: int, xmin: int, ymin: int, xmax: int, ymax: int, discard: bool):
    """Find a user specified object.
    :param str object: object bitmap filename
    :param float threshold: set threshold
    :param int mipmaps: set mipmaps"""
    filter = Filter(
        command="find_rect",
        params=[
            FilterOption(name="object", type=str),
            FilterOption(name="threshold", type=float),
            FilterOption(name="mipmaps", type=int),
            FilterOption(name="xmin", type=int),
            FilterOption(name="ymin", type=int),
            FilterOption(name="xmax", type=int),
            FilterOption(name="ymax", type=int),
            FilterOption(name="discard", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def fieldorder(graph: Stream, order: int):
    """Set the field order.
    :param int order: output field order"""
    filter = Filter(command="fieldorder", params=[FilterOption(name="order", type=int)])
    graph.append(filter)
    return graph


def mix(graph: Stream, inputs: int, weights: str, scale: float, planes: str, duration: int):
    """Mix video inputs.
    :param int inputs: set number of inputs
    :param str weights: set weight for each input
    :param float scale: set scale
    :param str planes: set what planes to filter
    :param int duration: how to determine end of stream"""
    filter = Filter(
        command="mix",
        params=[
            FilterOption(name="inputs", type=int),
            FilterOption(name="weights", type=str),
            FilterOption(name="scale", type=float),
            FilterOption(name="planes", type=str),
            FilterOption(name="duration", type=int),
        ],
    )
    graph.append(filter)
    return graph


def tmix(graph: Stream, frames: int, weights: str, scale: float, planes: str):
    """Mix successive video frames.
    :param int frames: set number of successive frames to mix
    :param str weights: set weight for each frame
    :param float scale: set scale
    :param str planes: set what planes to filter"""
    filter = Filter(
        command="tmix",
        params=[
            FilterOption(name="frames", type=int),
            FilterOption(name="weights", type=str),
            FilterOption(name="scale", type=float),
            FilterOption(name="planes", type=str),
        ],
    )
    graph.append(filter)
    return graph


def transpose_vulkan(graph: Stream, dir: int, passthrough: int):
    """Transpose Vulkan Filter
    :param int dir: set transpose direction
    :param int passthrough: do not apply transposition if the input matches the specified geometry"""
    filter = Filter(command="transpose_vulkan", params=[FilterOption(name="dir", type=int), FilterOption(name="passthrough", type=int)])
    graph.append(filter)
    return graph


def derain(graph: Stream, filter_type: int, dnn_backend: int, model: str, input: str, output: str):
    """Apply derain filter to the input.
    :param int filter_type: filter type(derain/dehaze)
    :param int dnn_backend: DNN backend
    :param str model: path to model file
    :param str input: input name of the model
    :param str output: output name of the model"""
    filter = Filter(
        command="derain",
        params=[
            FilterOption(name="filter_type", type=int),
            FilterOption(name="dnn_backend", type=int),
            FilterOption(name="model", type=str),
            FilterOption(name="input", type=str),
            FilterOption(name="output", type=str),
        ],
    )
    graph.append(filter)
    return graph


def limitdiff(graph: Stream, threshold: float, elasticity: float, reference: bool, planes: int):
    """Apply filtering with limiting difference.
    :param float threshold: set the threshold
    :param float elasticity: set the elasticity
    :param bool reference: enable reference stream
    :param int planes: set the planes to filter"""
    filter = Filter(
        command="limitdiff",
        params=[
            FilterOption(name="threshold", type=float),
            FilterOption(name="elasticity", type=float),
            FilterOption(name="reference", type=bool),
            FilterOption(name="planes", type=int),
        ],
    )
    graph.append(filter)
    return graph


def identity(
    graph: Stream,
):
    """Calculate the Identity between two video streams."""
    filter = Filter(command="identity", params=[])
    graph.append(filter)
    return graph


def msad(
    graph: Stream,
):
    """Calculate the MSAD between two video streams."""
    filter = Filter(command="msad", params=[])
    graph.append(filter)
    return graph


def erosion(
    graph: Stream,
):
    """Apply erosion effect."""
    filter = Filter(command="erosion", params=[])
    graph.append(filter)
    return graph


def dilation(
    graph: Stream,
):
    """Apply dilation effect."""
    filter = Filter(command="dilation", params=[])
    graph.append(filter)
    return graph


def deflate(
    graph: Stream,
):
    """Apply deflate effect."""
    filter = Filter(command="deflate", params=[])
    graph.append(filter)
    return graph


def inflate(
    graph: Stream,
):
    """Apply inflate effect."""
    filter = Filter(command="inflate", params=[])
    graph.append(filter)
    return graph


def maskedmin(
    graph: Stream,
):
    """Apply filtering with minimum difference of two streams."""
    filter = Filter(command="maskedmin", params=[])
    graph.append(filter)
    return graph


def maskedmax(
    graph: Stream,
):
    """Apply filtering with maximum difference of two streams."""
    filter = Filter(command="maskedmax", params=[])
    graph.append(filter)
    return graph


def histogram(
    graph: Stream,
    level_height: int,
    scale_height: int,
    display_mode: int,
    levels_mode: int,
    components: int,
    fgopacity: float,
    bgopacity: float,
    colors_mode: int,
):
    """Compute and draw a histogram.
    :param int level_height: set level height
    :param int scale_height: set scale height
    :param int display_mode: set display mode
    :param int levels_mode: set levels mode
    :param int components: set color components to display
    :param float fgopacity: set foreground opacity
    :param float bgopacity: set background opacity
    :param int colors_mode: set colors mode"""
    filter = Filter(
        command="histogram",
        params=[
            FilterOption(name="level_height", type=int),
            FilterOption(name="scale_height", type=int),
            FilterOption(name="display_mode", type=int),
            FilterOption(name="levels_mode", type=int),
            FilterOption(name="components", type=int),
            FilterOption(name="fgopacity", type=float),
            FilterOption(name="bgopacity", type=float),
            FilterOption(name="colors_mode", type=int),
        ],
    )
    graph.append(filter)
    return graph


def thistogram(
    graph: Stream,
    width: int,
    display_mode: int,
    levels_mode: int,
    components: int,
    bgopacity: float,
    envelope: bool,
    ecolor: str,
    slide: int,
):
    """Compute and draw a temporal histogram.
    :param int width: set width
    :param int display_mode: set display mode
    :param int levels_mode: set levels mode
    :param int components: set color components to display
    :param float bgopacity: set background opacity
    :param bool envelope: display envelope
    :param str ecolor: set envelope color
    :param int slide: set slide mode"""
    filter = Filter(
        command="thistogram",
        params=[
            FilterOption(name="width", type=int),
            FilterOption(name="display_mode", type=int),
            FilterOption(name="levels_mode", type=int),
            FilterOption(name="components", type=int),
            FilterOption(name="bgopacity", type=float),
            FilterOption(name="envelope", type=bool),
            FilterOption(name="ecolor", type=str),
            FilterOption(name="slide", type=int),
        ],
    )
    graph.append(filter)
    return graph


def extractplanes(graph: Stream, planes: str):
    """Extract planes as grayscale frames.
    :param str planes: set planes"""
    filter = Filter(command="extractplanes", params=[FilterOption(name="planes", type=str)])
    graph.append(filter)
    return graph


def alphaextract(
    graph: Stream,
):
    """Extract an alpha channel as a grayscale image component."""
    filter = Filter(command="alphaextract", params=[])
    graph.append(filter)
    return graph


def hqx(graph: Stream, n: int):
    """Scale the input by 2, 3 or 4 using the hq*x magnification algorithm.
    :param int n: set scale factor"""
    filter = Filter(command="hqx", params=[FilterOption(name="n", type=int)])
    graph.append(filter)
    return graph


def datascope(graph: Stream, size: int, x: int, y: int, mode: int, axis: bool, opacity: float, format: int, components: int):
    """Video data analysis.
    :param int size: set output size
    :param int x: set x offset
    :param int y: set y offset
    :param int mode: set scope mode
    :param bool axis: draw column/row numbers
    :param float opacity: set background opacity
    :param int format: set display number format
    :param int components: set components to display"""
    filter = Filter(
        command="datascope",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="x", type=int),
            FilterOption(name="y", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="axis", type=bool),
            FilterOption(name="opacity", type=float),
            FilterOption(name="format", type=int),
            FilterOption(name="components", type=int),
        ],
    )
    graph.append(filter)
    return graph


def pixscope(graph: Stream, x: float, y: float, w: int, h: int, o: float, wx: float, wy: float):
    """Pixel data analysis.
    :param float x: set scope x offset
    :param float y: set scope y offset
    :param int w: set scope width
    :param int h: set scope height
    :param float o: set window opacity
    :param float wx: set window x offset
    :param float wy: set window y offset"""
    filter = Filter(
        command="pixscope",
        params=[
            FilterOption(name="x", type=float),
            FilterOption(name="y", type=float),
            FilterOption(name="w", type=int),
            FilterOption(name="h", type=int),
            FilterOption(name="o", type=float),
            FilterOption(name="wx", type=float),
            FilterOption(name="wy", type=float),
        ],
    )
    graph.append(filter)
    return graph


def oscilloscope(
    graph: Stream,
    x: float,
    y: float,
    s: float,
    t: float,
    o: float,
    tx: float,
    ty: float,
    tw: float,
    th: float,
    c: int,
    g: bool,
    st: bool,
    sc: bool,
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
    :param bool sc: draw scope"""
    filter = Filter(
        command="oscilloscope",
        params=[
            FilterOption(name="x", type=float),
            FilterOption(name="y", type=float),
            FilterOption(name="s", type=float),
            FilterOption(name="t", type=float),
            FilterOption(name="o", type=float),
            FilterOption(name="tx", type=float),
            FilterOption(name="ty", type=float),
            FilterOption(name="tw", type=float),
            FilterOption(name="th", type=float),
            FilterOption(name="c", type=int),
            FilterOption(name="g", type=bool),
            FilterOption(name="st", type=bool),
            FilterOption(name="sc", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def guided(graph: Stream, radius: int, eps: float, mode: int, sub: int, guidance: int, planes: int):
    """Apply Guided filter.
    :param int radius: set the box radius
    :param float eps: set the regularization parameter (with square)
    :param int mode: set filtering mode (0: basic mode; 1: fast mode)
    :param int sub: subsampling ratio for fast mode
    :param int guidance: set guidance mode (0: off mode; 1: on mode)
    :param int planes: set planes to filter"""
    filter = Filter(
        command="guided",
        params=[
            FilterOption(name="radius", type=int),
            FilterOption(name="eps", type=float),
            FilterOption(name="mode", type=int),
            FilterOption(name="sub", type=int),
            FilterOption(name="guidance", type=int),
            FilterOption(name="planes", type=int),
        ],
    )
    graph.append(filter)
    return graph


def ahistogram(
    graph: Stream, dmode: int, rate: str, size: int, scale: int, ascale: int, acount: int, rheight: float, slide: int, hmode: int
):
    """Convert input audio to histogram video output.
    :param int dmode: set method to display channels
    :param str rate: set video rate
    :param int size: set video size
    :param int scale: set display scale
    :param int ascale: set amplitude scale
    :param int acount: how much frames to accumulate
    :param float rheight: set histogram ratio of window height
    :param int slide: set sonogram sliding
    :param int hmode: set histograms mode"""
    filter = Filter(
        command="ahistogram",
        params=[
            FilterOption(name="dmode", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="scale", type=int),
            FilterOption(name="ascale", type=int),
            FilterOption(name="acount", type=int),
            FilterOption(name="rheight", type=float),
            FilterOption(name="slide", type=int),
            FilterOption(name="hmode", type=int),
        ],
    )
    graph.append(filter)
    return graph


def qp(graph: Stream, qp: str):
    """Change video quantization parameters.
    :param str qp: set qp expression"""
    filter = Filter(command="qp", params=[FilterOption(name="qp", type=str)])
    graph.append(filter)
    return graph


def geq(
    graph: Stream,
    lum_expr: str,
    lum: str,
    cb_expr: str,
    cb: str,
    cr_expr: str,
    cr: str,
    alpha_expr: str,
    a: str,
    red_expr: str,
    r: str,
    green_expr: str,
    g: str,
    blue_expr: str,
    b: str,
    interpolation: int,
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
    :param int interpolation: set interpolation method"""
    filter = Filter(
        command="geq",
        params=[
            FilterOption(name="lum_expr", type=str),
            FilterOption(name="lum", type=str),
            FilterOption(name="cb_expr", type=str),
            FilterOption(name="cb", type=str),
            FilterOption(name="cr_expr", type=str),
            FilterOption(name="cr", type=str),
            FilterOption(name="alpha_expr", type=str),
            FilterOption(name="a", type=str),
            FilterOption(name="red_expr", type=str),
            FilterOption(name="r", type=str),
            FilterOption(name="green_expr", type=str),
            FilterOption(name="g", type=str),
            FilterOption(name="blue_expr", type=str),
            FilterOption(name="b", type=str),
            FilterOption(name="interpolation", type=int),
        ],
    )
    graph.append(filter)
    return graph


def curves(
    graph: Stream,
    preset: int,
    master: str,
    m: str,
    red: str,
    r: str,
    green: str,
    g: str,
    blue: str,
    b: str,
    all: str,
    psfile: str,
    plot: str,
    interp: int,
):
    """Adjust components curves.
    :param int preset: select a color curves preset
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
    :param int interp: specify the kind of interpolation"""
    filter = Filter(
        command="curves",
        params=[
            FilterOption(name="preset", type=int),
            FilterOption(name="master", type=str),
            FilterOption(name="m", type=str),
            FilterOption(name="red", type=str),
            FilterOption(name="r", type=str),
            FilterOption(name="green", type=str),
            FilterOption(name="g", type=str),
            FilterOption(name="blue", type=str),
            FilterOption(name="b", type=str),
            FilterOption(name="all", type=str),
            FilterOption(name="psfile", type=str),
            FilterOption(name="plot", type=str),
            FilterOption(name="interp", type=int),
        ],
    )
    graph.append(filter)
    return graph


def mcdeint(graph: Stream, mode: int, parity: int, qp: int):
    """Apply motion compensating deinterlacing.
    :param int mode: set mode
    :param int parity: set the assumed picture field parity
    :param int qp: set qp"""
    filter = Filter(
        command="mcdeint",
        params=[FilterOption(name="mode", type=int), FilterOption(name="parity", type=int), FilterOption(name="qp", type=int)],
    )
    graph.append(filter)
    return graph


def freezedetect(graph: Stream, n: float, d: int):
    """Detects frozen video input.
    :param float n: set noise tolerance
    :param int d: set minimum duration in seconds"""
    filter = Filter(command="freezedetect", params=[FilterOption(name="n", type=float), FilterOption(name="d", type=int)])
    graph.append(filter)
    return graph


def fps(graph: Stream, fps: str, start_time: float, round: int, eof_action: int):
    """Force constant framerate.
    :param str fps: A string describing desired output framerate
    :param float start_time: Assume the first PTS should be this value.
    :param int round: set rounding method for timestamps
    :param int eof_action: action performed for last frame"""
    filter = Filter(
        command="fps",
        params=[
            FilterOption(name="fps", type=str),
            FilterOption(name="start_time", type=float),
            FilterOption(name="round", type=int),
            FilterOption(name="eof_action", type=int),
        ],
    )
    graph.append(filter)
    return graph


def chromaber_vulkan(graph: Stream, dist_x: float, dist_y: float):
    """Offset chroma of input video (chromatic aberration)
    :param float dist_x: Set horizontal distortion amount
    :param float dist_y: Set vertical distortion amount"""
    filter = Filter(command="chromaber_vulkan", params=[FilterOption(name="dist_x", type=float), FilterOption(name="dist_y", type=float)])
    graph.append(filter)
    return graph


def reverse(
    graph: Stream,
):
    """Reverse a clip."""
    filter = Filter(command="reverse", params=[])
    graph.append(filter)
    return graph


def areverse(
    graph: Stream,
):
    """Reverse an audio clip."""
    filter = Filter(command="areverse", params=[])
    graph.append(filter)
    return graph


def tpad(graph: Stream, start: int, stop: int, start_mode: int, stop_mode: int, start_duration: int, stop_duration: int, color: str):
    """Temporarily pad video frames.
    :param int start: set the number of frames to delay input
    :param int stop: set the number of frames to add after input finished
    :param int start_mode: set the mode of added frames to start
    :param int stop_mode: set the mode of added frames to end
    :param int start_duration: set the duration to delay input
    :param int stop_duration: set the duration to pad input
    :param str color: set the color of the added frames"""
    filter = Filter(
        command="tpad",
        params=[
            FilterOption(name="start", type=int),
            FilterOption(name="stop", type=int),
            FilterOption(name="start_mode", type=int),
            FilterOption(name="stop_mode", type=int),
            FilterOption(name="start_duration", type=int),
            FilterOption(name="stop_duration", type=int),
            FilterOption(name="color", type=str),
        ],
    )
    graph.append(filter)
    return graph


def fftfilt(graph: Stream, dc_Y: int, dc_U: int, dc_V: int, weight_Y: str, weight_U: str, weight_V: str, eval: int):
    """Apply arbitrary expressions to pixels in frequency domain.
    :param int dc_Y: adjust gain in Y plane
    :param int dc_U: adjust gain in U plane
    :param int dc_V: adjust gain in V plane
    :param str weight_Y: set luminance expression in Y plane
    :param str weight_U: set chrominance expression in U plane
    :param str weight_V: set chrominance expression in V plane
    :param int eval: specify when to evaluate expressions"""
    filter = Filter(
        command="fftfilt",
        params=[
            FilterOption(name="dc_Y", type=int),
            FilterOption(name="dc_U", type=int),
            FilterOption(name="dc_V", type=int),
            FilterOption(name="weight_Y", type=str),
            FilterOption(name="weight_U", type=str),
            FilterOption(name="weight_V", type=str),
            FilterOption(name="eval", type=int),
        ],
    )
    graph.append(filter)
    return graph


def compensationdelay(graph: Stream, mm: int, cm: int, m: int, dry: float, wet: float, temp: int):
    """Audio Compensation Delay Line.
    :param int mm: set mm distance
    :param int cm: set cm distance
    :param int m: set meter distance
    :param float dry: set dry amount
    :param float wet: set wet amount
    :param int temp: set temperature °C"""
    filter = Filter(
        command="compensationdelay",
        params=[
            FilterOption(name="mm", type=int),
            FilterOption(name="cm", type=int),
            FilterOption(name="m", type=int),
            FilterOption(name="dry", type=float),
            FilterOption(name="wet", type=float),
            FilterOption(name="temp", type=int),
        ],
    )
    graph.append(filter)
    return graph


def arls(graph: Stream, order: int, _lambda: float, delta: float, out_mode: int, precision: int):
    """Apply Recursive Least Squares algorithm to first audio stream.
    :param int order: set the filter order
    :param float lambda: set the filter lambda
    :param float delta: set the filter delta
    :param int out_mode: set output mode
    :param int precision: set processing precision"""
    filter = Filter(
        command="arls",
        params=[
            FilterOption(name="order", type=int),
            FilterOption(name="lambda", type=float),
            FilterOption(name="delta", type=float),
            FilterOption(name="out_mode", type=int),
            FilterOption(name="precision", type=int),
        ],
    )
    graph.append(filter)
    return graph


def entropy(graph: Stream, mode: int):
    """Measure video frames entropy.
    :param int mode: set kind of histogram entropy measurement"""
    filter = Filter(command="entropy", params=[FilterOption(name="mode", type=int)])
    graph.append(filter)
    return graph


def ccrepack(
    graph: Stream,
):
    """Repack CEA-708 closed caption metadata"""
    filter = Filter(command="ccrepack", params=[])
    graph.append(filter)
    return graph


def mpdecimate(graph: Stream, max: int, keep: int, hi: int, lo: int, frac: float):
    """Remove near-duplicate frames.
    :param int max: set the maximum number of consecutive dropped frames (positive), or the minimum interval between dropped frames (negative)
    :param int keep: set the number of similar consecutive frames to be kept before starting to drop similar frames
    :param int hi: set high dropping threshold
    :param int lo: set low dropping threshold
    :param float frac: set fraction dropping threshold"""
    filter = Filter(
        command="mpdecimate",
        params=[
            FilterOption(name="max", type=int),
            FilterOption(name="keep", type=int),
            FilterOption(name="hi", type=int),
            FilterOption(name="lo", type=int),
            FilterOption(name="frac", type=float),
        ],
    )
    graph.append(filter)
    return graph


def avgblur(graph: Stream, sizeX: int, planes: int, sizeY: int):
    """Apply Average Blur filter.
    :param int sizeX: set horizontal size
    :param int planes: set planes to filter
    :param int sizeY: set vertical size"""
    filter = Filter(
        command="avgblur",
        params=[FilterOption(name="sizeX", type=int), FilterOption(name="planes", type=int), FilterOption(name="sizeY", type=int)],
    )
    graph.append(filter)
    return graph


def transpose(graph: Stream, dir: int, passthrough: int):
    """Transpose input video.
    :param int dir: set transpose direction
    :param int passthrough: do not apply transposition if the input matches the specified geometry"""
    filter = Filter(command="transpose", params=[FilterOption(name="dir", type=int), FilterOption(name="passthrough", type=int)])
    graph.append(filter)
    return graph


def transpose_vaapi(graph: Stream, dir: int, passthrough: int):
    """VAAPI VPP for transpose
    :param int dir: set transpose direction
    :param int passthrough: do not apply transposition if the input matches the specified geometry"""
    filter = Filter(command="transpose_vaapi", params=[FilterOption(name="dir", type=int), FilterOption(name="passthrough", type=int)])
    graph.append(filter)
    return graph


def virtualbass(graph: Stream, cutoff: float, strength: float):
    """Audio Virtual Bass.
    :param float cutoff: set virtual bass cutoff
    :param float strength: set virtual bass strength"""
    filter = Filter(command="virtualbass", params=[FilterOption(name="cutoff", type=float), FilterOption(name="strength", type=float)])
    graph.append(filter)
    return graph


def hflip(
    graph: Stream,
):
    """Horizontally flip the input video."""
    filter = Filter(command="hflip", params=[])
    graph.append(filter)
    return graph


def afir(
    graph: Stream,
    dry: float,
    wet: float,
    length: float,
    gtype: int,
    irnorm: float,
    irlink: bool,
    irgain: float,
    irfmt: int,
    maxir: float,
    response: bool,
    channel: int,
    size: int,
    rate: str,
    minp: int,
    maxp: int,
    nbirs: int,
    ir: int,
    precision: int,
    irload: int,
):
    """Apply Finite Impulse Response filter with supplied coefficients in additional stream(s).
    :param float dry: set dry gain
    :param float wet: set wet gain
    :param float length: set IR length
    :param int gtype: set IR auto gain type
    :param float irnorm: set IR norm
    :param bool irlink: set IR link
    :param float irgain: set IR gain
    :param int irfmt: set IR format
    :param float maxir: set max IR length
    :param bool response: show IR frequency response
    :param int channel: set IR channel to display frequency response
    :param int size: set video size
    :param str rate: set video rate
    :param int minp: set min partition size
    :param int maxp: set max partition size
    :param int nbirs: set number of input IRs
    :param int ir: select IR
    :param int precision: set processing precision
    :param int irload: set IR loading type"""
    filter = Filter(
        command="afir",
        params=[
            FilterOption(name="dry", type=float),
            FilterOption(name="wet", type=float),
            FilterOption(name="length", type=float),
            FilterOption(name="gtype", type=int),
            FilterOption(name="irnorm", type=float),
            FilterOption(name="irlink", type=bool),
            FilterOption(name="irgain", type=float),
            FilterOption(name="irfmt", type=int),
            FilterOption(name="maxir", type=float),
            FilterOption(name="response", type=bool),
            FilterOption(name="channel", type=int),
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="minp", type=int),
            FilterOption(name="maxp", type=int),
            FilterOption(name="nbirs", type=int),
            FilterOption(name="ir", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="irload", type=int),
        ],
    )
    graph.append(filter)
    return graph


def waveform(
    graph: Stream,
    mode: int,
    intensity: float,
    mirror: bool,
    display: int,
    components: int,
    envelope: int,
    filter: int,
    graticule: int,
    opacity: float,
    flags: str,
    scale: int,
    bgopacity: float,
    tint0: float,
    t0: float,
    tint1: float,
    t1: float,
    fitmode: int,
    input: int,
):
    """Video waveform monitor.
    :param int mode: set mode
    :param float intensity: set intensity
    :param bool mirror: set mirroring
    :param int display: set display mode
    :param int components: set components to display
    :param int envelope: set envelope to display
    :param int filter: set filter
    :param int graticule: set graticule
    :param float opacity: set graticule opacity
    :param str flags: set graticule flags
    :param int scale: set scale
    :param float bgopacity: set background opacity
    :param float tint0: set 1st tint
    :param float t0: set 1st tint
    :param float tint1: set 2nd tint
    :param float t1: set 2nd tint
    :param int fitmode: set fit mode
    :param int input: set input formats selection"""
    filter = Filter(
        command="waveform",
        params=[
            FilterOption(name="mode", type=int),
            FilterOption(name="intensity", type=float),
            FilterOption(name="mirror", type=bool),
            FilterOption(name="display", type=int),
            FilterOption(name="components", type=int),
            FilterOption(name="envelope", type=int),
            FilterOption(name="filter", type=int),
            FilterOption(name="graticule", type=int),
            FilterOption(name="opacity", type=float),
            FilterOption(name="flags", type=str),
            FilterOption(name="scale", type=int),
            FilterOption(name="bgopacity", type=float),
            FilterOption(name="tint0", type=float),
            FilterOption(name="t0", type=float),
            FilterOption(name="tint1", type=float),
            FilterOption(name="t1", type=float),
            FilterOption(name="fitmode", type=int),
            FilterOption(name="input", type=int),
        ],
    )
    graph.append(filter)
    return graph


def silencedetect(graph: Stream, n: float, d: int, mono: bool):
    """Detect silence.
    :param float n: set noise tolerance
    :param int d: set minimum duration in seconds
    :param bool mono: check each channel separately"""
    filter = Filter(
        command="silencedetect",
        params=[FilterOption(name="n", type=float), FilterOption(name="d", type=int), FilterOption(name="mono", type=bool)],
    )
    graph.append(filter)
    return graph


def anullsrc(graph: Stream, channel_layout: str, sample_rate: int, nb_samples: int, duration: int):
    """Null audio source, return empty audio frames.
    :param str channel_layout: set channel_layout
    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param int duration: set the audio duration"""
    filter = Filter(
        command="anullsrc",
        params=[
            FilterOption(name="channel_layout", type=str),
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="duration", type=int),
        ],
    )
    graph.append(filter)
    return graph


def afreqshift(graph: Stream, shift: float, level: float, order: int):
    """Apply frequency shifting to input audio.
    :param float shift: set frequency shift
    :param float level: set output level
    :param int order: set filter order"""
    filter = Filter(
        command="afreqshift",
        params=[FilterOption(name="shift", type=float), FilterOption(name="level", type=float), FilterOption(name="order", type=int)],
    )
    graph.append(filter)
    return graph


def aphaseshift(graph: Stream, shift: float, level: float, order: int):
    """Apply phase shifting to input audio.
    :param float shift: set phase shift
    :param float level: set output level
    :param int order: set filter order"""
    filter = Filter(
        command="aphaseshift",
        params=[FilterOption(name="shift", type=float), FilterOption(name="level", type=float), FilterOption(name="order", type=int)],
    )
    graph.append(filter)
    return graph


def colormatrix(graph: Stream, src: int, dst: int):
    """Convert color matrix.
    :param int src: set source color matrix
    :param int dst: set destination color matrix"""
    filter = Filter(command="colormatrix", params=[FilterOption(name="src", type=int), FilterOption(name="dst", type=int)])
    graph.append(filter)
    return graph


def pad_vaapi(graph: Stream, width: str, height: str, x: str, y: str, color: str, aspect: int):
    """Pad the input video.
    :param str width: set the pad area width
    :param str height: set the pad area height
    :param str x: set the x offset for the input image position
    :param str y: set the y offset for the input image position
    :param str color: set the color of the padded area border
    :param int aspect: pad to fit an aspect instead of a resolution"""
    filter = Filter(
        command="pad_vaapi",
        params=[
            FilterOption(name="width", type=str),
            FilterOption(name="height", type=str),
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="color", type=str),
            FilterOption(name="aspect", type=int),
        ],
    )
    graph.append(filter)
    return graph


def vectorscope(
    graph: Stream,
    mode: int,
    x: int,
    y: int,
    intensity: float,
    envelope: int,
    graticule: int,
    opacity: float,
    flags: str,
    bgopacity: float,
    lthreshold: float,
    hthreshold: float,
    colorspace: int,
    tint0: float,
    t0: float,
    tint1: float,
    t1: float,
):
    """Video vectorscope.
    :param int mode: set vectorscope mode
    :param int x: set color component on X axis
    :param int y: set color component on Y axis
    :param float intensity: set intensity
    :param int envelope: set envelope
    :param int graticule: set graticule
    :param float opacity: set graticule opacity
    :param str flags: set graticule flags
    :param float bgopacity: set background opacity
    :param float lthreshold: set low threshold
    :param float hthreshold: set high threshold
    :param int colorspace: set colorspace
    :param float tint0: set 1st tint
    :param float t0: set 1st tint
    :param float tint1: set 2nd tint
    :param float t1: set 2nd tint"""
    filter = Filter(
        command="vectorscope",
        params=[
            FilterOption(name="mode", type=int),
            FilterOption(name="x", type=int),
            FilterOption(name="y", type=int),
            FilterOption(name="intensity", type=float),
            FilterOption(name="envelope", type=int),
            FilterOption(name="graticule", type=int),
            FilterOption(name="opacity", type=float),
            FilterOption(name="flags", type=str),
            FilterOption(name="bgopacity", type=float),
            FilterOption(name="lthreshold", type=float),
            FilterOption(name="hthreshold", type=float),
            FilterOption(name="colorspace", type=int),
            FilterOption(name="tint0", type=float),
            FilterOption(name="t0", type=float),
            FilterOption(name="tint1", type=float),
            FilterOption(name="t1", type=float),
        ],
    )
    graph.append(filter)
    return graph


def showvolume(
    graph: Stream,
    rate: str,
    b: int,
    w: int,
    h: int,
    f: float,
    c: str,
    t: bool,
    v: bool,
    dm: float,
    dmc: str,
    o: int,
    s: int,
    p: float,
    m: int,
    ds: int,
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
    :param int o: set orientation
    :param int s: set step size
    :param float p: set background opacity
    :param int m: set mode
    :param int ds: set display scale"""
    filter = Filter(
        command="showvolume",
        params=[
            FilterOption(name="rate", type=str),
            FilterOption(name="b", type=int),
            FilterOption(name="w", type=int),
            FilterOption(name="h", type=int),
            FilterOption(name="f", type=float),
            FilterOption(name="c", type=str),
            FilterOption(name="t", type=bool),
            FilterOption(name="v", type=bool),
            FilterOption(name="dm", type=float),
            FilterOption(name="dmc", type=str),
            FilterOption(name="o", type=int),
            FilterOption(name="s", type=int),
            FilterOption(name="p", type=float),
            FilterOption(name="m", type=int),
            FilterOption(name="ds", type=int),
        ],
    )
    graph.append(filter)
    return graph


def colorize(graph: Stream, hue: float, saturation: float, lightness: float, mix: float):
    """Overlay a solid color on the video stream.
    :param float hue: set the hue
    :param float saturation: set the saturation
    :param float lightness: set the lightness
    :param float mix: set the mix of source lightness"""
    filter = Filter(
        command="colorize",
        params=[
            FilterOption(name="hue", type=float),
            FilterOption(name="saturation", type=float),
            FilterOption(name="lightness", type=float),
            FilterOption(name="mix", type=float),
        ],
    )
    graph.append(filter)
    return graph


def drawgraph(
    graph: Stream,
    m1: str,
    fg1: str,
    m2: str,
    fg2: str,
    m3: str,
    fg3: str,
    m4: str,
    fg4: str,
    bg: str,
    min: float,
    max: float,
    mode: int,
    slide: int,
    size: int,
    rate: str,
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
    :param int mode: set graph mode
    :param int slide: set slide mode
    :param int size: set graph size
    :param str rate: set video rate"""
    filter = Filter(
        command="drawgraph",
        params=[
            FilterOption(name="m1", type=str),
            FilterOption(name="fg1", type=str),
            FilterOption(name="m2", type=str),
            FilterOption(name="fg2", type=str),
            FilterOption(name="m3", type=str),
            FilterOption(name="fg3", type=str),
            FilterOption(name="m4", type=str),
            FilterOption(name="fg4", type=str),
            FilterOption(name="bg", type=str),
            FilterOption(name="min", type=float),
            FilterOption(name="max", type=float),
            FilterOption(name="mode", type=int),
            FilterOption(name="slide", type=int),
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
        ],
    )
    graph.append(filter)
    return graph


def adrawgraph(
    graph: Stream,
):
    """Draw a graph using input audio metadata."""
    filter = Filter(command="adrawgraph", params=[])
    graph.append(filter)
    return graph


def convolve(graph: Stream, planes: int, impulse: int, noise: float):
    """Convolve first video stream with second video stream.
    :param int planes: set planes to convolve
    :param int impulse: when to process impulses
    :param float noise: set noise"""
    filter = Filter(
        command="convolve",
        params=[FilterOption(name="planes", type=int), FilterOption(name="impulse", type=int), FilterOption(name="noise", type=float)],
    )
    graph.append(filter)
    return graph


def deconvolve(graph: Stream, planes: int, impulse: int, noise: float):
    """Deconvolve first video stream with second video stream.
    :param int planes: set planes to deconvolve
    :param int impulse: when to process impulses
    :param float noise: set noise"""
    filter = Filter(
        command="deconvolve",
        params=[FilterOption(name="planes", type=int), FilterOption(name="impulse", type=int), FilterOption(name="noise", type=float)],
    )
    graph.append(filter)
    return graph


def xcorrelate(graph: Stream, planes: int, secondary: int):
    """Cross-correlate first video stream with second video stream.
    :param int planes: set planes to cross-correlate
    :param int secondary: when to process secondary frame"""
    filter = Filter(command="xcorrelate", params=[FilterOption(name="planes", type=int), FilterOption(name="secondary", type=int)])
    graph.append(filter)
    return graph


def gblur(graph: Stream, sigma: float, steps: int, planes: int, sigmaV: float):
    """Apply Gaussian Blur filter.
    :param float sigma: set sigma
    :param int steps: set number of steps
    :param int planes: set planes to filter
    :param float sigmaV: set vertical sigma"""
    filter = Filter(
        command="gblur",
        params=[
            FilterOption(name="sigma", type=float),
            FilterOption(name="steps", type=int),
            FilterOption(name="planes", type=int),
            FilterOption(name="sigmaV", type=float),
        ],
    )
    graph.append(filter)
    return graph


def colorspace(
    graph: Stream,
    all: int,
    space: int,
    range: int,
    primaries: int,
    trc: int,
    format: int,
    fast: bool,
    dither: int,
    wpadapt: int,
    iall: int,
    ispace: int,
    irange: int,
    iprimaries: int,
    itrc: int,
):
    """Convert between colorspaces.
    :param int all: Set all color properties together
    :param int space: Output colorspace
    :param int range: Output color range
    :param int primaries: Output color primaries
    :param int trc: Output transfer characteristics
    :param int format: Output pixel format
    :param bool fast: Ignore primary chromaticity and gamma correction
    :param int dither: Dithering mode
    :param int wpadapt: Whitepoint adaptation method
    :param int iall: Set all input color properties together
    :param int ispace: Input colorspace
    :param int irange: Input color range
    :param int iprimaries: Input color primaries
    :param int itrc: Input transfer characteristics"""
    filter = Filter(
        command="colorspace",
        params=[
            FilterOption(name="all", type=int),
            FilterOption(name="space", type=int),
            FilterOption(name="range", type=int),
            FilterOption(name="primaries", type=int),
            FilterOption(name="trc", type=int),
            FilterOption(name="format", type=int),
            FilterOption(name="fast", type=bool),
            FilterOption(name="dither", type=int),
            FilterOption(name="wpadapt", type=int),
            FilterOption(name="iall", type=int),
            FilterOption(name="ispace", type=int),
            FilterOption(name="irange", type=int),
            FilterOption(name="iprimaries", type=int),
            FilterOption(name="itrc", type=int),
        ],
    )
    graph.append(filter)
    return graph


def shear(graph: Stream, shx: float, shy: float, fillcolor: str, interp: int):
    """Shear transform the input image.
    :param float shx: set x shear factor
    :param float shy: set y shear factor
    :param str fillcolor: set background fill color
    :param int interp: set interpolation"""
    filter = Filter(
        command="shear",
        params=[
            FilterOption(name="shx", type=float),
            FilterOption(name="shy", type=float),
            FilterOption(name="fillcolor", type=str),
            FilterOption(name="interp", type=int),
        ],
    )
    graph.append(filter)
    return graph


def buffersink(graph: Stream, pix_fmts: str, color_spaces: str, color_ranges: str):
    """Buffer video frames, and make them available to the end of the filter graph.
    :param str pix_fmts: set the supported pixel formats
    :param str color_spaces: set the supported color spaces
    :param str color_ranges: set the supported color ranges"""
    filter = Filter(
        command="buffersink",
        params=[
            FilterOption(name="pix_fmts", type=str),
            FilterOption(name="color_spaces", type=str),
            FilterOption(name="color_ranges", type=str),
        ],
    )
    graph.append(filter)
    return graph


def abuffersink(graph: Stream, sample_fmts: str, sample_rates: str, ch_layouts: str, all_channel_counts: bool):
    """Buffer audio frames, and make them available to the end of the filter graph.
    :param str sample_fmts: set the supported sample formats
    :param str sample_rates: set the supported sample rates
    :param str ch_layouts: set a '|'-separated list of supported channel layouts
    :param bool all_channel_counts: accept all channel counts"""
    filter = Filter(
        command="abuffersink",
        params=[
            FilterOption(name="sample_fmts", type=str),
            FilterOption(name="sample_rates", type=str),
            FilterOption(name="ch_layouts", type=str),
            FilterOption(name="all_channel_counts", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def cue(
    graph: Stream,
):
    """Delay filtering to match a cue."""
    filter = Filter(command="cue", params=[])
    graph.append(filter)
    return graph


def acue(
    graph: Stream,
):
    """Delay filtering to match a cue."""
    filter = Filter(command="acue", params=[])
    graph.append(filter)
    return graph


def photosensitivity(graph: Stream, frames: int, threshold: float, skip: int, bypass: bool):
    """Filter out photosensitive epilepsy seizure-inducing flashes.
    :param int frames: set how many frames to use
    :param float threshold: set detection threshold factor (lower is stricter)
    :param int skip: set pixels to skip when sampling frames
    :param bool bypass: leave frames unchanged"""
    filter = Filter(
        command="photosensitivity",
        params=[
            FilterOption(name="frames", type=int),
            FilterOption(name="threshold", type=float),
            FilterOption(name="skip", type=int),
            FilterOption(name="bypass", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def drmeter(graph: Stream, length: float):
    """Measure audio dynamic range.
    :param float length: set the window length"""
    filter = Filter(command="drmeter", params=[FilterOption(name="length", type=float)])
    graph.append(filter)
    return graph


def astats(graph: Stream, length: float, metadata: bool, reset: int, measure_perchannel: str, measure_overall: str):
    """Show time domain statistics about audio frames.
    :param float length: set the window length
    :param bool metadata: inject metadata in the filtergraph
    :param int reset: Set the number of frames over which cumulative stats are calculated before being reset
    :param str measure_perchannel: Select the parameters which are measured per channel
    :param str measure_overall: Select the parameters which are measured overall"""
    filter = Filter(
        command="astats",
        params=[
            FilterOption(name="length", type=float),
            FilterOption(name="metadata", type=bool),
            FilterOption(name="reset", type=int),
            FilterOption(name="measure_perchannel", type=str),
            FilterOption(name="measure_overall", type=str),
        ],
    )
    graph.append(filter)
    return graph


def bwdif_vulkan(
    graph: Stream,
):
    """Deinterlace Vulkan frames via bwdif"""
    filter = Filter(command="bwdif_vulkan", params=[])
    graph.append(filter)
    return graph


def fftdnoiz(
    graph: Stream, sigma: float, amount: float, block: int, overlap: float, method: int, prev: int, next: int, planes: int, window: int
):
    """Denoise frames using 3D FFT.
    :param float sigma: set denoise strength
    :param float amount: set amount of denoising
    :param int block: set block size
    :param float overlap: set block overlap
    :param int method: set method of denoising
    :param int prev: set number of previous frames for temporal denoising
    :param int next: set number of next frames for temporal denoising
    :param int planes: set planes to filter
    :param int window: set window function"""
    filter = Filter(
        command="fftdnoiz",
        params=[
            FilterOption(name="sigma", type=float),
            FilterOption(name="amount", type=float),
            FilterOption(name="block", type=int),
            FilterOption(name="overlap", type=float),
            FilterOption(name="method", type=int),
            FilterOption(name="prev", type=int),
            FilterOption(name="next", type=int),
            FilterOption(name="planes", type=int),
            FilterOption(name="window", type=int),
        ],
    )
    graph.append(filter)
    return graph


def dblur(graph: Stream, angle: float, radius: float, planes: int):
    """Apply Directional Blur filter.
    :param float angle: set angle
    :param float radius: set radius
    :param int planes: set planes to filter"""
    filter = Filter(
        command="dblur",
        params=[FilterOption(name="angle", type=float), FilterOption(name="radius", type=float), FilterOption(name="planes", type=int)],
    )
    graph.append(filter)
    return graph


def atadenoise(
    graph: Stream,
    _0a: float,
    _0b: float,
    _1a: float,
    _1b: float,
    _2a: float,
    _2b: float,
    s: int,
    p: str,
    a: int,
    _0s: float,
    _1s: float,
    _2s: float,
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
    :param int a: set variant of algorithm
    :param float 0s: set sigma for 1st plane
    :param float 1s: set sigma for 2nd plane
    :param float 2s: set sigma for 3rd plane"""
    filter = Filter(
        command="atadenoise",
        params=[
            FilterOption(name="0a", type=float),
            FilterOption(name="0b", type=float),
            FilterOption(name="1a", type=float),
            FilterOption(name="1b", type=float),
            FilterOption(name="2a", type=float),
            FilterOption(name="2b", type=float),
            FilterOption(name="s", type=int),
            FilterOption(name="p", type=str),
            FilterOption(name="a", type=int),
            FilterOption(name="0s", type=float),
            FilterOption(name="1s", type=float),
            FilterOption(name="2s", type=float),
        ],
    )
    graph.append(filter)
    return graph


def psnr(graph: Stream, stats_file: str, stats_version: int, output_max: bool):
    """Calculate the PSNR between two video streams.
    :param str stats_file: Set file where to store per-frame difference information
    :param int stats_version: Set the format version for the stats file.
    :param bool output_max: Add raw stats (max values) to the output log."""
    filter = Filter(
        command="psnr",
        params=[
            FilterOption(name="stats_file", type=str),
            FilterOption(name="stats_version", type=int),
            FilterOption(name="output_max", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def dialoguenhance(graph: Stream, original: float, enhance: float, voice: float):
    """Audio Dialogue Enhancement.
    :param float original: set original center factor
    :param float enhance: set dialogue enhance factor
    :param float voice: set voice detection factor"""
    filter = Filter(
        command="dialoguenhance",
        params=[
            FilterOption(name="original", type=float),
            FilterOption(name="enhance", type=float),
            FilterOption(name="voice", type=float),
        ],
    )
    graph.append(filter)
    return graph


def v360(
    graph: Stream,
    input: int,
    output: int,
    interp: int,
    w: int,
    h: int,
    in_stereo: int,
    out_stereo: int,
    in_forder: str,
    out_forder: str,
    in_frot: str,
    out_frot: str,
    in_pad: float,
    out_pad: float,
    fin_pad: int,
    fout_pad: int,
    yaw: float,
    pitch: float,
    roll: float,
    rorder: str,
    h_fov: float,
    v_fov: float,
    d_fov: float,
    h_flip: bool,
    v_flip: bool,
    d_flip: bool,
    ih_flip: bool,
    iv_flip: bool,
    in_trans: bool,
    out_trans: bool,
    ih_fov: float,
    iv_fov: float,
    id_fov: float,
    h_offset: float,
    v_offset: float,
    alpha_mask: bool,
    reset_rot: bool,
):
    """Convert 360 projection of video.
    :param int input: set input projection
    :param int output: set output projection
    :param int interp: set interpolation method
    :param int w: output width
    :param int h: output height
    :param int in_stereo: input stereo format
    :param int out_stereo: output stereo format
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
    :param bool reset_rot: reset rotation"""
    filter = Filter(
        command="v360",
        params=[
            FilterOption(name="input", type=int),
            FilterOption(name="output", type=int),
            FilterOption(name="interp", type=int),
            FilterOption(name="w", type=int),
            FilterOption(name="h", type=int),
            FilterOption(name="in_stereo", type=int),
            FilterOption(name="out_stereo", type=int),
            FilterOption(name="in_forder", type=str),
            FilterOption(name="out_forder", type=str),
            FilterOption(name="in_frot", type=str),
            FilterOption(name="out_frot", type=str),
            FilterOption(name="in_pad", type=float),
            FilterOption(name="out_pad", type=float),
            FilterOption(name="fin_pad", type=int),
            FilterOption(name="fout_pad", type=int),
            FilterOption(name="yaw", type=float),
            FilterOption(name="pitch", type=float),
            FilterOption(name="roll", type=float),
            FilterOption(name="rorder", type=str),
            FilterOption(name="h_fov", type=float),
            FilterOption(name="v_fov", type=float),
            FilterOption(name="d_fov", type=float),
            FilterOption(name="h_flip", type=bool),
            FilterOption(name="v_flip", type=bool),
            FilterOption(name="d_flip", type=bool),
            FilterOption(name="ih_flip", type=bool),
            FilterOption(name="iv_flip", type=bool),
            FilterOption(name="in_trans", type=bool),
            FilterOption(name="out_trans", type=bool),
            FilterOption(name="ih_fov", type=float),
            FilterOption(name="iv_fov", type=float),
            FilterOption(name="id_fov", type=float),
            FilterOption(name="h_offset", type=float),
            FilterOption(name="v_offset", type=float),
            FilterOption(name="alpha_mask", type=bool),
            FilterOption(name="reset_rot", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def fspp(graph: Stream, quality: int, qp: int, strength: int, use_bframe_qp: bool):
    """Apply Fast Simple Post-processing filter.
    :param int quality: set quality
    :param int qp: force a constant quantizer parameter
    :param int strength: set filter strength
    :param bool use_bframe_qp: use B-frames' QP"""
    filter = Filter(
        command="fspp",
        params=[
            FilterOption(name="quality", type=int),
            FilterOption(name="qp", type=int),
            FilterOption(name="strength", type=int),
            FilterOption(name="use_bframe_qp", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def yaepblur(graph: Stream, radius: int, planes: int, sigma: int):
    """Yet another edge preserving blur filter.
    :param int radius: set window radius
    :param int planes: set planes to filter
    :param int sigma: set blur strength"""
    filter = Filter(
        command="yaepblur",
        params=[FilterOption(name="radius", type=int), FilterOption(name="planes", type=int), FilterOption(name="sigma", type=int)],
    )
    graph.append(filter)
    return graph


def vflip(
    graph: Stream,
):
    """Flip the input video vertically."""
    filter = Filter(command="vflip", params=[])
    graph.append(filter)
    return graph


def normalize(graph: Stream, blackpt: str, whitept: str, smoothing: int, independence: float, strength: float):
    """Normalize RGB video.
    :param str blackpt: output color to which darkest input color is mapped
    :param str whitept: output color to which brightest input color is mapped
    :param int smoothing: amount of temporal smoothing of the input range, to reduce flicker
    :param float independence: proportion of independent to linked channel normalization
    :param float strength: strength of filter, from no effect to full normalization"""
    filter = Filter(
        command="normalize",
        params=[
            FilterOption(name="blackpt", type=str),
            FilterOption(name="whitept", type=str),
            FilterOption(name="smoothing", type=int),
            FilterOption(name="independence", type=float),
            FilterOption(name="strength", type=float),
        ],
    )
    graph.append(filter)
    return graph


def minterpolate(
    graph: Stream,
    fps: str,
    mi_mode: int,
    mc_mode: int,
    me_mode: int,
    me: int,
    mb_size: int,
    search_param: int,
    vsbmc: int,
    scd: int,
    scd_threshold: float,
):
    """Frame rate conversion using Motion Interpolation.
    :param str fps: output's frame rate
    :param int mi_mode: motion interpolation mode
    :param int mc_mode: motion compensation mode
    :param int me_mode: motion estimation mode
    :param int me: motion estimation method
    :param int mb_size: macroblock size
    :param int search_param: search parameter
    :param int vsbmc: variable-size block motion compensation
    :param int scd: scene change detection method
    :param float scd_threshold: scene change threshold"""
    filter = Filter(
        command="minterpolate",
        params=[
            FilterOption(name="fps", type=str),
            FilterOption(name="mi_mode", type=int),
            FilterOption(name="mc_mode", type=int),
            FilterOption(name="me_mode", type=int),
            FilterOption(name="me", type=int),
            FilterOption(name="mb_size", type=int),
            FilterOption(name="search_param", type=int),
            FilterOption(name="vsbmc", type=int),
            FilterOption(name="scd", type=int),
            FilterOption(name="scd_threshold", type=float),
        ],
    )
    graph.append(filter)
    return graph


def convolution(
    graph: Stream,
    _0m: str,
    _1m: str,
    _2m: str,
    _3m: str,
    _0rdiv: float,
    _1rdiv: float,
    _2rdiv: float,
    _3rdiv: float,
    _0bias: float,
    _1bias: float,
    _2bias: float,
    _3bias: float,
    _0mode: int,
    _1mode: int,
    _2mode: int,
    _3mode: int,
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
    :param int 0mode: set matrix mode for 1st plane
    :param int 1mode: set matrix mode for 2nd plane
    :param int 2mode: set matrix mode for 3rd plane
    :param int 3mode: set matrix mode for 4th plane"""
    filter = Filter(
        command="convolution",
        params=[
            FilterOption(name="0m", type=str),
            FilterOption(name="1m", type=str),
            FilterOption(name="2m", type=str),
            FilterOption(name="3m", type=str),
            FilterOption(name="0rdiv", type=float),
            FilterOption(name="1rdiv", type=float),
            FilterOption(name="2rdiv", type=float),
            FilterOption(name="3rdiv", type=float),
            FilterOption(name="0bias", type=float),
            FilterOption(name="1bias", type=float),
            FilterOption(name="2bias", type=float),
            FilterOption(name="3bias", type=float),
            FilterOption(name="0mode", type=int),
            FilterOption(name="1mode", type=int),
            FilterOption(name="2mode", type=int),
            FilterOption(name="3mode", type=int),
        ],
    )
    graph.append(filter)
    return graph


def prewitt(
    graph: Stream,
):
    """Apply prewitt operator."""
    filter = Filter(command="prewitt", params=[])
    graph.append(filter)
    return graph


def sobel(
    graph: Stream,
):
    """Apply sobel operator."""
    filter = Filter(command="sobel", params=[])
    graph.append(filter)
    return graph


def roberts(
    graph: Stream,
):
    """Apply roberts cross operator."""
    filter = Filter(command="roberts", params=[])
    graph.append(filter)
    return graph


def kirsch(
    graph: Stream,
):
    """Apply kirsch operator."""
    filter = Filter(command="kirsch", params=[])
    graph.append(filter)
    return graph


def scharr(
    graph: Stream,
):
    """Apply scharr operator."""
    filter = Filter(command="scharr", params=[])
    graph.append(filter)
    return graph


def ametadata(graph: Stream, mode: int, key: str, value: str, function: int, expr: str, file: str, direct: bool):
    """Manipulate audio frame metadata.
    :param int mode: set a mode of operation
    :param str key: set metadata key
    :param str value: set metadata value
    :param int function: function for comparing values
    :param str expr: set expression for expr function
    :param str file: set file where to print metadata information
    :param bool direct: reduce buffering when printing to user-set file or pipe"""
    filter = Filter(
        command="ametadata",
        params=[
            FilterOption(name="mode", type=int),
            FilterOption(name="key", type=str),
            FilterOption(name="value", type=str),
            FilterOption(name="function", type=int),
            FilterOption(name="expr", type=str),
            FilterOption(name="file", type=str),
            FilterOption(name="direct", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def metadata(graph: Stream, mode: int, key: str, value: str, function: int, expr: str, file: str, direct: bool):
    """Manipulate video frame metadata.
    :param int mode: set a mode of operation
    :param str key: set metadata key
    :param str value: set metadata value
    :param int function: function for comparing values
    :param str expr: set expression for expr function
    :param str file: set file where to print metadata information
    :param bool direct: reduce buffering when printing to user-set file or pipe"""
    filter = Filter(
        command="metadata",
        params=[
            FilterOption(name="mode", type=int),
            FilterOption(name="key", type=str),
            FilterOption(name="value", type=str),
            FilterOption(name="function", type=int),
            FilterOption(name="expr", type=str),
            FilterOption(name="file", type=str),
            FilterOption(name="direct", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def eq(
    graph: Stream,
    contrast: str,
    brightness: str,
    saturation: str,
    gamma: str,
    gamma_r: str,
    gamma_g: str,
    gamma_b: str,
    gamma_weight: str,
    eval: int,
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
    :param int eval: specify when to evaluate expressions"""
    filter = Filter(
        command="eq",
        params=[
            FilterOption(name="contrast", type=str),
            FilterOption(name="brightness", type=str),
            FilterOption(name="saturation", type=str),
            FilterOption(name="gamma", type=str),
            FilterOption(name="gamma_r", type=str),
            FilterOption(name="gamma_g", type=str),
            FilterOption(name="gamma_b", type=str),
            FilterOption(name="gamma_weight", type=str),
            FilterOption(name="eval", type=int),
        ],
    )
    graph.append(filter)
    return graph


def chorus(graph: Stream, in_gain: float, out_gain: float, delays: str, decays: str, speeds: str, depths: str):
    """Add a chorus effect to the audio.
    :param float in_gain: set input gain
    :param float out_gain: set output gain
    :param str delays: set delays
    :param str decays: set decays
    :param str speeds: set speeds
    :param str depths: set depths"""
    filter = Filter(
        command="chorus",
        params=[
            FilterOption(name="in_gain", type=float),
            FilterOption(name="out_gain", type=float),
            FilterOption(name="delays", type=str),
            FilterOption(name="decays", type=str),
            FilterOption(name="speeds", type=str),
            FilterOption(name="depths", type=str),
        ],
    )
    graph.append(filter)
    return graph


def readvitc(graph: Stream, scan_max: int, thr_b: float, thr_w: float):
    """Read vertical interval timecode and write it to frame metadata.
    :param int scan_max: maximum line numbers to scan for VITC data
    :param float thr_b: black color threshold
    :param float thr_w: white color threshold"""
    filter = Filter(
        command="readvitc",
        params=[FilterOption(name="scan_max", type=int), FilterOption(name="thr_b", type=float), FilterOption(name="thr_w", type=float)],
    )
    graph.append(filter)
    return graph


def codecview(graph: Stream, mv: str, qp: bool, mv_type: str, frame_type: str, block: bool):
    """Visualize information about some codecs.
    :param str mv: set motion vectors to visualize
    :param str mv_type: set motion vectors type
    :param str frame_type: set frame types to visualize motion vectors of
    :param bool block: set block partitioning structure to visualize"""
    filter = Filter(
        command="codecview",
        params=[
            FilterOption(name="mv", type=str),
            FilterOption(name="qp", type=bool),
            FilterOption(name="mv_type", type=str),
            FilterOption(name="frame_type", type=str),
            FilterOption(name="block", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def vif(
    graph: Stream,
):
    """Calculate the VIF between two video streams."""
    filter = Filter(command="vif", params=[])
    graph.append(filter)
    return graph


def aformat(graph: Stream, sample_fmts: str, sample_rates: str, channel_layouts: str):
    """Convert the input audio to one of the specified formats.
    :param str sample_fmts: A '|'-separated list of sample formats.
    :param str sample_rates: A '|'-separated list of sample rates.
    :param str channel_layouts: A '|'-separated list of channel layouts."""
    filter = Filter(
        command="aformat",
        params=[
            FilterOption(name="sample_fmts", type=str),
            FilterOption(name="sample_rates", type=str),
            FilterOption(name="channel_layouts", type=str),
        ],
    )
    graph.append(filter)
    return graph


def asoftclip(graph: Stream, type: int, threshold: float, output: float, param: float, oversample: int):
    """Audio Soft Clipper.
    :param int type: set softclip type
    :param float threshold: set softclip threshold
    :param float output: set softclip output gain
    :param float param: set softclip parameter
    :param int oversample: set oversample factor"""
    filter = Filter(
        command="asoftclip",
        params=[
            FilterOption(name="type", type=int),
            FilterOption(name="threshold", type=float),
            FilterOption(name="output", type=float),
            FilterOption(name="param", type=float),
            FilterOption(name="oversample", type=int),
        ],
    )
    graph.append(filter)
    return graph


def bitplanenoise(graph: Stream, bitplane: int, filter: bool):
    """Measure bit plane noise.
    :param int bitplane: set bit plane to use for measuring noise
    :param bool filter: show noisy pixels"""
    filter = Filter(command="bitplanenoise", params=[FilterOption(name="bitplane", type=int), FilterOption(name="filter", type=bool)])
    graph.append(filter)
    return graph


def stereotools(
    graph: Stream,
    level_in: float,
    level_out: float,
    balance_in: float,
    balance_out: float,
    softclip: bool,
    mutel: bool,
    muter: bool,
    phasel: bool,
    phaser: bool,
    mode: int,
    slev: float,
    sbal: float,
    mlev: float,
    mpan: float,
    base: float,
    delay: float,
    sclevel: float,
    phase: float,
    bmode_in: int,
    bmode_out: int,
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
    :param int mode: set stereo mode
    :param float slev: set side level
    :param float sbal: set side balance
    :param float mlev: set middle level
    :param float mpan: set middle pan
    :param float base: set stereo base
    :param float delay: set delay
    :param float sclevel: set S/C level
    :param float phase: set stereo phase
    :param int bmode_in: set balance in mode
    :param int bmode_out: set balance out mode"""
    filter = Filter(
        command="stereotools",
        params=[
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="balance_in", type=float),
            FilterOption(name="balance_out", type=float),
            FilterOption(name="softclip", type=bool),
            FilterOption(name="mutel", type=bool),
            FilterOption(name="muter", type=bool),
            FilterOption(name="phasel", type=bool),
            FilterOption(name="phaser", type=bool),
            FilterOption(name="mode", type=int),
            FilterOption(name="slev", type=float),
            FilterOption(name="sbal", type=float),
            FilterOption(name="mlev", type=float),
            FilterOption(name="mpan", type=float),
            FilterOption(name="base", type=float),
            FilterOption(name="delay", type=float),
            FilterOption(name="sclevel", type=float),
            FilterOption(name="phase", type=float),
            FilterOption(name="bmode_in", type=int),
            FilterOption(name="bmode_out", type=int),
        ],
    )
    graph.append(filter)
    return graph


def framerate(graph: Stream, fps: str, interp_start: int, interp_end: int, scene: float, flags: str):
    """Upsamples or downsamples progressive source between specified frame rates.
    :param str fps: required output frames per second rate
    :param int interp_start: point to start linear interpolation
    :param int interp_end: point to end linear interpolation
    :param float scene: scene change level
    :param str flags: set flags"""
    filter = Filter(
        command="framerate",
        params=[
            FilterOption(name="fps", type=str),
            FilterOption(name="interp_start", type=int),
            FilterOption(name="interp_end", type=int),
            FilterOption(name="scene", type=float),
            FilterOption(name="flags", type=str),
        ],
    )
    graph.append(filter)
    return graph


def nlmeans_opencl(graph: Stream, s: float, p: int, pc: int, r: int, rc: int):
    """Non-local means denoiser through OpenCL
    :param float s: denoising strength
    :param int p: patch size
    :param int pc: patch size for chroma planes
    :param int r: research window
    :param int rc: research window for chroma planes"""
    filter = Filter(
        command="nlmeans_opencl",
        params=[
            FilterOption(name="s", type=float),
            FilterOption(name="p", type=int),
            FilterOption(name="pc", type=int),
            FilterOption(name="r", type=int),
            FilterOption(name="rc", type=int),
        ],
    )
    graph.append(filter)
    return graph


def hwupload_cuda(
    graph: Stream,
):
    """Upload a system memory frame to a CUDA device."""
    filter = Filter(command="hwupload_cuda", params=[])
    graph.append(filter)
    return graph


def anequalizer(graph: Stream, params: str, curves: bool, size: int, mgain: float, fscale: int, colors: str):
    """Apply high-order audio parametric multi band equalizer.
    :param bool curves: draw frequency response curves
    :param int size: set video size
    :param float mgain: set max gain
    :param int fscale: set frequency scale
    :param str colors: set channels curves colors"""
    filter = Filter(
        command="anequalizer",
        params=[
            FilterOption(name="params", type=str),
            FilterOption(name="curves", type=bool),
            FilterOption(name="size", type=int),
            FilterOption(name="mgain", type=float),
            FilterOption(name="fscale", type=int),
            FilterOption(name="colors", type=str),
        ],
    )
    graph.append(filter)
    return graph


def telecine(graph: Stream, first_field: int, pattern: str):
    """Apply a telecine pattern.
    :param int first_field: select first field
    :param str pattern: pattern that describe for how many fields a frame is to be displayed"""
    filter = Filter(command="telecine", params=[FilterOption(name="first_field", type=int), FilterOption(name="pattern", type=str)])
    graph.append(filter)
    return graph


def pixelize(graph: Stream, width: int, w: int, height: int, h: int, mode: int, planes: str):
    """Pixelize video.
    :param int width: set block width
    :param int w: set block width
    :param int height: set block height
    :param int h: set block height
    :param int mode: set the pixelize mode
    :param str planes: set what planes to filter"""
    filter = Filter(
        command="pixelize",
        params=[
            FilterOption(name="width", type=int),
            FilterOption(name="w", type=int),
            FilterOption(name="height", type=int),
            FilterOption(name="h", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="planes", type=str),
        ],
    )
    graph.append(filter)
    return graph


def gradients(
    graph: Stream,
    size: int,
    rate: str,
    c0: str,
    c1: str,
    c2: str,
    c3: str,
    c4: str,
    c5: str,
    c6: str,
    c7: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    nb_colors: int,
    seed: int,
    duration: int,
    speed: float,
    type: int,
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
    :param int type: set gradient type"""
    filter = Filter(
        command="gradients",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="c0", type=str),
            FilterOption(name="c1", type=str),
            FilterOption(name="c2", type=str),
            FilterOption(name="c3", type=str),
            FilterOption(name="c4", type=str),
            FilterOption(name="c5", type=str),
            FilterOption(name="c6", type=str),
            FilterOption(name="c7", type=str),
            FilterOption(name="x0", type=int),
            FilterOption(name="y0", type=int),
            FilterOption(name="x1", type=int),
            FilterOption(name="y1", type=int),
            FilterOption(name="nb_colors", type=int),
            FilterOption(name="seed", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="speed", type=float),
            FilterOption(name="type", type=int),
        ],
    )
    graph.append(filter)
    return graph


def smartblur(
    graph: Stream,
    luma_radius: float,
    lr: float,
    luma_strength: float,
    ls: float,
    luma_threshold: int,
    lt: int,
    chroma_radius: float,
    cr: float,
    chroma_strength: float,
    cs: float,
    chroma_threshold: int,
    ct: int,
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
    :param int ct: set chroma threshold"""
    filter = Filter(
        command="smartblur",
        params=[
            FilterOption(name="luma_radius", type=float),
            FilterOption(name="lr", type=float),
            FilterOption(name="luma_strength", type=float),
            FilterOption(name="ls", type=float),
            FilterOption(name="luma_threshold", type=int),
            FilterOption(name="lt", type=int),
            FilterOption(name="chroma_radius", type=float),
            FilterOption(name="cr", type=float),
            FilterOption(name="chroma_strength", type=float),
            FilterOption(name="cs", type=float),
            FilterOption(name="chroma_threshold", type=int),
            FilterOption(name="ct", type=int),
        ],
    )
    graph.append(filter)
    return graph


def acontrast(graph: Stream, contrast: float):
    """Simple audio dynamic range compression/expansion filter.
    :param float contrast: set contrast"""
    filter = Filter(command="acontrast", params=[FilterOption(name="contrast", type=float)])
    graph.append(filter)
    return graph


def ssim360(
    graph: Stream,
    stats_file: str,
    compute_chroma: int,
    frame_skip_ratio: int,
    ref_projection: int,
    main_projection: int,
    ref_stereo: int,
    main_stereo: int,
    ref_pad: float,
    main_pad: float,
    use_tape: int,
    heatmap_str: str,
    default_heatmap_width: int,
    default_heatmap_height: int,
):
    """Calculate the SSIM between two 360 video streams.
    :param str stats_file: Set file where to store per-frame difference information
    :param int compute_chroma: Specifies if non-luma channels must be computed
    :param int frame_skip_ratio: Specifies the number of frames to be skipped from evaluation, for every evaluated frame
    :param int ref_projection: projection of the reference video
    :param int main_projection: projection of the main video
    :param int ref_stereo: stereo format of the reference video
    :param int main_stereo: stereo format of main video
    :param float ref_pad: Expansion (padding) coefficient for each cube face of the reference video
    :param float main_pad: Expansion (padding) coeffiecient for each cube face of the main video
    :param int use_tape: Specifies if the tape based SSIM 360 algorithm must be used independent of the input video types
    :param str heatmap_str: Heatmap data for view-based evaluation. For heatmap file format, please refer to EntSphericalVideoHeatmapData.
    :param int default_heatmap_width: Default heatmap dimension. Will be used when dimension is not specified in heatmap data.
    :param int default_heatmap_height: Default heatmap dimension. Will be used when dimension is not specified in heatmap data."""
    filter = Filter(
        command="ssim360",
        params=[
            FilterOption(name="stats_file", type=str),
            FilterOption(name="compute_chroma", type=int),
            FilterOption(name="frame_skip_ratio", type=int),
            FilterOption(name="ref_projection", type=int),
            FilterOption(name="main_projection", type=int),
            FilterOption(name="ref_stereo", type=int),
            FilterOption(name="main_stereo", type=int),
            FilterOption(name="ref_pad", type=float),
            FilterOption(name="main_pad", type=float),
            FilterOption(name="use_tape", type=int),
            FilterOption(name="heatmap_str", type=str),
            FilterOption(name="default_heatmap_width", type=int),
            FilterOption(name="default_heatmap_height", type=int),
        ],
    )
    graph.append(filter)
    return graph


def mcompand(graph: Stream, args: str):
    """Multiband Compress or expand audio dynamic range.
    :param str args: set parameters for each band"""
    filter = Filter(command="mcompand", params=[FilterOption(name="args", type=str)])
    graph.append(filter)
    return graph


def anlmdn(graph: Stream, strength: float, patch: int, research: int, output: int, smooth: float):
    """Reduce broadband noise from stream using Non-Local Means.
    :param float strength: set denoising strength
    :param int patch: set patch duration
    :param int research: set research duration
    :param int output: set output mode
    :param float smooth: set smooth factor"""
    filter = Filter(
        command="anlmdn",
        params=[
            FilterOption(name="strength", type=float),
            FilterOption(name="patch", type=int),
            FilterOption(name="research", type=int),
            FilterOption(name="output", type=int),
            FilterOption(name="smooth", type=float),
        ],
    )
    graph.append(filter)
    return graph


def fade(graph: Stream, type: int, start_frame: int, nb_frames: int, alpha: bool, start_time: int, duration: int, color: str):
    """Fade in/out input video.
    :param int type: set the fade direction
    :param int start_frame: Number of the first frame to which to apply the effect.
    :param int nb_frames: Number of frames to which the effect should be applied.
    :param bool alpha: fade alpha if it is available on the input
    :param int start_time: Number of seconds of the beginning of the effect.
    :param int duration: Duration of the effect in seconds.
    :param str color: set color"""
    filter = Filter(
        command="fade",
        params=[
            FilterOption(name="type", type=int),
            FilterOption(name="start_frame", type=int),
            FilterOption(name="nb_frames", type=int),
            FilterOption(name="alpha", type=bool),
            FilterOption(name="start_time", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="color", type=str),
        ],
    )
    graph.append(filter)
    return graph


def freezeframes(graph: Stream, first: int, last: int, replace: int):
    """Freeze video frames.
    :param int first: set first frame to freeze
    :param int last: set last frame to freeze
    :param int replace: set frame to replace"""
    filter = Filter(
        command="freezeframes",
        params=[FilterOption(name="first", type=int), FilterOption(name="last", type=int), FilterOption(name="replace", type=int)],
    )
    graph.append(filter)
    return graph


def phase(graph: Stream, mode: int):
    """Phase shift fields.
    :param int mode: set phase mode"""
    filter = Filter(command="phase", params=[FilterOption(name="mode", type=int)])
    graph.append(filter)
    return graph


def headphone(graph: Stream, map: str, gain: float, lfe: float, type: int, size: int, hrir: int):
    """Apply headphone binaural spatialization with HRTFs in additional streams.
    :param str map: set channels convolution mappings
    :param float gain: set gain in dB
    :param float lfe: set lfe gain in dB
    :param int type: set processing
    :param int size: set frame size
    :param int hrir: set hrir format"""
    filter = Filter(
        command="headphone",
        params=[
            FilterOption(name="map", type=str),
            FilterOption(name="gain", type=float),
            FilterOption(name="lfe", type=float),
            FilterOption(name="type", type=int),
            FilterOption(name="size", type=int),
            FilterOption(name="hrir", type=int),
        ],
    )
    graph.append(filter)
    return graph


def gblur_vulkan(graph: Stream, sigma: float, sigmaV: float, planes: int, size: int, sizeV: int):
    """Gaussian Blur in Vulkan
    :param float sigma: Set sigma
    :param float sigmaV: Set vertical sigma
    :param int planes: Set planes to filter
    :param int size: Set kernel size
    :param int sizeV: Set vertical kernel size"""
    filter = Filter(
        command="gblur_vulkan",
        params=[
            FilterOption(name="sigma", type=float),
            FilterOption(name="sigmaV", type=float),
            FilterOption(name="planes", type=int),
            FilterOption(name="size", type=int),
            FilterOption(name="sizeV", type=int),
        ],
    )
    graph.append(filter)
    return graph


def yadif(
    graph: Stream,
):
    """Deinterlace the input image."""
    filter = Filter(command="yadif", params=[])
    graph.append(filter)
    return graph


def hysteresis(graph: Stream, planes: int, threshold: int):
    """Grow first stream into second stream by connecting components.
    :param int planes: set planes
    :param int threshold: set threshold"""
    filter = Filter(command="hysteresis", params=[FilterOption(name="planes", type=int), FilterOption(name="threshold", type=int)])
    graph.append(filter)
    return graph


def dnn_detect(
    graph: Stream,
    dnn_backend: int,
    model: str,
    input: str,
    output: str,
    backend_configs: str,
    options: str,
    _async: bool,
    confidence: float,
    labels: str,
    model_type: int,
    cell_w: int,
    cell_h: int,
    nb_classes: int,
    anchors: str,
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
    :param int model_type: DNN detection model type
    :param int cell_w: cell width
    :param int cell_h: cell height
    :param int nb_classes: The number of class
    :param str anchors: anchors, splited by '&'"""
    filter = Filter(
        command="dnn_detect",
        params=[
            FilterOption(name="dnn_backend", type=int),
            FilterOption(name="model", type=str),
            FilterOption(name="input", type=str),
            FilterOption(name="output", type=str),
            FilterOption(name="backend_configs", type=str),
            FilterOption(name="options", type=str),
            FilterOption(name="async", type=bool),
            FilterOption(name="confidence", type=float),
            FilterOption(name="labels", type=str),
            FilterOption(name="model_type", type=int),
            FilterOption(name="cell_w", type=int),
            FilterOption(name="cell_h", type=int),
            FilterOption(name="nb_classes", type=int),
            FilterOption(name="anchors", type=str),
        ],
    )
    graph.append(filter)
    return graph


def floodfill(graph: Stream, x: int, y: int, s0: int, s1: int, s2: int, s3: int, d0: int, d1: int, d2: int, d3: int):
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
    :param int d3: set destination #3 component value"""
    filter = Filter(
        command="floodfill",
        params=[
            FilterOption(name="x", type=int),
            FilterOption(name="y", type=int),
            FilterOption(name="s0", type=int),
            FilterOption(name="s1", type=int),
            FilterOption(name="s2", type=int),
            FilterOption(name="s3", type=int),
            FilterOption(name="d0", type=int),
            FilterOption(name="d1", type=int),
            FilterOption(name="d2", type=int),
            FilterOption(name="d3", type=int),
        ],
    )
    graph.append(filter)
    return graph


def delogo(graph: Stream, x: str, y: str, w: str, h: str, show: bool):
    """Remove logo from input video.
    :param str x: set logo x position
    :param str y: set logo y position
    :param str w: set logo width
    :param str h: set logo height
    :param bool show: show delogo area"""
    filter = Filter(
        command="delogo",
        params=[
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="w", type=str),
            FilterOption(name="h", type=str),
            FilterOption(name="show", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def repeatfields(
    graph: Stream,
):
    """Hard repeat fields based on MPEG repeat field flag."""
    filter = Filter(command="repeatfields", params=[])
    graph.append(filter)
    return graph


def color(graph: Stream, color: str, size: int, rate: str, duration: int, sar: int):
    """Provide an uniformly colored input.
    :param str color: set color
    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio"""
    filter = Filter(
        command="color",
        params=[
            FilterOption(name="color", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
        ],
    )
    graph.append(filter)
    return graph


def haldclutsrc(graph: Stream, level: int, rate: str, duration: int, sar: int):
    """Provide an identity Hald CLUT.
    :param int level: set level
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio"""
    filter = Filter(
        command="haldclutsrc",
        params=[
            FilterOption(name="level", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
        ],
    )
    graph.append(filter)
    return graph


def nullsrc(
    graph: Stream,
):
    """Null video source, return unprocessed video frames."""
    filter = Filter(command="nullsrc", params=[])
    graph.append(filter)
    return graph


def testsrc(graph: Stream, size: int, rate: str, duration: int, sar: int, decimals: int):
    """Generate test pattern.
    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param int decimals: set number of decimals to show"""
    filter = Filter(
        command="testsrc",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
            FilterOption(name="decimals", type=int),
        ],
    )
    graph.append(filter)
    return graph


def testsrc2(graph: Stream, size: int, rate: str, duration: int, sar: int, alpha: int):
    """Generate another test pattern.
    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param int alpha: set global alpha (opacity)"""
    filter = Filter(
        command="testsrc2",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
            FilterOption(name="alpha", type=int),
        ],
    )
    graph.append(filter)
    return graph


def rgbtestsrc(graph: Stream, size: int, rate: str, duration: int, sar: int, complement: bool):
    """Generate RGB test pattern.
    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param bool complement: set complement colors"""
    filter = Filter(
        command="rgbtestsrc",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
            FilterOption(name="complement", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def yuvtestsrc(
    graph: Stream,
):
    """Generate YUV test pattern."""
    filter = Filter(command="yuvtestsrc", params=[])
    graph.append(filter)
    return graph


def pal75bars(
    graph: Stream,
):
    """Generate PAL 75% color bars."""
    filter = Filter(command="pal75bars", params=[])
    graph.append(filter)
    return graph


def pal100bars(
    graph: Stream,
):
    """Generate PAL 100% color bars."""
    filter = Filter(command="pal100bars", params=[])
    graph.append(filter)
    return graph


def smptebars(
    graph: Stream,
):
    """Generate SMPTE color bars."""
    filter = Filter(command="smptebars", params=[])
    graph.append(filter)
    return graph


def smptehdbars(
    graph: Stream,
):
    """Generate SMPTE HD color bars."""
    filter = Filter(command="smptehdbars", params=[])
    graph.append(filter)
    return graph


def allyuv(
    graph: Stream,
):
    """Generate all yuv colors."""
    filter = Filter(command="allyuv", params=[])
    graph.append(filter)
    return graph


def allrgb(
    graph: Stream,
):
    """Generate all RGB colors."""
    filter = Filter(command="allrgb", params=[])
    graph.append(filter)
    return graph


def colorspectrum(graph: Stream, size: int, rate: str, duration: int, sar: int, type: int):
    """Generate colors spectrum.
    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param int type: set the color spectrum type"""
    filter = Filter(
        command="colorspectrum",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
            FilterOption(name="type", type=int),
        ],
    )
    graph.append(filter)
    return graph


def colorchart(graph: Stream, rate: str, duration: int, sar: int, patch_size: int, preset: int):
    """Generate color checker chart.
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param int patch_size: set the single patch size
    :param int preset: set the color checker chart preset"""
    filter = Filter(
        command="colorchart",
        params=[
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
            FilterOption(name="patch_size", type=int),
            FilterOption(name="preset", type=int),
        ],
    )
    graph.append(filter)
    return graph


def zoneplate(
    graph: Stream,
    size: int,
    rate: str,
    duration: int,
    sar: int,
    precision: int,
    xo: int,
    yo: int,
    to: int,
    k0: int,
    kx: int,
    ky: int,
    kt: int,
    kxt: int,
    kyt: int,
    kxy: int,
    kx2: int,
    ky2: int,
    kt2: int,
    ku: int,
    kv: int,
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
    :param int kv: set 0-order V-color phase"""
    filter = Filter(
        command="zoneplate",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="xo", type=int),
            FilterOption(name="yo", type=int),
            FilterOption(name="to", type=int),
            FilterOption(name="k0", type=int),
            FilterOption(name="kx", type=int),
            FilterOption(name="ky", type=int),
            FilterOption(name="kt", type=int),
            FilterOption(name="kxt", type=int),
            FilterOption(name="kyt", type=int),
            FilterOption(name="kxy", type=int),
            FilterOption(name="kx2", type=int),
            FilterOption(name="ky2", type=int),
            FilterOption(name="kt2", type=int),
            FilterOption(name="ku", type=int),
            FilterOption(name="kv", type=int),
        ],
    )
    graph.append(filter)
    return graph


def colorchannelmixer(
    graph: Stream,
    rr: float,
    rg: float,
    rb: float,
    ra: float,
    gr: float,
    gg: float,
    gb: float,
    ga: float,
    br: float,
    bg: float,
    bb: float,
    ba: float,
    ar: float,
    ag: float,
    ab: float,
    aa: float,
    pc: int,
    pa: float,
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
    :param int pc: set the preserve color mode
    :param float pa: set the preserve color amount"""
    filter = Filter(
        command="colorchannelmixer",
        params=[
            FilterOption(name="rr", type=float),
            FilterOption(name="rg", type=float),
            FilterOption(name="rb", type=float),
            FilterOption(name="ra", type=float),
            FilterOption(name="gr", type=float),
            FilterOption(name="gg", type=float),
            FilterOption(name="gb", type=float),
            FilterOption(name="ga", type=float),
            FilterOption(name="br", type=float),
            FilterOption(name="bg", type=float),
            FilterOption(name="bb", type=float),
            FilterOption(name="ba", type=float),
            FilterOption(name="ar", type=float),
            FilterOption(name="ag", type=float),
            FilterOption(name="ab", type=float),
            FilterOption(name="aa", type=float),
            FilterOption(name="pc", type=int),
            FilterOption(name="pa", type=float),
        ],
    )
    graph.append(filter)
    return graph


def backgroundkey(graph: Stream, threshold: float, similarity: float, blend: float):
    """Turns a static background into transparency.
    :param float threshold: set the scene change threshold
    :param float similarity: set the similarity
    :param float blend: set the blend value"""
    filter = Filter(
        command="backgroundkey",
        params=[
            FilterOption(name="threshold", type=float),
            FilterOption(name="similarity", type=float),
            FilterOption(name="blend", type=float),
        ],
    )
    graph.append(filter)
    return graph


def displace(graph: Stream, edge: int):
    """Displace pixels.
    :param int edge: set edge mode"""
    filter = Filter(command="displace", params=[FilterOption(name="edge", type=int)])
    graph.append(filter)
    return graph


def pan(graph: Stream, args: str):
    """Remix channels with coefficients (panning)."""
    filter = Filter(command="pan", params=[FilterOption(name="args", type=str)])
    graph.append(filter)
    return graph


def afdelaysrc(graph: Stream, delay: float, sample_rate: int, nb_samples: int, taps: int, channel_layout: str):
    """Generate a Fractional delay FIR coefficients.
    :param float delay: set fractional delay
    :param int sample_rate: set sample rate
    :param int nb_samples: set the number of samples per requested frame
    :param int taps: set number of taps for delay filter
    :param str channel_layout: set channel layout"""
    filter = Filter(
        command="afdelaysrc",
        params=[
            FilterOption(name="delay", type=float),
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="taps", type=int),
            FilterOption(name="channel_layout", type=str),
        ],
    )
    graph.append(filter)
    return graph


def tinterlace(graph: Stream, mode: int, flags: str):
    """Perform temporal field interlacing.
    :param int mode: select interlace mode
    :param str flags: set flags"""
    filter = Filter(command="tinterlace", params=[FilterOption(name="mode", type=int), FilterOption(name="flags", type=str)])
    graph.append(filter)
    return graph


def interlace(graph: Stream, scan: int, lowpass: int):
    """Convert progressive video into interlaced.
    :param int scan: scanning mode
    :param int lowpass: set vertical low-pass filter"""
    filter = Filter(command="interlace", params=[FilterOption(name="scan", type=int), FilterOption(name="lowpass", type=int)])
    graph.append(filter)
    return graph


def format(
    graph: Stream,
):
    """Convert the input video to one of the specified pixel formats."""
    filter = Filter(command="format", params=[])
    graph.append(filter)
    return graph


def noformat(
    graph: Stream,
):
    """Force libavfilter not to use any of the specified pixel formats for the input to the next filter."""
    filter = Filter(command="noformat", params=[])
    graph.append(filter)
    return graph


def framestep(graph: Stream, step: int):
    """Select one frame every N frames.
    :param int step: set frame step"""
    filter = Filter(command="framestep", params=[FilterOption(name="step", type=int)])
    graph.append(filter)
    return graph


def ssim(graph: Stream, stats_file: str):
    """Calculate the SSIM between two video streams.
    :param str stats_file: Set file where to store per-frame difference information"""
    filter = Filter(command="ssim", params=[FilterOption(name="stats_file", type=str)])
    graph.append(filter)
    return graph


def cropdetect(
    graph: Stream, limit: float, round: int, reset: int, skip: int, max_outliers: int, mode: int, high: float, low: float, mv_threshold: int
):
    """Auto-detect crop size.
    :param float limit: Threshold below which the pixel is considered black
    :param int round: Value by which the width/height should be divisible
    :param int reset: Recalculate the crop area after this many frames
    :param int skip: Number of initial frames to skip
    :param int max_outliers: Threshold count of outliers
    :param int mode: set mode
    :param float high: Set high threshold for edge detection
    :param float low: Set low threshold for edge detection
    :param int mv_threshold: motion vector threshold when estimating video window size"""
    filter = Filter(
        command="cropdetect",
        params=[
            FilterOption(name="limit", type=float),
            FilterOption(name="round", type=int),
            FilterOption(name="reset", type=int),
            FilterOption(name="skip", type=int),
            FilterOption(name="max_outliers", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="high", type=float),
            FilterOption(name="low", type=float),
            FilterOption(name="mv_threshold", type=int),
        ],
    )
    graph.append(filter)
    return graph


def chromashift(graph: Stream, cbh: int, cbv: int, crh: int, crv: int, edge: int):
    """Shift chroma.
    :param int cbh: shift chroma-blue horizontally
    :param int cbv: shift chroma-blue vertically
    :param int crh: shift chroma-red horizontally
    :param int crv: shift chroma-red vertically
    :param int edge: set edge operation"""
    filter = Filter(
        command="chromashift",
        params=[
            FilterOption(name="cbh", type=int),
            FilterOption(name="cbv", type=int),
            FilterOption(name="crh", type=int),
            FilterOption(name="crv", type=int),
            FilterOption(name="edge", type=int),
        ],
    )
    graph.append(filter)
    return graph


def rgbashift(graph: Stream, rh: int, rv: int, gh: int, gv: int, bh: int, bv: int, ah: int, av: int, edge: int):
    """Shift RGBA.
    :param int rh: shift red horizontally
    :param int rv: shift red vertically
    :param int gh: shift green horizontally
    :param int gv: shift green vertically
    :param int bh: shift blue horizontally
    :param int bv: shift blue vertically
    :param int ah: shift alpha horizontally
    :param int av: shift alpha vertically
    :param int edge: set edge operation"""
    filter = Filter(
        command="rgbashift",
        params=[
            FilterOption(name="rh", type=int),
            FilterOption(name="rv", type=int),
            FilterOption(name="gh", type=int),
            FilterOption(name="gv", type=int),
            FilterOption(name="bh", type=int),
            FilterOption(name="bv", type=int),
            FilterOption(name="ah", type=int),
            FilterOption(name="av", type=int),
            FilterOption(name="edge", type=int),
        ],
    )
    graph.append(filter)
    return graph


def segment(graph: Stream, timestamps: str, frames: str):
    """Segment video stream.
    :param str timestamps: timestamps of input at which to split input
    :param str frames: frames at which to split input"""
    filter = Filter(command="segment", params=[FilterOption(name="timestamps", type=str), FilterOption(name="frames", type=str)])
    graph.append(filter)
    return graph


def asegment(graph: Stream, timestamps: str, samples: str):
    """Segment audio stream.
    :param str timestamps: timestamps of input at which to split input
    :param str samples: samples at which to split input"""
    filter = Filter(command="asegment", params=[FilterOption(name="timestamps", type=str), FilterOption(name="samples", type=str)])
    graph.append(filter)
    return graph


def aiir(
    graph: Stream,
    zeros: str,
    poles: str,
    gains: str,
    dry: float,
    wet: float,
    format: int,
    process: int,
    precision: int,
    normalize: bool,
    mix: float,
    response: bool,
    channel: int,
    size: int,
    rate: str,
):
    """Apply Infinite Impulse Response filter with supplied coefficients.
    :param str zeros: set B/numerator/zeros/reflection coefficients
    :param str poles: set A/denominator/poles/ladder coefficients
    :param str gains: set channels gains
    :param float dry: set dry gain
    :param float wet: set wet gain
    :param int format: set coefficients format
    :param int process: set kind of processing
    :param int precision: set filtering precision
    :param bool normalize: normalize coefficients
    :param float mix: set mix
    :param bool response: show IR frequency response
    :param int channel: set IR channel to display frequency response
    :param int size: set video size
    :param str rate: set video rate"""
    filter = Filter(
        command="aiir",
        params=[
            FilterOption(name="zeros", type=str),
            FilterOption(name="poles", type=str),
            FilterOption(name="gains", type=str),
            FilterOption(name="dry", type=float),
            FilterOption(name="wet", type=float),
            FilterOption(name="format", type=int),
            FilterOption(name="process", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="normalize", type=bool),
            FilterOption(name="mix", type=float),
            FilterOption(name="response", type=bool),
            FilterOption(name="channel", type=int),
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
        ],
    )
    graph.append(filter)
    return graph


def showspatial(graph: Stream, size: int, win_size: int, win_func: int, rate: str):
    """Convert input audio to a spatial video output.
    :param int size: set video size
    :param int win_size: set window size
    :param int win_func: set window function
    :param str rate: set video rate"""
    filter = Filter(
        command="showspatial",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="win_size", type=int),
            FilterOption(name="win_func", type=int),
            FilterOption(name="rate", type=str),
        ],
    )
    graph.append(filter)
    return graph


def overlay(graph: Stream, x: str, y: str, eof_action: int, eval: int, shortest: bool, format: int, repeatlast: bool, alpha: int):
    """Overlay a video source on top of the input.
    :param str x: set the x expression
    :param str y: set the y expression
    :param int eof_action: Action to take when encountering EOF from secondary input
    :param int eval: specify when to evaluate expressions
    :param bool shortest: force termination when the shortest input terminates
    :param int format: set output format
    :param bool repeatlast: repeat overlay of the last overlay frame
    :param int alpha: alpha format"""
    filter = Filter(
        command="overlay",
        params=[
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="eof_action", type=int),
            FilterOption(name="eval", type=int),
            FilterOption(name="shortest", type=bool),
            FilterOption(name="format", type=int),
            FilterOption(name="repeatlast", type=bool),
            FilterOption(name="alpha", type=int),
        ],
    )
    graph.append(filter)
    return graph


def null(
    graph: Stream,
):
    """Pass the source unchanged to the output."""
    filter = Filter(command="null", params=[])
    graph.append(filter)
    return graph


def deshake(
    graph: Stream,
    x: int,
    y: int,
    w: int,
    h: int,
    rx: int,
    ry: int,
    edge: int,
    blocksize: int,
    contrast: int,
    search: int,
    filename: str,
    opencl: bool,
):
    """Stabilize shaky video.
    :param int x: set x for the rectangular search area
    :param int y: set y for the rectangular search area
    :param int w: set width for the rectangular search area
    :param int h: set height for the rectangular search area
    :param int rx: set x for the rectangular search area
    :param int ry: set y for the rectangular search area
    :param int edge: set edge mode
    :param int blocksize: set motion search blocksize
    :param int contrast: set contrast threshold for blocks
    :param int search: set search strategy
    :param str filename: set motion search detailed log file name
    :param bool opencl: ignored"""
    filter = Filter(
        command="deshake",
        params=[
            FilterOption(name="x", type=int),
            FilterOption(name="y", type=int),
            FilterOption(name="w", type=int),
            FilterOption(name="h", type=int),
            FilterOption(name="rx", type=int),
            FilterOption(name="ry", type=int),
            FilterOption(name="edge", type=int),
            FilterOption(name="blocksize", type=int),
            FilterOption(name="contrast", type=int),
            FilterOption(name="search", type=int),
            FilterOption(name="filename", type=str),
            FilterOption(name="opencl", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def estdif(graph: Stream, mode: int, parity: int, deint: int, rslope: int, redge: int, ecost: int, mcost: int, dcost: int, interp: int):
    """Apply Edge Slope Tracing deinterlace.
    :param int mode: specify the mode
    :param int parity: specify the assumed picture field parity
    :param int deint: specify which frames to deinterlace
    :param int rslope: specify the search radius for edge slope tracing
    :param int redge: specify the search radius for best edge matching
    :param int ecost: specify the edge cost for edge matching
    :param int mcost: specify the middle cost for edge matching
    :param int dcost: specify the distance cost for edge matching
    :param int interp: specify the type of interpolation"""
    filter = Filter(
        command="estdif",
        params=[
            FilterOption(name="mode", type=int),
            FilterOption(name="parity", type=int),
            FilterOption(name="deint", type=int),
            FilterOption(name="rslope", type=int),
            FilterOption(name="redge", type=int),
            FilterOption(name="ecost", type=int),
            FilterOption(name="mcost", type=int),
            FilterOption(name="dcost", type=int),
            FilterOption(name="interp", type=int),
        ],
    )
    graph.append(filter)
    return graph


def anoisesrc(graph: Stream, sample_rate: int, amplitude: float, duration: int, color: int, seed: int, nb_samples: int, density: float):
    """Generate a noise audio signal.
    :param int sample_rate: set sample rate
    :param float amplitude: set amplitude
    :param int duration: set duration
    :param int color: set noise color
    :param int seed: set random seed
    :param int nb_samples: set the number of samples per requested frame
    :param float density: set density"""
    filter = Filter(
        command="anoisesrc",
        params=[
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="amplitude", type=float),
            FilterOption(name="duration", type=int),
            FilterOption(name="color", type=int),
            FilterOption(name="seed", type=int),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="density", type=float),
        ],
    )
    graph.append(filter)
    return graph


def aecho(graph: Stream, in_gain: float, out_gain: float, delays: str, decays: str):
    """Add echoing to the audio.
    :param float in_gain: set signal input gain
    :param float out_gain: set signal output gain
    :param str delays: set list of signal delays
    :param str decays: set list of signal decays"""
    filter = Filter(
        command="aecho",
        params=[
            FilterOption(name="in_gain", type=float),
            FilterOption(name="out_gain", type=float),
            FilterOption(name="delays", type=str),
            FilterOption(name="decays", type=str),
        ],
    )
    graph.append(filter)
    return graph


def scale(
    graph: Stream,
    w: str,
    h: str,
    flags: str,
    interl: bool,
    size: str,
    in_color_matrix: int,
    out_color_matrix: int,
    in_range: int,
    out_range: int,
    in_v_chr_pos: int,
    in_h_chr_pos: int,
    out_v_chr_pos: int,
    out_h_chr_pos: int,
    force_original_aspect_ratio: int,
    force_divisible_by: int,
    param0: float,
    param1: float,
    eval: int,
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
    filter = Filter(
        command="scale",
        params=[
            FilterOption(name="w", type=str),
            FilterOption(name="h", type=str),
            FilterOption(name="flags", type=str),
            FilterOption(name="interl", type=bool),
            FilterOption(name="size", type=str),
            FilterOption(name="in_color_matrix", type=int),
            FilterOption(name="out_color_matrix", type=int),
            FilterOption(name="in_range", type=int),
            FilterOption(name="out_range", type=int),
            FilterOption(name="in_v_chr_pos", type=int),
            FilterOption(name="in_h_chr_pos", type=int),
            FilterOption(name="out_v_chr_pos", type=int),
            FilterOption(name="out_h_chr_pos", type=int),
            FilterOption(name="force_original_aspect_ratio", type=int),
            FilterOption(name="force_divisible_by", type=int),
            FilterOption(name="param0", type=float),
            FilterOption(name="param1", type=float),
            FilterOption(name="eval", type=int),
        ],
    )
    graph.append(filter)
    return graph


def scale2ref(
    graph: Stream,
):
    """Scale the input video size and/or convert the image format to the given reference."""
    filter = Filter(command="scale2ref", params=[])
    graph.append(filter)
    return graph


def decimate(
    graph: Stream, cycle: int, dupthresh: float, scthresh: float, blockx: int, blocky: int, ppsrc: bool, chroma: bool, mixed: bool
):
    """Decimate frames (post field matching filter).
    :param int cycle: set the number of frame from which one will be dropped
    :param float dupthresh: set duplicate threshold
    :param float scthresh: set scene change threshold
    :param int blockx: set the size of the x-axis blocks used during metric calculations
    :param int blocky: set the size of the y-axis blocks used during metric calculations
    :param bool ppsrc: mark main input as a pre-processed input and activate clean source input stream
    :param bool chroma: set whether or not chroma is considered in the metric calculations
    :param bool mixed: set whether or not the input only partially contains content to be decimated"""
    filter = Filter(
        command="decimate",
        params=[
            FilterOption(name="cycle", type=int),
            FilterOption(name="dupthresh", type=float),
            FilterOption(name="scthresh", type=float),
            FilterOption(name="blockx", type=int),
            FilterOption(name="blocky", type=int),
            FilterOption(name="ppsrc", type=bool),
            FilterOption(name="chroma", type=bool),
            FilterOption(name="mixed", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def monochrome(graph: Stream, cb: float, cr: float, size: float, high: float):
    """Convert video to gray using custom color filter.
    :param float cb: set the chroma blue spot
    :param float cr: set the chroma red spot
    :param float size: set the color filter size
    :param float high: set the highlights strength"""
    filter = Filter(
        command="monochrome",
        params=[
            FilterOption(name="cb", type=float),
            FilterOption(name="cr", type=float),
            FilterOption(name="size", type=float),
            FilterOption(name="high", type=float),
        ],
    )
    graph.append(filter)
    return graph


def equalizer(
    graph: Stream,
    frequency: float,
    width_type: int,
    width: float,
    gain: float,
    mix: float,
    channels: str,
    normalize: bool,
    transform: int,
    precision: int,
    blocksize: int,
):
    """Apply two-pole peaking equalization (EQ) filter.
    :param float frequency: set central frequency
    :param int width_type: set filter-width type
    :param float width: set width
    :param float gain: set gain
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param int transform: set transform type
    :param int precision: set filtering precision
    :param int blocksize: set the block size"""
    filter = Filter(
        command="equalizer",
        params=[
            FilterOption(name="frequency", type=float),
            FilterOption(name="width_type", type=int),
            FilterOption(name="width", type=float),
            FilterOption(name="gain", type=float),
            FilterOption(name="mix", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="normalize", type=bool),
            FilterOption(name="transform", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="blocksize", type=int),
        ],
    )
    graph.append(filter)
    return graph


def bass(
    graph: Stream,
):
    """Boost or cut lower frequencies."""
    filter = Filter(command="bass", params=[])
    graph.append(filter)
    return graph


def lowshelf(
    graph: Stream,
):
    """Apply a low shelf filter."""
    filter = Filter(command="lowshelf", params=[])
    graph.append(filter)
    return graph


def treble(
    graph: Stream,
):
    """Boost or cut upper frequencies."""
    filter = Filter(command="treble", params=[])
    graph.append(filter)
    return graph


def highshelf(
    graph: Stream,
):
    """Apply a high shelf filter."""
    filter = Filter(command="highshelf", params=[])
    graph.append(filter)
    return graph


def tiltshelf(
    graph: Stream,
):
    """Apply a tilt shelf filter."""
    filter = Filter(command="tiltshelf", params=[])
    graph.append(filter)
    return graph


def bandpass(
    graph: Stream,
    frequency: float,
    width_type: int,
    width: float,
    csg: bool,
    mix: float,
    channels: str,
    normalize: bool,
    transform: int,
    precision: int,
    blocksize: int,
):
    """Apply a two-pole Butterworth band-pass filter.
    :param float frequency: set central frequency
    :param int width_type: set filter-width type
    :param float width: set width
    :param bool csg: use constant skirt gain
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param int transform: set transform type
    :param int precision: set filtering precision
    :param int blocksize: set the block size"""
    filter = Filter(
        command="bandpass",
        params=[
            FilterOption(name="frequency", type=float),
            FilterOption(name="width_type", type=int),
            FilterOption(name="width", type=float),
            FilterOption(name="csg", type=bool),
            FilterOption(name="mix", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="normalize", type=bool),
            FilterOption(name="transform", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="blocksize", type=int),
        ],
    )
    graph.append(filter)
    return graph


def bandreject(
    graph: Stream,
    frequency: float,
    width_type: int,
    width: float,
    mix: float,
    channels: str,
    normalize: bool,
    transform: int,
    precision: int,
    blocksize: int,
):
    """Apply a two-pole Butterworth band-reject filter.
    :param float frequency: set central frequency
    :param int width_type: set filter-width type
    :param float width: set width
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param int transform: set transform type
    :param int precision: set filtering precision
    :param int blocksize: set the block size"""
    filter = Filter(
        command="bandreject",
        params=[
            FilterOption(name="frequency", type=float),
            FilterOption(name="width_type", type=int),
            FilterOption(name="width", type=float),
            FilterOption(name="mix", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="normalize", type=bool),
            FilterOption(name="transform", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="blocksize", type=int),
        ],
    )
    graph.append(filter)
    return graph


def lowpass(
    graph: Stream,
    frequency: float,
    width_type: int,
    width: float,
    poles: int,
    mix: float,
    channels: str,
    normalize: bool,
    transform: int,
    precision: int,
    blocksize: int,
):
    """Apply a low-pass filter with 3dB point frequency.
    :param float frequency: set frequency
    :param int width_type: set filter-width type
    :param float width: set width
    :param int poles: set number of poles
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param int transform: set transform type
    :param int precision: set filtering precision
    :param int blocksize: set the block size"""
    filter = Filter(
        command="lowpass",
        params=[
            FilterOption(name="frequency", type=float),
            FilterOption(name="width_type", type=int),
            FilterOption(name="width", type=float),
            FilterOption(name="poles", type=int),
            FilterOption(name="mix", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="normalize", type=bool),
            FilterOption(name="transform", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="blocksize", type=int),
        ],
    )
    graph.append(filter)
    return graph


def highpass(
    graph: Stream,
    frequency: float,
    width_type: int,
    width: float,
    poles: int,
    mix: float,
    channels: str,
    normalize: bool,
    transform: int,
    precision: int,
    blocksize: int,
):
    """Apply a high-pass filter with 3dB point frequency.
    :param float frequency: set frequency
    :param int width_type: set filter-width type
    :param float width: set width
    :param int poles: set number of poles
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param int transform: set transform type
    :param int precision: set filtering precision
    :param int blocksize: set the block size"""
    filter = Filter(
        command="highpass",
        params=[
            FilterOption(name="frequency", type=float),
            FilterOption(name="width_type", type=int),
            FilterOption(name="width", type=float),
            FilterOption(name="poles", type=int),
            FilterOption(name="mix", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="normalize", type=bool),
            FilterOption(name="transform", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="blocksize", type=int),
        ],
    )
    graph.append(filter)
    return graph


def allpass(
    graph: Stream,
    frequency: float,
    width_type: int,
    width: float,
    mix: float,
    channels: str,
    normalize: bool,
    order: int,
    transform: int,
    precision: int,
):
    """Apply a two-pole all-pass filter.
    :param float frequency: set central frequency
    :param int width_type: set filter-width type
    :param float width: set width
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param int order: set filter order
    :param int transform: set transform type
    :param int precision: set filtering precision"""
    filter = Filter(
        command="allpass",
        params=[
            FilterOption(name="frequency", type=float),
            FilterOption(name="width_type", type=int),
            FilterOption(name="width", type=float),
            FilterOption(name="mix", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="normalize", type=bool),
            FilterOption(name="order", type=int),
            FilterOption(name="transform", type=int),
            FilterOption(name="precision", type=int),
        ],
    )
    graph.append(filter)
    return graph


def biquad(
    graph: Stream,
    a0: float,
    a1: float,
    a2: float,
    b0: float,
    b1: float,
    b2: float,
    mix: float,
    channels: str,
    normalize: bool,
    transform: int,
    precision: int,
    blocksize: int,
):
    """Apply a biquad IIR filter with the given coefficients.
    :param float mix: set mix
    :param str channels: set channels to filter
    :param bool normalize: normalize coefficients
    :param int transform: set transform type
    :param int precision: set filtering precision
    :param int blocksize: set the block size"""
    filter = Filter(
        command="biquad",
        params=[
            FilterOption(name="a0", type=float),
            FilterOption(name="a1", type=float),
            FilterOption(name="a2", type=float),
            FilterOption(name="b0", type=float),
            FilterOption(name="b1", type=float),
            FilterOption(name="b2", type=float),
            FilterOption(name="mix", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="normalize", type=bool),
            FilterOption(name="transform", type=int),
            FilterOption(name="precision", type=int),
            FilterOption(name="blocksize", type=int),
        ],
    )
    graph.append(filter)
    return graph


def spectrumsynth(graph: Stream, sample_rate: int, channels: int, scale: int, slide: int, win_func: int, overlap: float, orientation: int):
    """Convert input spectrum videos to audio output.
    :param int sample_rate: set sample rate
    :param int channels: set channels
    :param int scale: set input amplitude scale
    :param int slide: set input sliding mode
    :param int win_func: set window function
    :param float overlap: set window overlap
    :param int orientation: set orientation"""
    filter = Filter(
        command="spectrumsynth",
        params=[
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="channels", type=int),
            FilterOption(name="scale", type=int),
            FilterOption(name="slide", type=int),
            FilterOption(name="win_func", type=int),
            FilterOption(name="overlap", type=float),
            FilterOption(name="orientation", type=int),
        ],
    )
    graph.append(filter)
    return graph


def amix(graph: Stream, inputs: int, duration: int, dropout_transition: float, weights: str, normalize: bool):
    """Audio mixing.
    :param int inputs: Number of inputs.
    :param int duration: How to determine the end-of-stream.
    :param float dropout_transition: Transition time, in seconds, for volume renormalization when an input stream ends.
    :param str weights: Set weight for each input.
    :param bool normalize: Scale inputs"""
    filter = Filter(
        command="amix",
        params=[
            FilterOption(name="inputs", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="dropout_transition", type=float),
            FilterOption(name="weights", type=str),
            FilterOption(name="normalize", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def blackframe(graph: Stream, amount: int, threshold: int):
    """Detect frames that are (almost) black.
    :param int amount: percentage of the pixels that have to be below the threshold for the frame to be considered black
    :param int threshold: threshold below which a pixel value is considered black"""
    filter = Filter(command="blackframe", params=[FilterOption(name="amount", type=int), FilterOption(name="threshold", type=int)])
    graph.append(filter)
    return graph


def asdr(
    graph: Stream,
):
    """Measure Audio Signal-to-Distortion Ratio."""
    filter = Filter(command="asdr", params=[])
    graph.append(filter)
    return graph


def apsnr(
    graph: Stream,
):
    """Measure Audio Peak Signal-to-Noise Ratio."""
    filter = Filter(command="apsnr", params=[])
    graph.append(filter)
    return graph


def asisdr(
    graph: Stream,
):
    """Measure Audio Scale-Invariant Signal-to-Distortion Ratio."""
    filter = Filter(command="asisdr", params=[])
    graph.append(filter)
    return graph


def nlmeans_vulkan(
    graph: Stream, s: float, p: int, r: int, t: int, s1: float, s2: float, s3: float, s4: float, p1: int, p2: int, p3: int, p4: int
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
    :param int p4: patch size for component 4"""
    filter = Filter(
        command="nlmeans_vulkan",
        params=[
            FilterOption(name="s", type=float),
            FilterOption(name="p", type=int),
            FilterOption(name="r", type=int),
            FilterOption(name="t", type=int),
            FilterOption(name="s1", type=float),
            FilterOption(name="s2", type=float),
            FilterOption(name="s3", type=float),
            FilterOption(name="s4", type=float),
            FilterOption(name="p1", type=int),
            FilterOption(name="p2", type=int),
            FilterOption(name="p3", type=int),
            FilterOption(name="p4", type=int),
        ],
    )
    graph.append(filter)
    return graph


def chromakey(graph: Stream, color: str, similarity: float, blend: float, yuv: bool):
    """Turns a certain color into transparency. Operates on YUV colors.
    :param str color: set the chromakey key color
    :param float similarity: set the chromakey similarity value
    :param float blend: set the chromakey key blend value
    :param bool yuv: color parameter is in yuv instead of rgb"""
    filter = Filter(
        command="chromakey",
        params=[
            FilterOption(name="color", type=str),
            FilterOption(name="similarity", type=float),
            FilterOption(name="blend", type=float),
            FilterOption(name="yuv", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def chromahold(graph: Stream, color: str, similarity: float, blend: float, yuv: bool):
    """Turns a certain color range into gray.
    :param str color: set the chromahold key color
    :param float similarity: set the chromahold similarity value
    :param float blend: set the chromahold blend value
    :param bool yuv: color parameter is in yuv instead of rgb"""
    filter = Filter(
        command="chromahold",
        params=[
            FilterOption(name="color", type=str),
            FilterOption(name="similarity", type=float),
            FilterOption(name="blend", type=float),
            FilterOption(name="yuv", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def dynaudnorm(
    graph: Stream,
    framelen: int,
    gausssize: int,
    peak: float,
    maxgain: float,
    targetrms: float,
    coupling: bool,
    correctdc: bool,
    altboundary: bool,
    compress: float,
    threshold: float,
    channels: str,
    overlap: float,
    curve: str,
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
    :param str curve: set the custom peak mapping curve"""
    filter = Filter(
        command="dynaudnorm",
        params=[
            FilterOption(name="framelen", type=int),
            FilterOption(name="gausssize", type=int),
            FilterOption(name="peak", type=float),
            FilterOption(name="maxgain", type=float),
            FilterOption(name="targetrms", type=float),
            FilterOption(name="coupling", type=bool),
            FilterOption(name="correctdc", type=bool),
            FilterOption(name="altboundary", type=bool),
            FilterOption(name="compress", type=float),
            FilterOption(name="threshold", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="overlap", type=float),
            FilterOption(name="curve", type=str),
        ],
    )
    graph.append(filter)
    return graph


def cellauto(
    graph: Stream,
    filename: str,
    pattern: str,
    rate: str,
    size: int,
    rule: int,
    random_fill_ratio: float,
    random_seed: int,
    scroll: bool,
    start_full: bool,
    stitch: bool,
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
    :param bool stitch: stitch boundaries"""
    filter = Filter(
        command="cellauto",
        params=[
            FilterOption(name="filename", type=str),
            FilterOption(name="pattern", type=str),
            FilterOption(name="rate", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="rule", type=int),
            FilterOption(name="random_fill_ratio", type=float),
            FilterOption(name="random_seed", type=int),
            FilterOption(name="scroll", type=bool),
            FilterOption(name="start_full", type=bool),
            FilterOption(name="stitch", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def showcwt(
    graph: Stream,
    size: int,
    rate: str,
    scale: int,
    iscale: int,
    min: float,
    max: float,
    imin: float,
    imax: float,
    logb: float,
    deviation: float,
    pps: int,
    mode: int,
    slide: int,
    direction: int,
    bar: float,
    rotation: float,
):
    """Convert input audio to a CWT (Continuous Wavelet Transform) spectrum video output.
    :param int size: set video size
    :param str rate: set video rate
    :param int scale: set frequency scale
    :param int iscale: set intensity scale
    :param float min: set minimum frequency
    :param float max: set maximum frequency
    :param float imin: set minimum intensity
    :param float imax: set maximum intensity
    :param float logb: set logarithmic basis
    :param float deviation: set frequency deviation
    :param int pps: set pixels per second
    :param int mode: set output mode
    :param int slide: set slide mode
    :param int direction: set direction mode
    :param float bar: set bargraph ratio
    :param float rotation: set color rotation"""
    filter = Filter(
        command="showcwt",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="scale", type=int),
            FilterOption(name="iscale", type=int),
            FilterOption(name="min", type=float),
            FilterOption(name="max", type=float),
            FilterOption(name="imin", type=float),
            FilterOption(name="imax", type=float),
            FilterOption(name="logb", type=float),
            FilterOption(name="deviation", type=float),
            FilterOption(name="pps", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="slide", type=int),
            FilterOption(name="direction", type=int),
            FilterOption(name="bar", type=float),
            FilterOption(name="rotation", type=float),
        ],
    )
    graph.append(filter)
    return graph


def atempo(graph: Stream, tempo: float):
    """Adjust audio tempo.
    :param float tempo: set tempo scale factor"""
    filter = Filter(command="atempo", params=[FilterOption(name="tempo", type=float)])
    graph.append(filter)
    return graph


def color_vulkan(graph: Stream, color: str, size: int, rate: str, duration: int, sar: int, format: str, out_range: int):
    """Generate a constant color (Vulkan)
    :param str color: set color
    :param int size: set video size
    :param str rate: set video rate
    :param int duration: set video duration
    :param int sar: set video sample aspect ratio
    :param str format: Output video format (software format of hardware frames)
    :param int out_range: Output colour range (from 0 to 2) (default 0)"""
    filter = Filter(
        command="color_vulkan",
        params=[
            FilterOption(name="color", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="sar", type=int),
            FilterOption(name="format", type=str),
            FilterOption(name="out_range", type=int),
        ],
    )
    graph.append(filter)
    return graph


def midequalizer(graph: Stream, planes: int):
    """Apply Midway Equalization.
    :param int planes: set planes"""
    filter = Filter(command="midequalizer", params=[FilterOption(name="planes", type=int)])
    graph.append(filter)
    return graph


def blend_vulkan(
    graph: Stream,
    c0_mode: int,
    c1_mode: int,
    c2_mode: int,
    c3_mode: int,
    all_mode: int,
    c0_opacity: float,
    c1_opacity: float,
    c2_opacity: float,
    c3_opacity: float,
    all_opacity: float,
):
    """Blend two video frames in Vulkan
    :param int c0_mode: set component #0 blend mode
    :param int c1_mode: set component #1 blend mode
    :param int c2_mode: set component #2 blend mode
    :param int c3_mode: set component #3 blend mode
    :param int all_mode: set blend mode for all components
    :param float c0_opacity: set color component #0 opacity
    :param float c1_opacity: set color component #1 opacity
    :param float c2_opacity: set color component #2 opacity
    :param float c3_opacity: set color component #3 opacity
    :param float all_opacity: set opacity for all color components"""
    filter = Filter(
        command="blend_vulkan",
        params=[
            FilterOption(name="c0_mode", type=int),
            FilterOption(name="c1_mode", type=int),
            FilterOption(name="c2_mode", type=int),
            FilterOption(name="c3_mode", type=int),
            FilterOption(name="all_mode", type=int),
            FilterOption(name="c0_opacity", type=float),
            FilterOption(name="c1_opacity", type=float),
            FilterOption(name="c2_opacity", type=float),
            FilterOption(name="c3_opacity", type=float),
            FilterOption(name="all_opacity", type=float),
        ],
    )
    graph.append(filter)
    return graph


def a3dscope(
    graph: Stream,
    rate: str,
    size: int,
    fov: float,
    roll: float,
    pitch: float,
    yaw: float,
    xzoom: float,
    yzoom: float,
    zzoom: float,
    xpos: float,
    ypos: float,
    zpos: float,
    length: int,
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
    :param int length: set length"""
    filter = Filter(
        command="a3dscope",
        params=[
            FilterOption(name="rate", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="fov", type=float),
            FilterOption(name="roll", type=float),
            FilterOption(name="pitch", type=float),
            FilterOption(name="yaw", type=float),
            FilterOption(name="xzoom", type=float),
            FilterOption(name="yzoom", type=float),
            FilterOption(name="zzoom", type=float),
            FilterOption(name="xpos", type=float),
            FilterOption(name="ypos", type=float),
            FilterOption(name="zpos", type=float),
            FilterOption(name="length", type=int),
        ],
    )
    graph.append(filter)
    return graph


def thumbnail(graph: Stream, n: int, log: int):
    """Select the most representative frame in a given sequence of consecutive frames.
    :param int n: set the frames batch size
    :param int log: force stats logging level"""
    filter = Filter(command="thumbnail", params=[FilterOption(name="n", type=int), FilterOption(name="log", type=int)])
    graph.append(filter)
    return graph


def deblock(graph: Stream, filter: int, block: int, alpha: float, beta: float, gamma: float, delta: float, planes: int):
    """Deblock video.
    :param int filter: set type of filter
    :param int block: set size of block
    :param float alpha: set 1st detection threshold
    :param float beta: set 2nd detection threshold
    :param float gamma: set 3rd detection threshold
    :param float delta: set 4th detection threshold
    :param int planes: set planes to filter"""
    filter = Filter(
        command="deblock",
        params=[
            FilterOption(name="filter", type=int),
            FilterOption(name="block", type=int),
            FilterOption(name="alpha", type=float),
            FilterOption(name="beta", type=float),
            FilterOption(name="gamma", type=float),
            FilterOption(name="delta", type=float),
            FilterOption(name="planes", type=int),
        ],
    )
    graph.append(filter)
    return graph


def crossfeed(graph: Stream, strength: float, range: float, slope: float, level_in: float, level_out: float, block_size: int):
    """Apply headphone crossfeed filter.
    :param float strength: set crossfeed strength
    :param float range: set soundstage wideness
    :param float slope: set curve slope
    :param float level_in: set level in
    :param float level_out: set level out
    :param int block_size: set the block size"""
    filter = Filter(
        command="crossfeed",
        params=[
            FilterOption(name="strength", type=float),
            FilterOption(name="range", type=float),
            FilterOption(name="slope", type=float),
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="block_size", type=int),
        ],
    )
    graph.append(filter)
    return graph


def deflicker(graph: Stream, size: int, mode: int, bypass: bool):
    """Remove temporal frame luminance variations.
    :param int size: set how many frames to use
    :param int mode: set how to smooth luminance
    :param bool bypass: leave frames unchanged"""
    filter = Filter(
        command="deflicker",
        params=[FilterOption(name="size", type=int), FilterOption(name="mode", type=int), FilterOption(name="bypass", type=bool)],
    )
    graph.append(filter)
    return graph


def showwaves(graph: Stream, size: int, mode: int, n: int, rate: str, split_channels: bool, colors: str, scale: int, draw: int):
    """Convert input audio to a video output.
    :param int size: set video size
    :param int mode: select display mode
    :param int n: set how many samples to show in the same point
    :param str rate: set video rate
    :param bool split_channels: draw channels separately
    :param str colors: set channels colors
    :param int scale: set amplitude scale
    :param int draw: set draw mode"""
    filter = Filter(
        command="showwaves",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="n", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="split_channels", type=bool),
            FilterOption(name="colors", type=str),
            FilterOption(name="scale", type=int),
            FilterOption(name="draw", type=int),
        ],
    )
    graph.append(filter)
    return graph


def showwavespic(graph: Stream, size: int, split_channels: bool, colors: str, scale: int, draw: int, filter: int):
    """Convert input audio to a video output single picture.
    :param int size: set video size
    :param bool split_channels: draw channels separately
    :param str colors: set channels colors
    :param int scale: set amplitude scale
    :param int draw: set draw mode
    :param int filter: set filter mode"""
    filter = Filter(
        command="showwavespic",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="split_channels", type=bool),
            FilterOption(name="colors", type=str),
            FilterOption(name="scale", type=int),
            FilterOption(name="draw", type=int),
            FilterOption(name="filter", type=int),
        ],
    )
    graph.append(filter)
    return graph


def aemphasis(graph: Stream, level_in: float, level_out: float, mode: int, type: int):
    """Audio emphasis.
    :param float level_in: set input gain
    :param float level_out: set output gain
    :param int mode: set filter mode
    :param int type: set filter type"""
    filter = Filter(
        command="aemphasis",
        params=[
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="mode", type=int),
            FilterOption(name="type", type=int),
        ],
    )
    graph.append(filter)
    return graph


def tile(graph: Stream, layout: int, nb_frames: int, margin: int, padding: int, color: str, overlap: int, init_padding: int):
    """Tile several successive frames together.
    :param int layout: set grid size
    :param int nb_frames: set maximum number of frame to render
    :param int margin: set outer border margin in pixels
    :param int padding: set inner border thickness in pixels
    :param str color: set the color of the unused area
    :param int overlap: set how many frames to overlap for each render
    :param int init_padding: set how many frames to initially pad"""
    filter = Filter(
        command="tile",
        params=[
            FilterOption(name="layout", type=int),
            FilterOption(name="nb_frames", type=int),
            FilterOption(name="margin", type=int),
            FilterOption(name="padding", type=int),
            FilterOption(name="color", type=str),
            FilterOption(name="overlap", type=int),
            FilterOption(name="init_padding", type=int),
        ],
    )
    graph.append(filter)
    return graph


def colorkey_opencl(graph: Stream, color: str, similarity: float, blend: float):
    """Turns a certain color into transparency. Operates on RGB colors.
    :param str color: set the colorkey key color
    :param float similarity: set the colorkey similarity value
    :param float blend: set the colorkey key blend value"""
    filter = Filter(
        command="colorkey_opencl",
        params=[FilterOption(name="color", type=str), FilterOption(name="similarity", type=float), FilterOption(name="blend", type=float)],
    )
    graph.append(filter)
    return graph


def fillborders(graph: Stream, left: int, right: int, top: int, bottom: int, mode: int, color: str):
    """Fill borders of the input video.
    :param int left: set the left fill border
    :param int right: set the right fill border
    :param int top: set the top fill border
    :param int bottom: set the bottom fill border
    :param int mode: set the fill borders mode
    :param str color: set the color for the fixed/fade mode"""
    filter = Filter(
        command="fillborders",
        params=[
            FilterOption(name="left", type=int),
            FilterOption(name="right", type=int),
            FilterOption(name="top", type=int),
            FilterOption(name="bottom", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="color", type=str),
        ],
    )
    graph.append(filter)
    return graph


def maskedmerge(graph: Stream, planes: int):
    """Merge first stream with second stream using third stream as mask.
    :param int planes: set planes"""
    filter = Filter(command="maskedmerge", params=[FilterOption(name="planes", type=int)])
    graph.append(filter)
    return graph


def bwdif(graph: Stream, mode: int, parity: int, deint: int):
    """Deinterlace the input image.
    :param int mode: specify the interlacing mode
    :param int parity: specify the assumed picture field parity
    :param int deint: specify which frames to deinterlace"""
    filter = Filter(
        command="bwdif",
        params=[FilterOption(name="mode", type=int), FilterOption(name="parity", type=int), FilterOption(name="deint", type=int)],
    )
    graph.append(filter)
    return graph


def movie(
    graph: Stream,
    filename: str,
    format_name: str,
    stream_index: int,
    seek_point: float,
    streams: str,
    loop: int,
    discontinuity: int,
    dec_threads: int,
    format_opts: str,
):
    """Read from a movie source.
    :param str format_name: set format name
    :param int stream_index: set stream index
    :param float seek_point: set seekpoint (seconds)
    :param str streams: set streams
    :param int loop: set loop count
    :param int discontinuity: set discontinuity threshold
    :param int dec_threads: set the number of threads for decoding
    :param str format_opts: set format options for the opened file"""
    filter = Filter(
        command="movie",
        params=[
            FilterOption(name="filename", type=str),
            FilterOption(name="format_name", type=str),
            FilterOption(name="stream_index", type=int),
            FilterOption(name="seek_point", type=float),
            FilterOption(name="streams", type=str),
            FilterOption(name="loop", type=int),
            FilterOption(name="discontinuity", type=int),
            FilterOption(name="dec_threads", type=int),
            FilterOption(name="format_opts", type=str),
        ],
    )
    graph.append(filter)
    return graph


def amovie(
    graph: Stream,
):
    """Read audio from a movie source."""
    filter = Filter(command="amovie", params=[])
    graph.append(filter)
    return graph


def avsynctest(
    graph: Stream,
    size: int,
    framerate: str,
    samplerate: int,
    amplitude: float,
    period: int,
    delay: int,
    cycle: bool,
    duration: int,
    fg: str,
    bg: str,
    ag: str,
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
    :param str ag: set additional color"""
    filter = Filter(
        command="avsynctest",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="framerate", type=str),
            FilterOption(name="samplerate", type=int),
            FilterOption(name="amplitude", type=float),
            FilterOption(name="period", type=int),
            FilterOption(name="delay", type=int),
            FilterOption(name="cycle", type=bool),
            FilterOption(name="duration", type=int),
            FilterOption(name="fg", type=str),
            FilterOption(name="bg", type=str),
            FilterOption(name="ag", type=str),
        ],
    )
    graph.append(filter)
    return graph


def dnn_classify(
    graph: Stream,
    dnn_backend: int,
    model: str,
    input: str,
    output: str,
    backend_configs: str,
    options: str,
    _async: bool,
    confidence: float,
    labels: str,
    target: str,
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
    :param str target: which one to be classified"""
    filter = Filter(
        command="dnn_classify",
        params=[
            FilterOption(name="dnn_backend", type=int),
            FilterOption(name="model", type=str),
            FilterOption(name="input", type=str),
            FilterOption(name="output", type=str),
            FilterOption(name="backend_configs", type=str),
            FilterOption(name="options", type=str),
            FilterOption(name="async", type=bool),
            FilterOption(name="confidence", type=float),
            FilterOption(name="labels", type=str),
            FilterOption(name="target", type=str),
        ],
    )
    graph.append(filter)
    return graph


def blurdetect(graph: Stream, high: float, low: float, radius: int, block_pct: int, block_width: int, block_height: int, planes: int):
    """Blurdetect filter.
    :param float high: set high threshold
    :param float low: set low threshold
    :param int radius: search radius for maxima detection
    :param int block_pct: block pooling threshold when calculating blurriness
    :param int block_width: block size for block-based abbreviation of blurriness
    :param int block_height: block size for block-based abbreviation of blurriness
    :param int planes: set planes to filter"""
    filter = Filter(
        command="blurdetect",
        params=[
            FilterOption(name="high", type=float),
            FilterOption(name="low", type=float),
            FilterOption(name="radius", type=int),
            FilterOption(name="block_pct", type=int),
            FilterOption(name="block_width", type=int),
            FilterOption(name="block_height", type=int),
            FilterOption(name="planes", type=int),
        ],
    )
    graph.append(filter)
    return graph


def grayworld(
    graph: Stream,
):
    """Adjust white balance using LAB gray world algorithm"""
    filter = Filter(command="grayworld", params=[])
    graph.append(filter)
    return graph


def cover_rect(graph: Stream, cover: str, mode: int):
    """Find and cover a user specified object.
    :param str cover: cover bitmap filename
    :param int mode: set removal mode"""
    filter = Filter(command="cover_rect", params=[FilterOption(name="cover", type=str), FilterOption(name="mode", type=int)])
    graph.append(filter)
    return graph


def channelmap(graph: Stream, map: str, channel_layout: str):
    """Remap audio channels.
    :param str map: A comma-separated list of input channel numbers in output order.
    :param str channel_layout: Output channel layout."""
    filter = Filter(command="channelmap", params=[FilterOption(name="map", type=str), FilterOption(name="channel_layout", type=str)])
    graph.append(filter)
    return graph


def dedot(graph: Stream, m: str, lt: float, tl: float, tc: float, ct: float):
    """Reduce cross-luminance and cross-color.
    :param str m: set filtering mode
    :param float lt: set spatial luma threshold
    :param float tl: set tolerance for temporal luma
    :param float tc: set tolerance for chroma temporal variation
    :param float ct: set temporal chroma threshold"""
    filter = Filter(
        command="dedot",
        params=[
            FilterOption(name="m", type=str),
            FilterOption(name="lt", type=float),
            FilterOption(name="tl", type=float),
            FilterOption(name="tc", type=float),
            FilterOption(name="ct", type=float),
        ],
    )
    graph.append(filter)
    return graph


def adeclick(graph: Stream, window: float, overlap: float, arorder: float, threshold: float, burst: float, method: int):
    """Remove impulsive noise from input audio.
    :param float window: set window size
    :param float overlap: set window overlap
    :param float arorder: set autoregression order
    :param float threshold: set threshold
    :param float burst: set burst fusion
    :param int method: set overlap method"""
    filter = Filter(
        command="adeclick",
        params=[
            FilterOption(name="window", type=float),
            FilterOption(name="overlap", type=float),
            FilterOption(name="arorder", type=float),
            FilterOption(name="threshold", type=float),
            FilterOption(name="burst", type=float),
            FilterOption(name="method", type=int),
        ],
    )
    graph.append(filter)
    return graph


def adeclip(graph: Stream, window: float, overlap: float, arorder: float, threshold: float, hsize: int, method: int):
    """Remove clipping from input audio.
    :param float window: set window size
    :param float overlap: set window overlap
    :param float arorder: set autoregression order
    :param float threshold: set threshold
    :param int hsize: set histogram size
    :param int method: set overlap method"""
    filter = Filter(
        command="adeclip",
        params=[
            FilterOption(name="window", type=float),
            FilterOption(name="overlap", type=float),
            FilterOption(name="arorder", type=float),
            FilterOption(name="threshold", type=float),
            FilterOption(name="hsize", type=int),
            FilterOption(name="method", type=int),
        ],
    )
    graph.append(filter)
    return graph


def settb(graph: Stream, expr: str):
    """Set timebase for the video output link.
    :param str expr: set expression determining the output timebase"""
    filter = Filter(command="settb", params=[FilterOption(name="expr", type=str)])
    graph.append(filter)
    return graph


def asettb(graph: Stream, expr: str):
    """Set timebase for the audio output link.
    :param str expr: set expression determining the output timebase"""
    filter = Filter(command="asettb", params=[FilterOption(name="expr", type=str)])
    graph.append(filter)
    return graph


def xfade_opencl(graph: Stream, transition: int, source: str, kernel: str, duration: int, offset: int):
    """Cross fade one video with another video.
    :param int transition: set cross fade transition
    :param str source: set OpenCL program source file for custom transition
    :param str kernel: set kernel name in program file for custom transition
    :param int duration: set cross fade duration
    :param int offset: set cross fade start relative to first input stream"""
    filter = Filter(
        command="xfade_opencl",
        params=[
            FilterOption(name="transition", type=int),
            FilterOption(name="source", type=str),
            FilterOption(name="kernel", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="offset", type=int),
        ],
    )
    graph.append(filter)
    return graph


def idet(graph: Stream, intl_thres: float, prog_thres: float, rep_thres: float, half_life: float, analyze_interlaced_flag: int):
    """Interlace detect Filter.
    :param float intl_thres: set interlacing threshold
    :param float prog_thres: set progressive threshold
    :param float rep_thres: set repeat threshold
    :param float half_life: half life of cumulative statistics
    :param int analyze_interlaced_flag: set number of frames to use to determine if the interlace flag is accurate"""
    filter = Filter(
        command="idet",
        params=[
            FilterOption(name="intl_thres", type=float),
            FilterOption(name="prog_thres", type=float),
            FilterOption(name="rep_thres", type=float),
            FilterOption(name="half_life", type=float),
            FilterOption(name="analyze_interlaced_flag", type=int),
        ],
    )
    graph.append(filter)
    return graph


def weave(graph: Stream, first_field: int):
    """Weave input video fields into frames.
    :param int first_field: set first field"""
    filter = Filter(command="weave", params=[FilterOption(name="first_field", type=int)])
    graph.append(filter)
    return graph


def doubleweave(
    graph: Stream,
):
    """Weave input video fields into double number of frames."""
    filter = Filter(command="doubleweave", params=[])
    graph.append(filter)
    return graph


def pad(graph: Stream, width: str, height: str, x: str, y: str, color: str, eval: int, aspect: int):
    """Pad the input video.
    :param str width: set the pad area width expression
    :param str height: set the pad area height expression
    :param str x: set the x offset expression for the input image position
    :param str y: set the y offset expression for the input image position
    :param str color: set the color of the padded area border
    :param int eval: specify when to evaluate expressions
    :param int aspect: pad to fit an aspect instead of a resolution"""
    filter = Filter(
        command="pad",
        params=[
            FilterOption(name="width", type=str),
            FilterOption(name="height", type=str),
            FilterOption(name="x", type=str),
            FilterOption(name="y", type=str),
            FilterOption(name="color", type=str),
            FilterOption(name="eval", type=int),
            FilterOption(name="aspect", type=int),
        ],
    )
    graph.append(filter)
    return graph


def amultiply(
    graph: Stream,
):
    """Multiply two audio streams."""
    filter = Filter(command="amultiply", params=[])
    graph.append(filter)
    return graph


def compand(graph: Stream, attacks: str, decays: str, points: str, softknee: float, gain: float, volume: float, delay: float):
    """Compress or expand audio dynamic range.
    :param str attacks: set time over which increase of volume is determined
    :param str decays: set time over which decrease of volume is determined
    :param str points: set points of transfer function
    :param float soft-knee: set soft-knee
    :param float gain: set output gain
    :param float volume: set initial volume
    :param float delay: set delay for samples before sending them to volume adjuster"""
    filter = Filter(
        command="compand",
        params=[
            FilterOption(name="attacks", type=str),
            FilterOption(name="decays", type=str),
            FilterOption(name="points", type=str),
            FilterOption(name="soft-knee", type=float),
            FilterOption(name="gain", type=float),
            FilterOption(name="volume", type=float),
            FilterOption(name="delay", type=float),
        ],
    )
    graph.append(filter)
    return graph


def life(
    graph: Stream,
    filename: str,
    size: int,
    rate: str,
    rule: str,
    random_fill_ratio: float,
    random_seed: int,
    stitch: bool,
    mold: int,
    life_color: str,
    death_color: str,
    mold_color: str,
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
    :param str mold_color: set mold color"""
    filter = Filter(
        command="life",
        params=[
            FilterOption(name="filename", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="rule", type=str),
            FilterOption(name="random_fill_ratio", type=float),
            FilterOption(name="random_seed", type=int),
            FilterOption(name="stitch", type=bool),
            FilterOption(name="mold", type=int),
            FilterOption(name="life_color", type=str),
            FilterOption(name="death_color", type=str),
            FilterOption(name="mold_color", type=str),
        ],
    )
    graph.append(filter)
    return graph


def speechnorm(
    graph: Stream,
    peak: float,
    expansion: float,
    compression: float,
    threshold: float,
    _raise: float,
    fall: float,
    channels: str,
    invert: bool,
    link: bool,
    rms: float,
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
    :param float rms: set the RMS value"""
    filter = Filter(
        command="speechnorm",
        params=[
            FilterOption(name="peak", type=float),
            FilterOption(name="expansion", type=float),
            FilterOption(name="compression", type=float),
            FilterOption(name="threshold", type=float),
            FilterOption(name="raise", type=float),
            FilterOption(name="fall", type=float),
            FilterOption(name="channels", type=str),
            FilterOption(name="invert", type=bool),
            FilterOption(name="link", type=bool),
            FilterOption(name="rms", type=float),
        ],
    )
    graph.append(filter)
    return graph


def amerge(graph: Stream, inputs: int):
    """Merge two or more audio streams into a single multi-channel stream.
    :param int inputs: specify the number of inputs"""
    filter = Filter(command="amerge", params=[FilterOption(name="inputs", type=int)])
    graph.append(filter)
    return graph


def xmedian(graph: Stream, inputs: int, planes: int, percentile: float):
    """Pick median pixels from several video inputs.
    :param int inputs: set number of inputs
    :param int planes: set planes to filter
    :param float percentile: set percentile"""
    filter = Filter(
        command="xmedian",
        params=[FilterOption(name="inputs", type=int), FilterOption(name="planes", type=int), FilterOption(name="percentile", type=float)],
    )
    graph.append(filter)
    return graph


def tmedian(graph: Stream, radius: int, planes: int, percentile: float):
    """Pick median pixels from successive frames.
    :param int radius: set median filter radius
    :param int planes: set planes to filter
    :param float percentile: set percentile"""
    filter = Filter(
        command="tmedian",
        params=[FilterOption(name="radius", type=int), FilterOption(name="planes", type=int), FilterOption(name="percentile", type=float)],
    )
    graph.append(filter)
    return graph


def detelecine(graph: Stream, first_field: int, pattern: str, start_frame: int):
    """Apply an inverse telecine pattern.
    :param int first_field: select first field
    :param str pattern: pattern that describe for how many fields a frame is to be displayed
    :param int start_frame: position of first frame with respect to the pattern if stream is cut"""
    filter = Filter(
        command="detelecine",
        params=[
            FilterOption(name="first_field", type=int),
            FilterOption(name="pattern", type=str),
            FilterOption(name="start_frame", type=int),
        ],
    )
    graph.append(filter)
    return graph


def framepack(graph: Stream, format: int):
    """Generate a frame packed stereoscopic video.
    :param int format: Frame pack output format"""
    filter = Filter(command="framepack", params=[FilterOption(name="format", type=int)])
    graph.append(filter)
    return graph


def asetrate(graph: Stream, sample_rate: int):
    """Change the sample rate without altering the data.
    :param int sample_rate: set the sample rate"""
    filter = Filter(command="asetrate", params=[FilterOption(name="sample_rate", type=int)])
    graph.append(filter)
    return graph


def stereowiden(graph: Stream, delay: float, feedback: float, crossfeed: float, drymix: float):
    """Apply stereo widening effect.
    :param float delay: set delay time
    :param float feedback: set feedback gain
    :param float crossfeed: set cross feed
    :param float drymix: set dry-mix"""
    filter = Filter(
        command="stereowiden",
        params=[
            FilterOption(name="delay", type=float),
            FilterOption(name="feedback", type=float),
            FilterOption(name="crossfeed", type=float),
            FilterOption(name="drymix", type=float),
        ],
    )
    graph.append(filter)
    return graph


def shuffleplanes(graph: Stream, map0: int, map1: int, map2: int, map3: int):
    """Shuffle video planes.
    :param int map0: Index of the input plane to be used as the first output plane
    :param int map1: Index of the input plane to be used as the second output plane
    :param int map2: Index of the input plane to be used as the third output plane
    :param int map3: Index of the input plane to be used as the fourth output plane"""
    filter = Filter(
        command="shuffleplanes",
        params=[
            FilterOption(name="map0", type=int),
            FilterOption(name="map1", type=int),
            FilterOption(name="map2", type=int),
            FilterOption(name="map3", type=int),
        ],
    )
    graph.append(filter)
    return graph


def split(
    graph: Stream,
):
    """Pass on the input to N video outputs."""
    filter = Filter(command="split", params=[])
    graph.append(filter)
    return graph


def asplit(
    graph: Stream,
):
    """Pass on the audio input to N audio outputs."""
    filter = Filter(command="asplit", params=[])
    graph.append(filter)
    return graph


def mptestsrc(graph: Stream, rate: str, duration: int, test: int, max_frames: int):
    """Generate various test pattern.
    :param str rate: set video rate
    :param int duration: set video duration
    :param int test: set test to perform
    :param int max_frames: Set the maximum number of frames generated for each test"""
    filter = Filter(
        command="mptestsrc",
        params=[
            FilterOption(name="rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="test", type=int),
            FilterOption(name="max_frames", type=int),
        ],
    )
    graph.append(filter)
    return graph


def xbr(graph: Stream, n: int):
    """Scale the input using xBR algorithm.
    :param int n: set scale factor"""
    filter = Filter(command="xbr", params=[FilterOption(name="n", type=int)])
    graph.append(filter)
    return graph


def anull(
    graph: Stream,
):
    """Pass the source unchanged to the output."""
    filter = Filter(command="anull", params=[])
    graph.append(filter)
    return graph


def hstack(
    graph: Stream,
):
    """Stack video inputs horizontally."""
    filter = Filter(command="hstack", params=[])
    graph.append(filter)
    return graph


def vstack(
    graph: Stream,
):
    """Stack video inputs vertically."""
    filter = Filter(command="vstack", params=[])
    graph.append(filter)
    return graph


def xstack(graph: Stream, inputs: int, layout: str, grid: int, shortest: bool, fill: str):
    """Stack video inputs into custom layout.
    :param int inputs: set number of inputs
    :param str layout: set custom layout
    :param int grid: set fixed size grid layout
    :param bool shortest: force termination when the shortest input terminates
    :param str fill: set the color for unused pixels"""
    filter = Filter(
        command="xstack",
        params=[
            FilterOption(name="inputs", type=int),
            FilterOption(name="layout", type=str),
            FilterOption(name="grid", type=int),
            FilterOption(name="shortest", type=bool),
            FilterOption(name="fill", type=str),
        ],
    )
    graph.append(filter)
    return graph


def scale_vulkan(graph: Stream, w: str, h: str, scaler: int, format: str, out_range: int):
    """Scale Vulkan frames
    :param str w: Output video width
    :param str h: Output video height
    :param int scaler: Scaler function
    :param str format: Output video format (software format of hardware frames)
    :param int out_range: Output colour range (from 0 to 2) (default 0)"""
    filter = Filter(
        command="scale_vulkan",
        params=[
            FilterOption(name="w", type=str),
            FilterOption(name="h", type=str),
            FilterOption(name="scaler", type=int),
            FilterOption(name="format", type=str),
            FilterOption(name="out_range", type=int),
        ],
    )
    graph.append(filter)
    return graph


def uspp(graph: Stream, quality: int, qp: int, use_bframe_qp: bool, codec: str):
    """Apply Ultra Simple / Slow Post-processing filter.
    :param int quality: set quality
    :param int qp: force a constant quantizer parameter
    :param bool use_bframe_qp: use B-frames' QP
    :param str codec: Codec name"""
    filter = Filter(
        command="uspp",
        params=[
            FilterOption(name="quality", type=int),
            FilterOption(name="qp", type=int),
            FilterOption(name="use_bframe_qp", type=bool),
            FilterOption(name="codec", type=str),
        ],
    )
    graph.append(filter)
    return graph


def acrusher(
    graph: Stream,
    level_in: float,
    level_out: float,
    bits: float,
    mix: float,
    mode: int,
    dc: float,
    aa: float,
    samples: float,
    lfo: bool,
    lforange: float,
    lforate: float,
):
    """Reduce audio bit resolution.
    :param float level_in: set level in
    :param float level_out: set level out
    :param float bits: set bit reduction
    :param float mix: set mix
    :param int mode: set mode
    :param float dc: set DC
    :param float aa: set anti-aliasing
    :param float samples: set sample reduction
    :param bool lfo: enable LFO
    :param float lforange: set LFO depth
    :param float lforate: set LFO rate"""
    filter = Filter(
        command="acrusher",
        params=[
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="bits", type=float),
            FilterOption(name="mix", type=float),
            FilterOption(name="mode", type=int),
            FilterOption(name="dc", type=float),
            FilterOption(name="aa", type=float),
            FilterOption(name="samples", type=float),
            FilterOption(name="lfo", type=bool),
            FilterOption(name="lforange", type=float),
            FilterOption(name="lforate", type=float),
        ],
    )
    graph.append(filter)
    return graph


def showcqt(
    graph: Stream,
    size: int,
    fps: str,
    bar_h: int,
    axis_h: int,
    sono_h: int,
    fullhd: bool,
    sono_v: str,
    bar_v: str,
    sono_g: float,
    bar_g: float,
    bar_t: float,
    timeclamp: float,
    attack: float,
    basefreq: float,
    endfreq: float,
    coeffclamp: float,
    tlength: str,
    count: int,
    fcount: int,
    fontfile: str,
    font: str,
    fontcolor: str,
    axisfile: str,
    axis: bool,
    csp: int,
    cscheme: str,
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
    :param int csp: set color space
    :param str cscheme: set color scheme"""
    filter = Filter(
        command="showcqt",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="fps", type=str),
            FilterOption(name="bar_h", type=int),
            FilterOption(name="axis_h", type=int),
            FilterOption(name="sono_h", type=int),
            FilterOption(name="fullhd", type=bool),
            FilterOption(name="sono_v", type=str),
            FilterOption(name="bar_v", type=str),
            FilterOption(name="sono_g", type=float),
            FilterOption(name="bar_g", type=float),
            FilterOption(name="bar_t", type=float),
            FilterOption(name="timeclamp", type=float),
            FilterOption(name="attack", type=float),
            FilterOption(name="basefreq", type=float),
            FilterOption(name="endfreq", type=float),
            FilterOption(name="coeffclamp", type=float),
            FilterOption(name="tlength", type=str),
            FilterOption(name="count", type=int),
            FilterOption(name="fcount", type=int),
            FilterOption(name="fontfile", type=str),
            FilterOption(name="font", type=str),
            FilterOption(name="fontcolor", type=str),
            FilterOption(name="axisfile", type=str),
            FilterOption(name="axis", type=bool),
            FilterOption(name="csp", type=int),
            FilterOption(name="cscheme", type=str),
        ],
    )
    graph.append(filter)
    return graph


def hflip_vulkan(
    graph: Stream,
):
    """Horizontally flip the input video in Vulkan"""
    filter = Filter(command="hflip_vulkan", params=[])
    graph.append(filter)
    return graph


def vflip_vulkan(
    graph: Stream,
):
    """Vertically flip the input video in Vulkan"""
    filter = Filter(command="vflip_vulkan", params=[])
    graph.append(filter)
    return graph


def flip_vulkan(
    graph: Stream,
):
    """Flip both horizontally and vertically"""
    filter = Filter(command="flip_vulkan", params=[])
    graph.append(filter)
    return graph


def adelay(graph: Stream, delays: str, all: bool):
    """Delay one or more audio channels.
    :param str delays: set list of delays for each channel
    :param bool all: use last available delay for remained channels"""
    filter = Filter(command="adelay", params=[FilterOption(name="delays", type=str), FilterOption(name="all", type=bool)])
    graph.append(filter)
    return graph


def aevalsrc(graph: Stream, exprs: str, nb_samples: int, sample_rate: str, duration: int, channel_layout: str):
    """Generate an audio signal generated by an expression.
    :param str exprs: set the '|'-separated list of channels expressions
    :param int nb_samples: set the number of samples per requested frame
    :param str sample_rate: set the sample rate
    :param int duration: set audio duration
    :param str channel_layout: set channel layout"""
    filter = Filter(
        command="aevalsrc",
        params=[
            FilterOption(name="exprs", type=str),
            FilterOption(name="nb_samples", type=int),
            FilterOption(name="sample_rate", type=str),
            FilterOption(name="duration", type=int),
            FilterOption(name="channel_layout", type=str),
        ],
    )
    graph.append(filter)
    return graph


def aeval(graph: Stream, exprs: str, channel_layout: str):
    """Filter audio signal according to a specified expression.
    :param str exprs: set the '|'-separated list of channels expressions
    :param str channel_layout: set channel layout"""
    filter = Filter(command="aeval", params=[FilterOption(name="exprs", type=str), FilterOption(name="channel_layout", type=str)])
    graph.append(filter)
    return graph


def greyedge(graph: Stream, difford: int, minknorm: int, sigma: float):
    """Estimates scene illumination by grey edge assumption.
    :param int difford: set differentiation order
    :param int minknorm: set Minkowski norm
    :param float sigma: set sigma"""
    filter = Filter(
        command="greyedge",
        params=[FilterOption(name="difford", type=int), FilterOption(name="minknorm", type=int), FilterOption(name="sigma", type=float)],
    )
    graph.append(filter)
    return graph


def latency(
    graph: Stream,
):
    """Report video filtering latency."""
    filter = Filter(command="latency", params=[])
    graph.append(filter)
    return graph


def alatency(
    graph: Stream,
):
    """Report audio filtering latency."""
    filter = Filter(command="alatency", params=[])
    graph.append(filter)
    return graph


def kerndeint(graph: Stream, thresh: int, map: bool, order: bool, sharp: bool, twoway: bool):
    """Apply kernel deinterlacing to the input.
    :param int thresh: set the threshold
    :param bool map: set the map
    :param bool order: set the order
    :param bool sharp: set sharpening
    :param bool twoway: set twoway"""
    filter = Filter(
        command="kerndeint",
        params=[
            FilterOption(name="thresh", type=int),
            FilterOption(name="map", type=bool),
            FilterOption(name="order", type=bool),
            FilterOption(name="sharp", type=bool),
            FilterOption(name="twoway", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def showpalette(graph: Stream, s: int):
    """Display frame palette.
    :param int s: set pixel box size"""
    filter = Filter(command="showpalette", params=[FilterOption(name="s", type=int)])
    graph.append(filter)
    return graph


def hwdownload(
    graph: Stream,
):
    """Download a hardware frame to a normal frame"""
    filter = Filter(command="hwdownload", params=[])
    graph.append(filter)
    return graph


def tonemap_opencl(
    graph: Stream,
    tonemap: int,
    transfer: int,
    matrix: int,
    primaries: int,
    range: int,
    format: str,
    peak: float,
    param: float,
    desat: float,
    threshold: float,
):
    """Perform HDR to SDR conversion with tonemapping.
    :param int tonemap: tonemap algorithm selection
    :param int transfer: set transfer characteristic
    :param int matrix: set colorspace matrix
    :param int primaries: set color primaries
    :param int range: set color range
    :param str format: output pixel format
    :param float peak: signal peak override
    :param float param: tonemap parameter
    :param float desat: desaturation parameter
    :param float threshold: scene detection threshold"""
    filter = Filter(
        command="tonemap_opencl",
        params=[
            FilterOption(name="tonemap", type=int),
            FilterOption(name="transfer", type=int),
            FilterOption(name="matrix", type=int),
            FilterOption(name="primaries", type=int),
            FilterOption(name="range", type=int),
            FilterOption(name="format", type=str),
            FilterOption(name="peak", type=float),
            FilterOption(name="param", type=float),
            FilterOption(name="desat", type=float),
            FilterOption(name="threshold", type=float),
        ],
    )
    graph.append(filter)
    return graph


def nnedi(graph: Stream, weights: str, deint: int, field: int, planes: int, nsize: int, nns: int, qual: int, etype: int, pscrn: int):
    """Apply neural network edge directed interpolation intra-only deinterlacer.
    :param str weights: set weights file
    :param int deint: set which frames to deinterlace
    :param int field: set mode of operation
    :param int planes: set which planes to process
    :param int nsize: set size of local neighborhood around each pixel, used by the predictor neural network
    :param int nns: set number of neurons in predictor neural network
    :param int qual: set quality
    :param int etype: set which set of weights to use in the predictor
    :param int pscrn: set prescreening"""
    filter = Filter(
        command="nnedi",
        params=[
            FilterOption(name="weights", type=str),
            FilterOption(name="deint", type=int),
            FilterOption(name="field", type=int),
            FilterOption(name="planes", type=int),
            FilterOption(name="nsize", type=int),
            FilterOption(name="nns", type=int),
            FilterOption(name="qual", type=int),
            FilterOption(name="etype", type=int),
            FilterOption(name="pscrn", type=int),
        ],
    )
    graph.append(filter)
    return graph


def pullup(graph: Stream, jl: int, jr: int, jt: int, jb: int, sb: bool, mp: int):
    """Pullup from field sequence to frames.
    :param int jl: set left junk size
    :param int jr: set right junk size
    :param int jt: set top junk size
    :param int jb: set bottom junk size
    :param bool sb: set strict breaks
    :param int mp: set metric plane"""
    filter = Filter(
        command="pullup",
        params=[
            FilterOption(name="jl", type=int),
            FilterOption(name="jr", type=int),
            FilterOption(name="jt", type=int),
            FilterOption(name="jb", type=int),
            FilterOption(name="sb", type=bool),
            FilterOption(name="mp", type=int),
        ],
    )
    graph.append(filter)
    return graph


def crystalizer(graph: Stream, i: float, c: bool):
    """Simple audio noise sharpening filter.
    :param float i: set intensity
    :param bool c: enable clipping"""
    filter = Filter(command="crystalizer", params=[FilterOption(name="i", type=float), FilterOption(name="c", type=bool)])
    graph.append(filter)
    return graph


def flanger(graph: Stream, delay: float, depth: float, regen: float, width: float, speed: float, shape: int, phase: float, interp: int):
    """Apply a flanging effect to the audio.
    :param float delay: base delay in milliseconds
    :param float depth: added swept delay in milliseconds
    :param float regen: percentage regeneration (delayed signal feedback)
    :param float width: percentage of delayed signal mixed with original
    :param float speed: sweeps per second (Hz)
    :param int shape: swept wave shape
    :param float phase: swept wave percentage phase-shift for multi-channel
    :param int interp: delay-line interpolation"""
    filter = Filter(
        command="flanger",
        params=[
            FilterOption(name="delay", type=float),
            FilterOption(name="depth", type=float),
            FilterOption(name="regen", type=float),
            FilterOption(name="width", type=float),
            FilterOption(name="speed", type=float),
            FilterOption(name="shape", type=int),
            FilterOption(name="phase", type=float),
            FilterOption(name="interp", type=int),
        ],
    )
    graph.append(filter)
    return graph


def owdenoise(graph: Stream, depth: int, luma_strength: float, chroma_strength: float):
    """Denoise using wavelets.
    :param int depth: set depth
    :param float luma_strength: set luma strength
    :param float chroma_strength: set chroma strength"""
    filter = Filter(
        command="owdenoise",
        params=[
            FilterOption(name="depth", type=int),
            FilterOption(name="luma_strength", type=float),
            FilterOption(name="chroma_strength", type=float),
        ],
    )
    graph.append(filter)
    return graph


def sr(graph: Stream, dnn_backend: int, scale_factor: int, model: str, input: str, output: str):
    """Apply DNN-based image super resolution to the input.
    :param int dnn_backend: DNN backend used for model execution
    :param int scale_factor: scale factor for SRCNN model
    :param str model: path to model file specifying network architecture and its parameters
    :param str input: input name of the model
    :param str output: output name of the model"""
    filter = Filter(
        command="sr",
        params=[
            FilterOption(name="dnn_backend", type=int),
            FilterOption(name="scale_factor", type=int),
            FilterOption(name="model", type=str),
            FilterOption(name="input", type=str),
            FilterOption(name="output", type=str),
        ],
    )
    graph.append(filter)
    return graph


def overlay_opencl(graph: Stream, x: int, y: int):
    """Overlay one video on top of another
    :param int x: Overlay x position
    :param int y: Overlay y position"""
    filter = Filter(command="overlay_opencl", params=[FilterOption(name="x", type=int), FilterOption(name="y", type=int)])
    graph.append(filter)
    return graph


def extrastereo(graph: Stream, m: float, c: bool):
    """Increase difference between stereo audio channels.
    :param float m: set the difference coefficient
    :param bool c: enable clipping"""
    filter = Filter(command="extrastereo", params=[FilterOption(name="m", type=float), FilterOption(name="c", type=bool)])
    graph.append(filter)
    return graph


def maskedclamp(graph: Stream, undershoot: int, overshoot: int, planes: int):
    """Clamp first stream with second stream and third stream.
    :param int undershoot: set undershoot
    :param int overshoot: set overshoot
    :param int planes: set planes"""
    filter = Filter(
        command="maskedclamp",
        params=[FilterOption(name="undershoot", type=int), FilterOption(name="overshoot", type=int), FilterOption(name="planes", type=int)],
    )
    graph.append(filter)
    return graph


def pseudocolor(graph: Stream, c0: str, c1: str, c2: str, c3: str, index: int, preset: int, opacity: float):
    """Make pseudocolored video frames.
    :param str c0: set component #0 expression
    :param str c1: set component #1 expression
    :param str c2: set component #2 expression
    :param str c3: set component #3 expression
    :param int index: set component as base
    :param int preset: set preset
    :param float opacity: set pseudocolor opacity"""
    filter = Filter(
        command="pseudocolor",
        params=[
            FilterOption(name="c0", type=str),
            FilterOption(name="c1", type=str),
            FilterOption(name="c2", type=str),
            FilterOption(name="c3", type=str),
            FilterOption(name="index", type=int),
            FilterOption(name="preset", type=int),
            FilterOption(name="opacity", type=float),
        ],
    )
    graph.append(filter)
    return graph


def xfade(graph: Stream, transition: int, duration: int, offset: int, expr: str):
    """Cross fade one video with another video.
    :param int transition: set cross fade transition
    :param int duration: set cross fade duration
    :param int offset: set cross fade start relative to first input stream
    :param str expr: set expression for custom transition"""
    filter = Filter(
        command="xfade",
        params=[
            FilterOption(name="transition", type=int),
            FilterOption(name="duration", type=int),
            FilterOption(name="offset", type=int),
            FilterOption(name="expr", type=str),
        ],
    )
    graph.append(filter)
    return graph


def perspective(
    graph: Stream, x0: str, y0: str, x1: str, y1: str, x2: str, y2: str, x3: str, y3: str, interpolation: int, sense: int, eval: int
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
    :param int interpolation: set interpolation
    :param int sense: specify the sense of the coordinates
    :param int eval: specify when to evaluate expressions"""
    filter = Filter(
        command="perspective",
        params=[
            FilterOption(name="x0", type=str),
            FilterOption(name="y0", type=str),
            FilterOption(name="x1", type=str),
            FilterOption(name="y1", type=str),
            FilterOption(name="x2", type=str),
            FilterOption(name="y2", type=str),
            FilterOption(name="x3", type=str),
            FilterOption(name="y3", type=str),
            FilterOption(name="interpolation", type=int),
            FilterOption(name="sense", type=int),
            FilterOption(name="eval", type=int),
        ],
    )
    graph.append(filter)
    return graph


def tremolo(graph: Stream, f: float, d: float):
    """Apply tremolo effect.
    :param float f: set frequency in hertz
    :param float d: set depth as percentage"""
    filter = Filter(command="tremolo", params=[FilterOption(name="f", type=float), FilterOption(name="d", type=float)])
    graph.append(filter)
    return graph


def adynamicsmooth(graph: Stream, sensitivity: float, basefreq: float):
    """Apply Dynamic Smoothing of input audio.
    :param float sensitivity: set smooth sensitivity
    :param float basefreq: set base frequency"""
    filter = Filter(
        command="adynamicsmooth", params=[FilterOption(name="sensitivity", type=float), FilterOption(name="basefreq", type=float)]
    )
    graph.append(filter)
    return graph


def bm3d(
    graph: Stream,
    sigma: float,
    block: int,
    bstep: int,
    group: int,
    range: int,
    mstep: int,
    thmse: float,
    hdthr: float,
    estim: int,
    ref: bool,
    planes: int,
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
    :param int estim: set filtering estimation mode
    :param bool ref: have reference stream
    :param int planes: set planes to filter"""
    filter = Filter(
        command="bm3d",
        params=[
            FilterOption(name="sigma", type=float),
            FilterOption(name="block", type=int),
            FilterOption(name="bstep", type=int),
            FilterOption(name="group", type=int),
            FilterOption(name="range", type=int),
            FilterOption(name="mstep", type=int),
            FilterOption(name="thmse", type=float),
            FilterOption(name="hdthr", type=float),
            FilterOption(name="estim", type=int),
            FilterOption(name="ref", type=bool),
            FilterOption(name="planes", type=int),
        ],
    )
    graph.append(filter)
    return graph


def shuffleframes(graph: Stream, mapping: str):
    """Shuffle video frames.
    :param str mapping: set destination indexes of input frames"""
    filter = Filter(command="shuffleframes", params=[FilterOption(name="mapping", type=str)])
    graph.append(filter)
    return graph


def hqdn3d(graph: Stream, luma_spatial: float, chroma_spatial: float, luma_tmp: float, chroma_tmp: float):
    """Apply a High Quality 3D Denoiser.
    :param float luma_spatial: spatial luma strength
    :param float chroma_spatial: spatial chroma strength
    :param float luma_tmp: temporal luma strength
    :param float chroma_tmp: temporal chroma strength"""
    filter = Filter(
        command="hqdn3d",
        params=[
            FilterOption(name="luma_spatial", type=float),
            FilterOption(name="chroma_spatial", type=float),
            FilterOption(name="luma_tmp", type=float),
            FilterOption(name="chroma_tmp", type=float),
        ],
    )
    graph.append(filter)
    return graph


def lut3d(
    graph: Stream,
):
    """Adjust colors using a 3D LUT."""
    filter = Filter(command="lut3d", params=[])
    graph.append(filter)
    return graph


def haldclut(
    graph: Stream,
):
    """Adjust colors using a Hald CLUT."""
    filter = Filter(command="haldclut", params=[])
    graph.append(filter)
    return graph


def lut1d(graph: Stream, file: str, interp: int):
    """Adjust colors using a 1D LUT.
    :param str file: set 1D LUT file name
    :param int interp: select interpolation mode"""
    filter = Filter(command="lut1d", params=[FilterOption(name="file", type=str), FilterOption(name="interp", type=int)])
    graph.append(filter)
    return graph


def elbg(graph: Stream, codebook_length: int, nb_steps: int, seed: int, pal8: bool, use_alpha: bool):
    """Apply posterize effect, using the ELBG algorithm.
    :param int codebook_length: set codebook length
    :param int nb_steps: set max number of steps used to compute the mapping
    :param int seed: set the random seed
    :param bool pal8: set the pal8 output
    :param bool use_alpha: use alpha channel for mapping"""
    filter = Filter(
        command="elbg",
        params=[
            FilterOption(name="codebook_length", type=int),
            FilterOption(name="nb_steps", type=int),
            FilterOption(name="seed", type=int),
            FilterOption(name="pal8", type=bool),
            FilterOption(name="use_alpha", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def blackdetect(graph: Stream, d: float, picture_black_ratio_th: float, pixel_black_th: float):
    """Detect video intervals that are (almost) black.
    :param float d: set minimum detected black duration in seconds
    :param float picture_black_ratio_th: set the picture black ratio threshold
    :param float pixel_black_th: set the pixel black threshold"""
    filter = Filter(
        command="blackdetect",
        params=[
            FilterOption(name="d", type=float),
            FilterOption(name="picture_black_ratio_th", type=float),
            FilterOption(name="pixel_black_th", type=float),
        ],
    )
    graph.append(filter)
    return graph


def gradfun(graph: Stream, strength: float, radius: int):
    """Debands video quickly using gradients.
    :param float strength: The maximum amount by which the filter will change any one pixel.
    :param int radius: The neighborhood to fit the gradient to."""
    filter = Filter(command="gradfun", params=[FilterOption(name="strength", type=float), FilterOption(name="radius", type=int)])
    graph.append(filter)
    return graph


def firequalizer(
    graph: Stream,
    gain: str,
    gain_entry: str,
    delay: float,
    accuracy: float,
    wfunc: int,
    fixed: bool,
    multi: bool,
    zero_phase: bool,
    scale: int,
    dumpfile: str,
    dumpscale: int,
    fft2: bool,
    min_phase: bool,
):
    """Finite Impulse Response Equalizer.
    :param str gain: set gain curve
    :param str gain_entry: set gain entry
    :param float delay: set delay
    :param float accuracy: set accuracy
    :param int wfunc: set window function
    :param bool fixed: set fixed frame samples
    :param bool multi: set multi channels mode
    :param bool zero_phase: set zero phase mode
    :param int scale: set gain scale
    :param str dumpfile: set dump file
    :param int dumpscale: set dump scale
    :param bool fft2: set 2-channels fft
    :param bool min_phase: set minimum phase mode"""
    filter = Filter(
        command="firequalizer",
        params=[
            FilterOption(name="gain", type=str),
            FilterOption(name="gain_entry", type=str),
            FilterOption(name="delay", type=float),
            FilterOption(name="accuracy", type=float),
            FilterOption(name="wfunc", type=int),
            FilterOption(name="fixed", type=bool),
            FilterOption(name="multi", type=bool),
            FilterOption(name="zero_phase", type=bool),
            FilterOption(name="scale", type=int),
            FilterOption(name="dumpfile", type=str),
            FilterOption(name="dumpscale", type=int),
            FilterOption(name="fft2", type=bool),
            FilterOption(name="min_phase", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def interleave(graph: Stream, nb_inputs: int, duration: int):
    """Temporally interleave video inputs.
    :param int nb_inputs: set number of inputs
    :param int duration: how to determine the end-of-stream"""
    filter = Filter(command="interleave", params=[FilterOption(name="nb_inputs", type=int), FilterOption(name="duration", type=int)])
    graph.append(filter)
    return graph


def ainterleave(graph: Stream, nb_inputs: int, duration: int):
    """Temporally interleave audio inputs.
    :param int nb_inputs: set number of inputs
    :param int duration: how to determine the end-of-stream"""
    filter = Filter(command="ainterleave", params=[FilterOption(name="nb_inputs", type=int), FilterOption(name="duration", type=int)])
    graph.append(filter)
    return graph


def transpose_opencl(graph: Stream, dir: int, passthrough: int):
    """Transpose input video
    :param int dir: set transpose direction
    :param int passthrough: do not apply transposition if the input matches the specified geometry"""
    filter = Filter(command="transpose_opencl", params=[FilterOption(name="dir", type=int), FilterOption(name="passthrough", type=int)])
    graph.append(filter)
    return graph


def streamselect(graph: Stream, inputs: int, map: str):
    """Select video streams
    :param int inputs: number of input streams
    :param str map: input indexes to remap to outputs"""
    filter = Filter(command="streamselect", params=[FilterOption(name="inputs", type=int), FilterOption(name="map", type=str)])
    graph.append(filter)
    return graph


def astreamselect(
    graph: Stream,
):
    """Select audio streams"""
    filter = Filter(command="astreamselect", params=[])
    graph.append(filter)
    return graph


def deshake_opencl(
    graph: Stream,
    tripod: bool,
    debug: bool,
    adaptive_crop: bool,
    refine_features: bool,
    smooth_strength: float,
    smooth_window_multiplier: float,
):
    """Feature-point based video stabilization filter
    :param bool tripod: simulates a tripod by preventing any camera movement whatsoever from the original frame
    :param bool debug: turn on additional debugging information
    :param bool adaptive_crop: attempt to subtly crop borders to reduce mirrored content
    :param bool refine_features: refine feature point locations at a sub-pixel level
    :param float smooth_strength: smoothing strength (0 attempts to adaptively determine optimal strength)
    :param float smooth_window_multiplier: multiplier for number of frames to buffer for motion data"""
    filter = Filter(
        command="deshake_opencl",
        params=[
            FilterOption(name="tripod", type=bool),
            FilterOption(name="debug", type=bool),
            FilterOption(name="adaptive_crop", type=bool),
            FilterOption(name="refine_features", type=bool),
            FilterOption(name="smooth_strength", type=float),
            FilterOption(name="smooth_window_multiplier", type=float),
        ],
    )
    graph.append(filter)
    return graph


def unsharp_opencl(
    graph: Stream,
    luma_msize_x: float,
    luma_msize_y: float,
    luma_amount: float,
    chroma_msize_x: float,
    chroma_msize_y: float,
    chroma_amount: float,
):
    """Apply unsharp mask to input video
    :param float luma_msize_x: Set luma mask horizontal diameter (pixels)
    :param float luma_msize_y: Set luma mask vertical diameter (pixels)
    :param float luma_amount: Set luma amount (multiplier)
    :param float chroma_msize_x: Set chroma mask horizontal diameter (pixels after subsampling)
    :param float chroma_msize_y: Set chroma mask vertical diameter (pixels after subsampling)
    :param float chroma_amount: Set chroma amount (multiplier)"""
    filter = Filter(
        command="unsharp_opencl",
        params=[
            FilterOption(name="luma_msize_x", type=float),
            FilterOption(name="luma_msize_y", type=float),
            FilterOption(name="luma_amount", type=float),
            FilterOption(name="chroma_msize_x", type=float),
            FilterOption(name="chroma_msize_y", type=float),
            FilterOption(name="chroma_amount", type=float),
        ],
    )
    graph.append(filter)
    return graph


def colorlevels(
    graph: Stream,
    rimin: float,
    gimin: float,
    bimin: float,
    aimin: float,
    rimax: float,
    gimax: float,
    bimax: float,
    aimax: float,
    romin: float,
    gomin: float,
    bomin: float,
    aomin: float,
    romax: float,
    gomax: float,
    bomax: float,
    aomax: float,
    preserve: int,
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
    :param int preserve: set preserve color mode"""
    filter = Filter(
        command="colorlevels",
        params=[
            FilterOption(name="rimin", type=float),
            FilterOption(name="gimin", type=float),
            FilterOption(name="bimin", type=float),
            FilterOption(name="aimin", type=float),
            FilterOption(name="rimax", type=float),
            FilterOption(name="gimax", type=float),
            FilterOption(name="bimax", type=float),
            FilterOption(name="aimax", type=float),
            FilterOption(name="romin", type=float),
            FilterOption(name="gomin", type=float),
            FilterOption(name="bomin", type=float),
            FilterOption(name="aomin", type=float),
            FilterOption(name="romax", type=float),
            FilterOption(name="gomax", type=float),
            FilterOption(name="bomax", type=float),
            FilterOption(name="aomax", type=float),
            FilterOption(name="preserve", type=int),
        ],
    )
    graph.append(filter)
    return graph


def super2xsai(
    graph: Stream,
):
    """Scale the input by 2x using the Super2xSaI pixel art algorithm."""
    filter = Filter(command="super2xsai", params=[])
    graph.append(filter)
    return graph


def buffer(graph: Stream, width: int, height: int, pix_fmt: str, sar: int, time_base: int, frame_rate: int, colorspace: int, range: int):
    """Buffer video frames, and make them accessible to the filterchain.
    :param int sar: sample aspect ratio
    :param int colorspace: select colorspace
    :param int range: select color range"""
    filter = Filter(
        command="buffer",
        params=[
            FilterOption(name="width", type=int),
            FilterOption(name="height", type=int),
            FilterOption(name="pix_fmt", type=str),
            FilterOption(name="sar", type=int),
            FilterOption(name="time_base", type=int),
            FilterOption(name="frame_rate", type=int),
            FilterOption(name="colorspace", type=int),
            FilterOption(name="range", type=int),
        ],
    )
    graph.append(filter)
    return graph


def abuffer(graph: Stream, time_base: int, sample_rate: int, sample_fmt: str, channel_layout: str, channels: int):
    """Buffer audio frames, and make them accessible to the filterchain."""
    filter = Filter(
        command="abuffer",
        params=[
            FilterOption(name="time_base", type=int),
            FilterOption(name="sample_rate", type=int),
            FilterOption(name="sample_fmt", type=str),
            FilterOption(name="channel_layout", type=str),
            FilterOption(name="channels", type=int),
        ],
    )
    graph.append(filter)
    return graph


def arnndn(graph: Stream, model: str, mix: float):
    """Reduce noise from speech using Recurrent Neural Networks.
    :param str model: set model name
    :param float mix: set output vs input mix"""
    filter = Filter(command="arnndn", params=[FilterOption(name="model", type=str), FilterOption(name="mix", type=float)])
    graph.append(filter)
    return graph


def adecorrelate(graph: Stream, stages: int, seed: int):
    """Apply decorrelation to input audio.
    :param int stages: set filtering stages
    :param int seed: set random seed"""
    filter = Filter(command="adecorrelate", params=[FilterOption(name="stages", type=int), FilterOption(name="seed", type=int)])
    graph.append(filter)
    return graph


def join(graph: Stream, inputs: int, channel_layout: str, map: str):
    """Join multiple audio streams into multi-channel output.
    :param int inputs: Number of input streams.
    :param str channel_layout: Channel layout of the output stream.
    :param str map: A comma-separated list of channels maps in the format 'input_stream.input_channel-output_channel."""
    filter = Filter(
        command="join",
        params=[FilterOption(name="inputs", type=int), FilterOption(name="channel_layout", type=str), FilterOption(name="map", type=str)],
    )
    graph.append(filter)
    return graph


def pixdesctest(
    graph: Stream,
):
    """Test pixel format definitions."""
    filter = Filter(command="pixdesctest", params=[])
    graph.append(filter)
    return graph


def dnn_processing(graph: Stream, dnn_backend: int, model: str, input: str, output: str, backend_configs: str, options: str, _async: bool):
    """Apply DNN processing filter to the input.
    :param int dnn_backend: DNN backend
    :param str model: path to model file
    :param str input: input name of the model
    :param str output: output name of the model
    :param str backend_configs: backend configs
    :param str options: backend configs (deprecated, use backend_configs)
    :param bool async: use DNN async inference (ignored, use backend_configs='async=1')"""
    filter = Filter(
        command="dnn_processing",
        params=[
            FilterOption(name="dnn_backend", type=int),
            FilterOption(name="model", type=str),
            FilterOption(name="input", type=str),
            FilterOption(name="output", type=str),
            FilterOption(name="backend_configs", type=str),
            FilterOption(name="options", type=str),
            FilterOption(name="async", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def hdcd(
    graph: Stream, disable_autoconvert: bool, process_stereo: bool, cdt_ms: int, force_pe: bool, analyze_mode: int, bits_per_sample: int
):
    """Apply High Definition Compatible Digital (HDCD) decoding.
    :param bool disable_autoconvert: Disable any format conversion or resampling in the filter graph.
    :param bool process_stereo: Process stereo channels together. Only apply target_gain when both channels match.
    :param int cdt_ms: Code detect timer period in ms.
    :param bool force_pe: Always extend peaks above -3dBFS even when PE is not signaled.
    :param int analyze_mode: Replace audio with solid tone and signal some processing aspect in the amplitude.
    :param int bits_per_sample: Valid bits per sample (location of the true LSB)."""
    filter = Filter(
        command="hdcd",
        params=[
            FilterOption(name="disable_autoconvert", type=bool),
            FilterOption(name="process_stereo", type=bool),
            FilterOption(name="cdt_ms", type=int),
            FilterOption(name="force_pe", type=bool),
            FilterOption(name="analyze_mode", type=int),
            FilterOption(name="bits_per_sample", type=int),
        ],
    )
    graph.append(filter)
    return graph


def avectorscope(
    graph: Stream,
    mode: int,
    rate: str,
    size: int,
    rc: int,
    gc: int,
    bc: int,
    ac: int,
    rf: int,
    gf: int,
    bf: int,
    af: int,
    zoom: float,
    draw: int,
    scale: int,
    swap: bool,
    mirror: int,
):
    """Convert input audio to vectorscope video output.
    :param int mode: set mode
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
    :param int draw: set draw mode
    :param int scale: set amplitude scale mode
    :param bool swap: swap x axis with y axis
    :param int mirror: mirror axis"""
    filter = Filter(
        command="avectorscope",
        params=[
            FilterOption(name="mode", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="rc", type=int),
            FilterOption(name="gc", type=int),
            FilterOption(name="bc", type=int),
            FilterOption(name="ac", type=int),
            FilterOption(name="rf", type=int),
            FilterOption(name="gf", type=int),
            FilterOption(name="bf", type=int),
            FilterOption(name="af", type=int),
            FilterOption(name="zoom", type=float),
            FilterOption(name="draw", type=int),
            FilterOption(name="scale", type=int),
            FilterOption(name="swap", type=bool),
            FilterOption(name="mirror", type=int),
        ],
    )
    graph.append(filter)
    return graph


def vibrance(
    graph: Stream, intensity: float, rbal: float, gbal: float, bbal: float, rlum: float, glum: float, blum: float, alternate: bool
):
    """Boost or alter saturation.
    :param float intensity: set the intensity value
    :param float rbal: set the red balance value
    :param float gbal: set the green balance value
    :param float bbal: set the blue balance value
    :param float rlum: set the red luma coefficient
    :param float glum: set the green luma coefficient
    :param float blum: set the blue luma coefficient
    :param bool alternate: use alternate colors"""
    filter = Filter(
        command="vibrance",
        params=[
            FilterOption(name="intensity", type=float),
            FilterOption(name="rbal", type=float),
            FilterOption(name="gbal", type=float),
            FilterOption(name="bbal", type=float),
            FilterOption(name="rlum", type=float),
            FilterOption(name="glum", type=float),
            FilterOption(name="blum", type=float),
            FilterOption(name="alternate", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def showspectrum(
    graph: Stream,
    size: int,
    slide: int,
    mode: int,
    color: int,
    scale: int,
    fscale: int,
    saturation: float,
    win_func: int,
    orientation: int,
    overlap: float,
    gain: float,
    data: int,
    rotation: float,
    start: int,
    stop: int,
    fps: str,
    legend: bool,
    drange: float,
    limit: float,
    opacity: float,
):
    """Convert input audio to a spectrum video output.
    :param int size: set video size
    :param int slide: set sliding mode
    :param int mode: set channel display mode
    :param int color: set channel coloring
    :param int scale: set display scale
    :param int fscale: set frequency scale
    :param float saturation: color saturation multiplier
    :param int win_func: set window function
    :param int orientation: set orientation
    :param float overlap: set window overlap
    :param float gain: set scale gain
    :param int data: set data mode
    :param float rotation: color rotation
    :param int start: start frequency
    :param int stop: stop frequency
    :param str fps: set video rate
    :param bool legend: draw legend
    :param float drange: set dynamic range in dBFS
    :param float limit: set upper limit in dBFS
    :param float opacity: set opacity strength"""
    filter = Filter(
        command="showspectrum",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="slide", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="color", type=int),
            FilterOption(name="scale", type=int),
            FilterOption(name="fscale", type=int),
            FilterOption(name="saturation", type=float),
            FilterOption(name="win_func", type=int),
            FilterOption(name="orientation", type=int),
            FilterOption(name="overlap", type=float),
            FilterOption(name="gain", type=float),
            FilterOption(name="data", type=int),
            FilterOption(name="rotation", type=float),
            FilterOption(name="start", type=int),
            FilterOption(name="stop", type=int),
            FilterOption(name="fps", type=str),
            FilterOption(name="legend", type=bool),
            FilterOption(name="drange", type=float),
            FilterOption(name="limit", type=float),
            FilterOption(name="opacity", type=float),
        ],
    )
    graph.append(filter)
    return graph


def showspectrumpic(
    graph: Stream,
    size: int,
    mode: int,
    color: int,
    scale: int,
    fscale: int,
    saturation: float,
    win_func: int,
    orientation: int,
    gain: float,
    legend: bool,
    rotation: float,
    start: int,
    stop: int,
    drange: float,
    limit: float,
    opacity: float,
):
    """Convert input audio to a spectrum video output single picture.
    :param int size: set video size
    :param int mode: set channel display mode
    :param int color: set channel coloring
    :param int scale: set display scale
    :param int fscale: set frequency scale
    :param float saturation: color saturation multiplier
    :param int win_func: set window function
    :param int orientation: set orientation
    :param float gain: set scale gain
    :param bool legend: draw legend
    :param float rotation: color rotation
    :param int start: start frequency
    :param int stop: stop frequency
    :param float drange: set dynamic range in dBFS
    :param float limit: set upper limit in dBFS
    :param float opacity: set opacity strength"""
    filter = Filter(
        command="showspectrumpic",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="color", type=int),
            FilterOption(name="scale", type=int),
            FilterOption(name="fscale", type=int),
            FilterOption(name="saturation", type=float),
            FilterOption(name="win_func", type=int),
            FilterOption(name="orientation", type=int),
            FilterOption(name="gain", type=float),
            FilterOption(name="legend", type=bool),
            FilterOption(name="rotation", type=float),
            FilterOption(name="start", type=int),
            FilterOption(name="stop", type=int),
            FilterOption(name="drange", type=float),
            FilterOption(name="limit", type=float),
            FilterOption(name="opacity", type=float),
        ],
    )
    graph.append(filter)
    return graph


def setpts(graph: Stream, expr: str):
    """Set PTS for the output video frame.
    :param str expr: Expression determining the frame timestamp"""
    filter = Filter(command="setpts", params=[FilterOption(name="expr", type=str)])
    graph.append(filter)
    return graph


def asetpts(graph: Stream, expr: str):
    """Set PTS for the output audio frame.
    :param str expr: Expression determining the frame timestamp"""
    filter = Filter(command="asetpts", params=[FilterOption(name="expr", type=str)])
    graph.append(filter)
    return graph


def volumedetect(
    graph: Stream,
):
    """Detect audio volume."""
    filter = Filter(command="volumedetect", params=[])
    graph.append(filter)
    return graph


def stereo3d(graph: Stream, _in: int, out: int):
    """Convert video stereoscopic 3D view.
    :param int in: set input format
    :param int out: set output format"""
    filter = Filter(command="stereo3d", params=[FilterOption(name="in", type=int), FilterOption(name="out", type=int)])
    graph.append(filter)
    return graph


def rotate(graph: Stream, angle: str, out_w: str, out_h: str, fillcolor: str, bilinear: bool):
    """Rotate the input image.
    :param str angle: set angle (in radians)
    :param str out_w: set output width expression
    :param str out_h: set output height expression
    :param str fillcolor: set background fill color
    :param bool bilinear: use bilinear interpolation"""
    filter = Filter(
        command="rotate",
        params=[
            FilterOption(name="angle", type=str),
            FilterOption(name="out_w", type=str),
            FilterOption(name="out_h", type=str),
            FilterOption(name="fillcolor", type=str),
            FilterOption(name="bilinear", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def apsyclip(graph: Stream, level_in: float, level_out: float, clip: float, diff: bool, adaptive: float, iterations: int, level: bool):
    """Audio Psychoacoustic Clipper.
    :param float level_in: set input level
    :param float level_out: set output level
    :param float clip: set clip level
    :param bool diff: enable difference
    :param float adaptive: set adaptive distortion
    :param int iterations: set iterations
    :param bool level: set auto level"""
    filter = Filter(
        command="apsyclip",
        params=[
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="clip", type=float),
            FilterOption(name="diff", type=bool),
            FilterOption(name="adaptive", type=float),
            FilterOption(name="iterations", type=int),
            FilterOption(name="level", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def dcshift(graph: Stream, shift: float, limitergain: float):
    """Apply a DC shift to the audio.
    :param float shift: set DC shift
    :param float limitergain: set limiter gain"""
    filter = Filter(command="dcshift", params=[FilterOption(name="shift", type=float), FilterOption(name="limitergain", type=float)])
    graph.append(filter)
    return graph


def varblur(graph: Stream, min_r: int, max_r: int, planes: int):
    """Apply Variable Blur filter.
    :param int min_r: set min blur radius
    :param int max_r: set max blur radius
    :param int planes: set planes to filter"""
    filter = Filter(
        command="varblur",
        params=[FilterOption(name="min_r", type=int), FilterOption(name="max_r", type=int), FilterOption(name="planes", type=int)],
    )
    graph.append(filter)
    return graph


def threshold(graph: Stream, planes: int):
    """Threshold first video stream using other video streams.
    :param int planes: set planes to filter"""
    filter = Filter(command="threshold", params=[FilterOption(name="planes", type=int)])
    graph.append(filter)
    return graph


def replaygain(graph: Stream, track_gain: float, track_peak: float):
    """ReplayGain scanner.
    :param float track_gain: track gain (dB)
    :param float track_peak: track peak"""
    filter = Filter(command="replaygain", params=[FilterOption(name="track_gain", type=float), FilterOption(name="track_peak", type=float)])
    graph.append(filter)
    return graph


def xfade_vulkan(graph: Stream, transition: int, duration: int, offset: int):
    """Cross fade one video with another video.
    :param int transition: set cross fade transition
    :param int duration: set cross fade duration
    :param int offset: set cross fade start relative to first input stream"""
    filter = Filter(
        command="xfade_vulkan",
        params=[FilterOption(name="transition", type=int), FilterOption(name="duration", type=int), FilterOption(name="offset", type=int)],
    )
    graph.append(filter)
    return graph


def selectivecolor(
    graph: Stream,
    correction_method: int,
    reds: str,
    yellows: str,
    greens: str,
    cyans: str,
    blues: str,
    magentas: str,
    whites: str,
    neutrals: str,
    blacks: str,
    psfile: str,
):
    """Apply CMYK adjustments to specific color ranges.
    :param int correction_method: select correction method
    :param str reds: adjust red regions
    :param str yellows: adjust yellow regions
    :param str greens: adjust green regions
    :param str cyans: adjust cyan regions
    :param str blues: adjust blue regions
    :param str magentas: adjust magenta regions
    :param str whites: adjust white regions
    :param str neutrals: adjust neutral regions
    :param str blacks: adjust black regions
    :param str psfile: set Photoshop selectivecolor file name"""
    filter = Filter(
        command="selectivecolor",
        params=[
            FilterOption(name="correction_method", type=int),
            FilterOption(name="reds", type=str),
            FilterOption(name="yellows", type=str),
            FilterOption(name="greens", type=str),
            FilterOption(name="cyans", type=str),
            FilterOption(name="blues", type=str),
            FilterOption(name="magentas", type=str),
            FilterOption(name="whites", type=str),
            FilterOption(name="neutrals", type=str),
            FilterOption(name="blacks", type=str),
            FilterOption(name="psfile", type=str),
        ],
    )
    graph.append(filter)
    return graph


def apad(graph: Stream, packet_size: int, pad_len: int, whole_len: int, pad_dur: int, whole_dur: int):
    """Pad audio with silence.
    :param int packet_size: set silence packet size
    :param int pad_len: set number of samples of silence to add
    :param int whole_len: set minimum target number of samples in the audio stream
    :param int pad_dur: set duration of silence to add
    :param int whole_dur: set minimum target duration in the audio stream"""
    filter = Filter(
        command="apad",
        params=[
            FilterOption(name="packet_size", type=int),
            FilterOption(name="pad_len", type=int),
            FilterOption(name="whole_len", type=int),
            FilterOption(name="pad_dur", type=int),
            FilterOption(name="whole_dur", type=int),
        ],
    )
    graph.append(filter)
    return graph


def fieldhint(graph: Stream, hint: str, mode: int):
    """Field matching using hints.
    :param str hint: set hint file
    :param int mode: set hint mode"""
    filter = Filter(command="fieldhint", params=[FilterOption(name="hint", type=str), FilterOption(name="mode", type=int)])
    graph.append(filter)
    return graph


def fieldmatch(
    graph: Stream,
    order: int,
    mode: int,
    ppsrc: bool,
    field: int,
    mchroma: bool,
    y0: int,
    y1: int,
    scthresh: float,
    combmatch: int,
    combdbg: int,
    cthresh: int,
    chroma: bool,
    blockx: int,
    blocky: int,
    combpel: int,
):
    """Field matching for inverse telecine.
    :param int order: specify the assumed field order
    :param int mode: set the matching mode or strategy to use
    :param bool ppsrc: mark main input as a pre-processed input and activate clean source input stream
    :param int field: set the field to match from
    :param bool mchroma: set whether or not chroma is included during the match comparisons
    :param int y0: define an exclusion band which excludes the lines between y0 and y1 from the field matching decision
    :param int y1: define an exclusion band which excludes the lines between y0 and y1 from the field matching decision
    :param float scthresh: set scene change detection threshold
    :param int combmatch: set combmatching mode
    :param int combdbg: enable comb debug
    :param int cthresh: set the area combing threshold used for combed frame detection
    :param bool chroma: set whether or not chroma is considered in the combed frame decision
    :param int blockx: set the x-axis size of the window used during combed frame detection
    :param int blocky: set the y-axis size of the window used during combed frame detection
    :param int combpel: set the number of combed pixels inside any of the blocky by blockx size blocks on the frame for the frame to be detected as combed
    """
    filter = Filter(
        command="fieldmatch",
        params=[
            FilterOption(name="order", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="ppsrc", type=bool),
            FilterOption(name="field", type=int),
            FilterOption(name="mchroma", type=bool),
            FilterOption(name="y0", type=int),
            FilterOption(name="y1", type=int),
            FilterOption(name="scthresh", type=float),
            FilterOption(name="combmatch", type=int),
            FilterOption(name="combdbg", type=int),
            FilterOption(name="cthresh", type=int),
            FilterOption(name="chroma", type=bool),
            FilterOption(name="blockx", type=int),
            FilterOption(name="blocky", type=int),
            FilterOption(name="combpel", type=int),
        ],
    )
    graph.append(filter)
    return graph


def bench(graph: Stream, action: int):
    """Benchmark part of a filtergraph.
    :param int action: set action"""
    filter = Filter(command="bench", params=[FilterOption(name="action", type=int)])
    graph.append(filter)
    return graph


def abench(graph: Stream, action: int):
    """Benchmark part of a filtergraph.
    :param int action: set action"""
    filter = Filter(command="abench", params=[FilterOption(name="action", type=int)])
    graph.append(filter)
    return graph


def aloop(graph: Stream, loop: int, size: int, start: int, time: int):
    """Loop audio samples.
    :param int loop: number of loops
    :param int size: max number of samples to loop
    :param int start: set the loop start sample
    :param int time: set the loop start time"""
    filter = Filter(
        command="aloop",
        params=[
            FilterOption(name="loop", type=int),
            FilterOption(name="size", type=int),
            FilterOption(name="start", type=int),
            FilterOption(name="time", type=int),
        ],
    )
    graph.append(filter)
    return graph


def loop(graph: Stream, loop: int, size: int, start: int, time: int):
    """Loop video frames.
    :param int loop: number of loops
    :param int size: max number of frames to loop
    :param int start: set the loop start frame
    :param int time: set the loop start time"""
    filter = Filter(
        command="loop",
        params=[
            FilterOption(name="loop", type=int),
            FilterOption(name="size", type=int),
            FilterOption(name="start", type=int),
            FilterOption(name="time", type=int),
        ],
    )
    graph.append(filter)
    return graph


def hue(graph: Stream, h: str, s: str, H: str, b: str):
    """Adjust the hue and saturation of the input video.
    :param str h: set the hue angle degrees expression
    :param str s: set the saturation expression
    :param str H: set the hue angle radians expression
    :param str b: set the brightness expression"""
    filter = Filter(
        command="hue",
        params=[
            FilterOption(name="h", type=str),
            FilterOption(name="s", type=str),
            FilterOption(name="H", type=str),
            FilterOption(name="b", type=str),
        ],
    )
    graph.append(filter)
    return graph


def remap_opencl(graph: Stream, interp: int, fill: str):
    """Remap pixels using OpenCL.
    :param int interp: set interpolation method
    :param str fill: set the color of the unmapped pixels"""
    filter = Filter(command="remap_opencl", params=[FilterOption(name="interp", type=int), FilterOption(name="fill", type=str)])
    graph.append(filter)
    return graph


def field(graph: Stream, type: int):
    """Extract a field from the input video.
    :param int type: set field type (top or bottom)"""
    filter = Filter(command="field", params=[FilterOption(name="type", type=int)])
    graph.append(filter)
    return graph


def aspectralstats(graph: Stream, win_size: int, win_func: int, overlap: float, measure: str):
    """Show frequency domain statistics about audio frames.
    :param int win_size: set the window size
    :param int win_func: set window function
    :param float overlap: set window overlap
    :param str measure: select the parameters which are measured"""
    filter = Filter(
        command="aspectralstats",
        params=[
            FilterOption(name="win_size", type=int),
            FilterOption(name="win_func", type=int),
            FilterOption(name="overlap", type=float),
            FilterOption(name="measure", type=str),
        ],
    )
    graph.append(filter)
    return graph


def axcorrelate(graph: Stream, size: int, algo: int):
    """Cross-correlate two audio streams.
    :param int size: set the segment size
    :param int algo: set the algorithm"""
    filter = Filter(command="axcorrelate", params=[FilterOption(name="size", type=int), FilterOption(name="algo", type=int)])
    graph.append(filter)
    return graph


def realtime(
    graph: Stream,
):
    """Slow down filtering to match realtime."""
    filter = Filter(command="realtime", params=[])
    graph.append(filter)
    return graph


def arealtime(
    graph: Stream,
):
    """Slow down filtering to match realtime."""
    filter = Filter(command="arealtime", params=[])
    graph.append(filter)
    return graph


def aperms(
    graph: Stream,
):
    """Set permissions for the output audio frame."""
    filter = Filter(command="aperms", params=[])
    graph.append(filter)
    return graph


def perms(
    graph: Stream,
):
    """Set permissions for the output video frame."""
    filter = Filter(command="perms", params=[])
    graph.append(filter)
    return graph


def nullsink(
    graph: Stream,
):
    """Do absolutely nothing with the input video."""
    filter = Filter(command="nullsink", params=[])
    graph.append(filter)
    return graph


def setparams(graph: Stream, field_mode: int, range: int, color_primaries: int, color_trc: int, colorspace: int):
    """Force field, or color property for the output video frame.
    :param int field_mode: select interlace mode
    :param int range: select color range
    :param int color_primaries: select color primaries
    :param int color_trc: select color transfer
    :param int colorspace: select colorspace"""
    filter = Filter(
        command="setparams",
        params=[
            FilterOption(name="field_mode", type=int),
            FilterOption(name="range", type=int),
            FilterOption(name="color_primaries", type=int),
            FilterOption(name="color_trc", type=int),
            FilterOption(name="colorspace", type=int),
        ],
    )
    graph.append(filter)
    return graph


def setrange(graph: Stream, range: int):
    """Force color range for the output video frame.
    :param int range: select color range"""
    filter = Filter(command="setrange", params=[FilterOption(name="range", type=int)])
    graph.append(filter)
    return graph


def setfield(graph: Stream, mode: int):
    """Force field for the output video frame.
    :param int mode: select interlace mode"""
    filter = Filter(command="setfield", params=[FilterOption(name="mode", type=int)])
    graph.append(filter)
    return graph


def setdar(graph: Stream, dar: str, max: int):
    """Set the frame display aspect ratio.
    :param str dar: set display aspect ratio
    :param int max: set max value for nominator or denominator in the ratio"""
    filter = Filter(command="setdar", params=[FilterOption(name="dar", type=str), FilterOption(name="max", type=int)])
    graph.append(filter)
    return graph


def setsar(graph: Stream, sar: str, max: int):
    """Set the pixel sample aspect ratio.
    :param str sar: set sample (pixel) aspect ratio
    :param int max: set max value for nominator or denominator in the ratio"""
    filter = Filter(command="setsar", params=[FilterOption(name="sar", type=str), FilterOption(name="max", type=int)])
    graph.append(filter)
    return graph


def lumakey(graph: Stream, threshold: float, tolerance: float, softness: float):
    """Turns a certain luma into transparency.
    :param float threshold: set the threshold value
    :param float tolerance: set the tolerance value
    :param float softness: set the softness value"""
    filter = Filter(
        command="lumakey",
        params=[
            FilterOption(name="threshold", type=float),
            FilterOption(name="tolerance", type=float),
            FilterOption(name="softness", type=float),
        ],
    )
    graph.append(filter)
    return graph


def cas(graph: Stream, strength: float, planes: str):
    """Contrast Adaptive Sharpen.
    :param float strength: set the sharpening strength
    :param str planes: set what planes to filter"""
    filter = Filter(command="cas", params=[FilterOption(name="strength", type=float), FilterOption(name="planes", type=str)])
    graph.append(filter)
    return graph


def spp(graph: Stream, quality: int, qp: int, mode: int, use_bframe_qp: bool):
    """Apply a simple post processing filter.
    :param int quality: set quality
    :param int qp: force a constant quantizer parameter
    :param int mode: set thresholding mode
    :param bool use_bframe_qp: use B-frames' QP"""
    filter = Filter(
        command="spp",
        params=[
            FilterOption(name="quality", type=int),
            FilterOption(name="qp", type=int),
            FilterOption(name="mode", type=int),
            FilterOption(name="use_bframe_qp", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def multiply(graph: Stream, scale: float, offset: float, planes: str):
    """Multiply first video stream with second video stream.
    :param float scale: set scale
    :param float offset: set offset
    :param str planes: set planes"""
    filter = Filter(
        command="multiply",
        params=[FilterOption(name="scale", type=float), FilterOption(name="offset", type=float), FilterOption(name="planes", type=str)],
    )
    graph.append(filter)
    return graph


def vignette(graph: Stream, angle: str, x0: str, y0: str, mode: int, eval: int, dither: bool, aspect: int):
    """Make or reverse a vignette effect.
    :param str angle: set lens angle
    :param str x0: set circle center position on x-axis
    :param str y0: set circle center position on y-axis
    :param int mode: set forward/backward mode
    :param int eval: specify when to evaluate expressions
    :param bool dither: set dithering
    :param int aspect: set aspect ratio"""
    filter = Filter(
        command="vignette",
        params=[
            FilterOption(name="angle", type=str),
            FilterOption(name="x0", type=str),
            FilterOption(name="y0", type=str),
            FilterOption(name="mode", type=int),
            FilterOption(name="eval", type=int),
            FilterOption(name="dither", type=bool),
            FilterOption(name="aspect", type=int),
        ],
    )
    graph.append(filter)
    return graph


def silenceremove(
    graph: Stream,
    start_periods: int,
    start_duration: int,
    start_threshold: float,
    start_silence: int,
    start_mode: int,
    stop_periods: int,
    stop_duration: int,
    stop_threshold: float,
    stop_silence: int,
    stop_mode: int,
    detection: int,
    window: int,
    timestamp: int,
):
    """Remove silence.
    :param int start_periods: set periods of silence parts to skip from start
    :param int start_duration: set start duration of non-silence part
    :param float start_threshold: set threshold for start silence detection
    :param int start_silence: set start duration of silence part to keep
    :param int start_mode: set which channel will trigger trimming from start
    :param int stop_periods: set periods of silence parts to skip from end
    :param int stop_duration: set stop duration of silence part
    :param float stop_threshold: set threshold for stop silence detection
    :param int stop_silence: set stop duration of silence part to keep
    :param int stop_mode: set which channel will trigger trimming from end
    :param int detection: set how silence is detected
    :param int window: set duration of window for silence detection
    :param int timestamp: set how every output frame timestamp is processed"""
    filter = Filter(
        command="silenceremove",
        params=[
            FilterOption(name="start_periods", type=int),
            FilterOption(name="start_duration", type=int),
            FilterOption(name="start_threshold", type=float),
            FilterOption(name="start_silence", type=int),
            FilterOption(name="start_mode", type=int),
            FilterOption(name="stop_periods", type=int),
            FilterOption(name="stop_duration", type=int),
            FilterOption(name="stop_threshold", type=float),
            FilterOption(name="stop_silence", type=int),
            FilterOption(name="stop_mode", type=int),
            FilterOption(name="detection", type=int),
            FilterOption(name="window", type=int),
            FilterOption(name="timestamp", type=int),
        ],
    )
    graph.append(filter)
    return graph


def deesser(graph: Stream, i: float, m: float, f: float, s: int):
    """Apply de-essing to the audio.
    :param float i: set intensity
    :param float m: set max deessing
    :param float f: set frequency
    :param int s: set output mode"""
    filter = Filter(
        command="deesser",
        params=[
            FilterOption(name="i", type=float),
            FilterOption(name="m", type=float),
            FilterOption(name="f", type=float),
            FilterOption(name="s", type=int),
        ],
    )
    graph.append(filter)
    return graph


def aselect(graph: Stream, expr: str, outputs: int):
    """Select audio frames to pass in output.
    :param str expr: set an expression to use for selecting frames
    :param int outputs: set the number of outputs"""
    filter = Filter(command="aselect", params=[FilterOption(name="expr", type=str), FilterOption(name="outputs", type=int)])
    graph.append(filter)
    return graph


def select(graph: Stream, expr: str, outputs: int):
    """Select video frames to pass in output.
    :param str expr: set an expression to use for selecting frames
    :param int outputs: set the number of outputs"""
    filter = Filter(command="select", params=[FilterOption(name="expr", type=str), FilterOption(name="outputs", type=int)])
    graph.append(filter)
    return graph


def nlmeans(graph: Stream, s: float, p: int, pc: int, r: int, rc: int):
    """Non-local means denoiser.
    :param float s: denoising strength
    :param int p: patch size
    :param int pc: patch size for chroma planes
    :param int r: research window
    :param int rc: research window for chroma planes"""
    filter = Filter(
        command="nlmeans",
        params=[
            FilterOption(name="s", type=float),
            FilterOption(name="p", type=int),
            FilterOption(name="pc", type=int),
            FilterOption(name="r", type=int),
            FilterOption(name="rc", type=int),
        ],
    )
    graph.append(filter)
    return graph


def mestimate(graph: Stream, method: int, mb_size: int, search_param: int):
    """Generate motion vectors.
    :param int method: motion estimation method
    :param int mb_size: macroblock size
    :param int search_param: search parameter"""
    filter = Filter(
        command="mestimate",
        params=[FilterOption(name="method", type=int), FilterOption(name="mb_size", type=int), FilterOption(name="search_param", type=int)],
    )
    graph.append(filter)
    return graph


def edgedetect(graph: Stream, high: float, low: float, mode: int, planes: str):
    """Detect and draw edge.
    :param float high: set high threshold
    :param float low: set low threshold
    :param int mode: set mode
    :param str planes: set planes to filter"""
    filter = Filter(
        command="edgedetect",
        params=[
            FilterOption(name="high", type=float),
            FilterOption(name="low", type=float),
            FilterOption(name="mode", type=int),
            FilterOption(name="planes", type=str),
        ],
    )
    graph.append(filter)
    return graph


def removegrain(graph: Stream, m0: int, m1: int, m2: int, m3: int):
    """Remove grain.
    :param int m0: set mode for 1st plane
    :param int m1: set mode for 2nd plane
    :param int m2: set mode for 3rd plane
    :param int m3: set mode for 4th plane"""
    filter = Filter(
        command="removegrain",
        params=[
            FilterOption(name="m0", type=int),
            FilterOption(name="m1", type=int),
            FilterOption(name="m2", type=int),
            FilterOption(name="m3", type=int),
        ],
    )
    graph.append(filter)
    return graph


def sierpinski(graph: Stream, size: int, rate: str, seed: int, jump: int, type: int):
    """Render a Sierpinski fractal.
    :param int size: set frame size
    :param str rate: set frame rate
    :param int seed: set the seed
    :param int jump: set the jump
    :param int type: set fractal type"""
    filter = Filter(
        command="sierpinski",
        params=[
            FilterOption(name="size", type=int),
            FilterOption(name="rate", type=str),
            FilterOption(name="seed", type=int),
            FilterOption(name="jump", type=int),
            FilterOption(name="type", type=int),
        ],
    )
    graph.append(filter)
    return graph


def acrossover(graph: Stream, split: str, order: int, level: float, gain: str, precision: int):
    """Split audio into per-bands streams.
    :param str split: set split frequencies
    :param int order: set filter order
    :param float level: set input gain
    :param str gain: set output bands gain
    :param int precision: set processing precision"""
    filter = Filter(
        command="acrossover",
        params=[
            FilterOption(name="split", type=str),
            FilterOption(name="order", type=int),
            FilterOption(name="level", type=float),
            FilterOption(name="gain", type=str),
            FilterOption(name="precision", type=int),
        ],
    )
    graph.append(filter)
    return graph


def swaprect(graph: Stream, w: str, h: str, x1: str, y1: str, x2: str, y2: str):
    """Swap 2 rectangular objects in video.
    :param str w: set rect width
    :param str h: set rect height
    :param str x1: set 1st rect x top left coordinate
    :param str y1: set 1st rect y top left coordinate
    :param str x2: set 2nd rect x top left coordinate
    :param str y2: set 2nd rect y top left coordinate"""
    filter = Filter(
        command="swaprect",
        params=[
            FilterOption(name="w", type=str),
            FilterOption(name="h", type=str),
            FilterOption(name="x1", type=str),
            FilterOption(name="y1", type=str),
            FilterOption(name="x2", type=str),
            FilterOption(name="y2", type=str),
        ],
    )
    graph.append(filter)
    return graph


def abitscope(graph: Stream, rate: str, size: int, colors: str, mode: int):
    """Convert input audio to audio bit scope video output.
    :param str rate: set video rate
    :param int size: set video size
    :param str colors: set channels colors
    :param int mode: set output mode"""
    filter = Filter(
        command="abitscope",
        params=[
            FilterOption(name="rate", type=str),
            FilterOption(name="size", type=int),
            FilterOption(name="colors", type=str),
            FilterOption(name="mode", type=int),
        ],
    )
    graph.append(filter)
    return graph


def lenscorrection(graph: Stream, cx: float, cy: float, k1: float, k2: float, i: int, fc: str):
    """Rectify the image by correcting for lens distortion.
    :param float cx: set relative center x
    :param float cy: set relative center y
    :param float k1: set quadratic distortion factor
    :param float k2: set double quadratic distortion factor
    :param int i: set interpolation type
    :param str fc: set the color of the unmapped pixels"""
    filter = Filter(
        command="lenscorrection",
        params=[
            FilterOption(name="cx", type=float),
            FilterOption(name="cy", type=float),
            FilterOption(name="k1", type=float),
            FilterOption(name="k2", type=float),
            FilterOption(name="i", type=int),
            FilterOption(name="fc", type=str),
        ],
    )
    graph.append(filter)
    return graph


def vfrdet(
    graph: Stream,
):
    """Variable frame rate detect filter."""
    filter = Filter(command="vfrdet", params=[])
    graph.append(filter)
    return graph


def atilt(graph: Stream, freq: float, slope: float, width: float, order: int, level: float):
    """Apply spectral tilt to audio.
    :param float freq: set central frequency
    :param float slope: set filter slope
    :param float width: set filter width
    :param int order: set filter order
    :param float level: set input level"""
    filter = Filter(
        command="atilt",
        params=[
            FilterOption(name="freq", type=float),
            FilterOption(name="slope", type=float),
            FilterOption(name="width", type=float),
            FilterOption(name="order", type=int),
            FilterOption(name="level", type=float),
        ],
    )
    graph.append(filter)
    return graph


def remap(graph: Stream, format: int, fill: str):
    """Remap pixels.
    :param int format: set output format
    :param str fill: set the color of the unmapped pixels"""
    filter = Filter(command="remap", params=[FilterOption(name="format", type=int), FilterOption(name="fill", type=str)])
    graph.append(filter)
    return graph


def bilateral(graph: Stream, sigmaS: float, sigmaR: float, planes: int):
    """Apply Bilateral filter.
    :param float sigmaS: set spatial sigma
    :param float sigmaR: set range sigma
    :param int planes: set planes to filter"""
    filter = Filter(
        command="bilateral",
        params=[FilterOption(name="sigmaS", type=float), FilterOption(name="sigmaR", type=float), FilterOption(name="planes", type=int)],
    )
    graph.append(filter)
    return graph


def huesaturation(
    graph: Stream,
    hue: float,
    saturation: float,
    intensity: float,
    colors: str,
    strength: float,
    rw: float,
    gw: float,
    bw: float,
    lightness: bool,
):
    """Apply hue-saturation-intensity adjustments.
    :param float hue: set the hue shift
    :param float saturation: set the saturation shift
    :param float intensity: set the intensity shift
    :param str colors: set colors range
    :param float strength: set the filtering strength
    :param float rw: set the red weight
    :param float gw: set the green weight
    :param float bw: set the blue weight
    :param bool lightness: set the preserve lightness"""
    filter = Filter(
        command="huesaturation",
        params=[
            FilterOption(name="hue", type=float),
            FilterOption(name="saturation", type=float),
            FilterOption(name="intensity", type=float),
            FilterOption(name="colors", type=str),
            FilterOption(name="strength", type=float),
            FilterOption(name="rw", type=float),
            FilterOption(name="gw", type=float),
            FilterOption(name="bw", type=float),
            FilterOption(name="lightness", type=bool),
        ],
    )
    graph.append(filter)
    return graph


def epx(graph: Stream, n: int):
    """Scale the input using EPX algorithm.
    :param int n: set scale factor"""
    filter = Filter(command="epx", params=[FilterOption(name="n", type=int)])
    graph.append(filter)
    return graph


def lagfun(graph: Stream, decay: float, planes: str):
    """Slowly update darker pixels.
    :param float decay: set decay
    :param str planes: set what planes to filter"""
    filter = Filter(command="lagfun", params=[FilterOption(name="decay", type=float), FilterOption(name="planes", type=str)])
    graph.append(filter)
    return graph


def alphamerge(
    graph: Stream,
):
    """Copy the luma value of the second input into the alpha channel of the first input."""
    filter = Filter(command="alphamerge", params=[])
    graph.append(filter)
    return graph


def signalstats(graph: Stream, stat: str, out: int, c: str):
    """Generate statistics from video analysis.
    :param str stat: set statistics filters
    :param int out: set video filter
    :param str c: set highlight color"""
    filter = Filter(
        command="signalstats",
        params=[FilterOption(name="stat", type=str), FilterOption(name="out", type=int), FilterOption(name="c", type=str)],
    )
    graph.append(filter)
    return graph


def asetnsamples(graph: Stream, nb_out_samples: int, pad: bool):
    """Set the number of samples for each output audio frames.
    :param int nb_out_samples: set the number of per-frame output samples
    :param bool pad: pad last frame with zeros"""
    filter = Filter(command="asetnsamples", params=[FilterOption(name="nb_out_samples", type=int), FilterOption(name="pad", type=bool)])
    graph.append(filter)
    return graph


def adenorm(graph: Stream, level: float, type: int):
    """Remedy denormals by adding extremely low-level noise.
    :param float level: set level
    :param int type: set type"""
    filter = Filter(command="adenorm", params=[FilterOption(name="level", type=float), FilterOption(name="type", type=int)])
    graph.append(filter)
    return graph


def surround(
    graph: Stream,
    chl_out: str,
    chl_in: str,
    level_in: float,
    level_out: float,
    lfe: bool,
    lfe_low: int,
    lfe_high: int,
    lfe_mode: int,
    smooth: float,
    angle: float,
    focus: float,
    fc_in: float,
    fc_out: float,
    fl_in: float,
    fl_out: float,
    fr_in: float,
    fr_out: float,
    sl_in: float,
    sl_out: float,
    sr_in: float,
    sr_out: float,
    bl_in: float,
    bl_out: float,
    br_in: float,
    br_out: float,
    bc_in: float,
    bc_out: float,
    lfe_in: float,
    lfe_out: float,
    allx: float,
    ally: float,
    fcx: float,
    flx: float,
    frx: float,
    blx: float,
    brx: float,
    slx: float,
    srx: float,
    bcx: float,
    fcy: float,
    fly: float,
    fry: float,
    bly: float,
    bry: float,
    sly: float,
    sry: float,
    bcy: float,
    win_size: int,
    win_func: int,
    overlap: float,
):
    """Apply audio surround upmix filter.
    :param str chl_out: set output channel layout
    :param str chl_in: set input channel layout
    :param float level_in: set input level
    :param float level_out: set output level
    :param bool lfe: output LFE
    :param int lfe_low: LFE low cut off
    :param int lfe_high: LFE high cut off
    :param int lfe_mode: set LFE channel mode
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
    :param int win_func: set window function
    :param float overlap: set window overlap"""
    filter = Filter(
        command="surround",
        params=[
            FilterOption(name="chl_out", type=str),
            FilterOption(name="chl_in", type=str),
            FilterOption(name="level_in", type=float),
            FilterOption(name="level_out", type=float),
            FilterOption(name="lfe", type=bool),
            FilterOption(name="lfe_low", type=int),
            FilterOption(name="lfe_high", type=int),
            FilterOption(name="lfe_mode", type=int),
            FilterOption(name="smooth", type=float),
            FilterOption(name="angle", type=float),
            FilterOption(name="focus", type=float),
            FilterOption(name="fc_in", type=float),
            FilterOption(name="fc_out", type=float),
            FilterOption(name="fl_in", type=float),
            FilterOption(name="fl_out", type=float),
            FilterOption(name="fr_in", type=float),
            FilterOption(name="fr_out", type=float),
            FilterOption(name="sl_in", type=float),
            FilterOption(name="sl_out", type=float),
            FilterOption(name="sr_in", type=float),
            FilterOption(name="sr_out", type=float),
            FilterOption(name="bl_in", type=float),
            FilterOption(name="bl_out", type=float),
            FilterOption(name="br_in", type=float),
            FilterOption(name="br_out", type=float),
            FilterOption(name="bc_in", type=float),
            FilterOption(name="bc_out", type=float),
            FilterOption(name="lfe_in", type=float),
            FilterOption(name="lfe_out", type=float),
            FilterOption(name="allx", type=float),
            FilterOption(name="ally", type=float),
            FilterOption(name="fcx", type=float),
            FilterOption(name="flx", type=float),
            FilterOption(name="frx", type=float),
            FilterOption(name="blx", type=float),
            FilterOption(name="brx", type=float),
            FilterOption(name="slx", type=float),
            FilterOption(name="srx", type=float),
            FilterOption(name="bcx", type=float),
            FilterOption(name="fcy", type=float),
            FilterOption(name="fly", type=float),
            FilterOption(name="fry", type=float),
            FilterOption(name="bly", type=float),
            FilterOption(name="bry", type=float),
            FilterOption(name="sly", type=float),
            FilterOption(name="sry", type=float),
            FilterOption(name="bcy", type=float),
            FilterOption(name="win_size", type=int),
            FilterOption(name="win_func", type=int),
            FilterOption(name="overlap", type=float),
        ],
    )
    graph.append(filter)
    return graph


def noise(
    graph: Stream,
    all_seed: int,
    all_strength: int,
    alls: int,
    all_flags: str,
    allf: str,
    c0_seed: int,
    c0_strength: int,
    c0s: int,
    c0_flags: str,
    c0f: str,
    c1_seed: int,
    c1_strength: int,
    c1s: int,
    c1_flags: str,
    c1f: str,
    c2_seed: int,
    c2_strength: int,
    c2s: int,
    c2_flags: str,
    c2f: str,
    c3_seed: int,
    c3_strength: int,
    c3s: int,
    c3_flags: str,
    c3f: str,
):
    """Add noise.
    :param int all_seed: set component #0 noise seed
    :param int all_strength: set component #0 strength
    :param int alls: set component #0 strength
    :param str all_flags: set component #0 flags
    :param str allf: set component #0 flags
    :param int c0_seed: set component #0 noise seed
    :param int c0_strength: set component #0 strength
    :param int c0s: set component #0 strength
    :param str c0_flags: set component #0 flags
    :param str c0f: set component #0 flags
    :param int c1_seed: set component #1 noise seed
    :param int c1_strength: set component #1 strength
    :param int c1s: set component #1 strength
    :param str c1_flags: set component #1 flags
    :param str c1f: set component #1 flags
    :param int c2_seed: set component #2 noise seed
    :param int c2_strength: set component #2 strength
    :param int c2s: set component #2 strength
    :param str c2_flags: set component #2 flags
    :param str c2f: set component #2 flags
    :param int c3_seed: set component #3 noise seed
    :param int c3_strength: set component #3 strength
    :param int c3s: set component #3 strength
    :param str c3_flags: set component #3 flags
    :param str c3f: set component #3 flags"""
    filter = Filter(
        command="noise",
        params=[
            FilterOption(name="all_seed", type=int),
            FilterOption(name="all_strength", type=int),
            FilterOption(name="alls", type=int),
            FilterOption(name="all_flags", type=str),
            FilterOption(name="allf", type=str),
            FilterOption(name="c0_seed", type=int),
            FilterOption(name="c0_strength", type=int),
            FilterOption(name="c0s", type=int),
            FilterOption(name="c0_flags", type=str),
            FilterOption(name="c0f", type=str),
            FilterOption(name="c1_seed", type=int),
            FilterOption(name="c1_strength", type=int),
            FilterOption(name="c1s", type=int),
            FilterOption(name="c1_flags", type=str),
            FilterOption(name="c1f", type=str),
            FilterOption(name="c2_seed", type=int),
            FilterOption(name="c2_strength", type=int),
            FilterOption(name="c2s", type=int),
            FilterOption(name="c2_flags", type=str),
            FilterOption(name="c2f", type=str),
            FilterOption(name="c3_seed", type=int),
            FilterOption(name="c3_strength", type=int),
            FilterOption(name="c3s", type=int),
            FilterOption(name="c3_flags", type=str),
            FilterOption(name="c3f", type=str),
        ],
    )
    graph.append(filter)
    return graph


def pp7(graph: Stream, qp: int, mode: int):
    """Apply Postprocessing 7 filter.
    :param int qp: force a constant quantizer parameter
    :param int mode: set thresholding mode"""
    filter = Filter(command="pp7", params=[FilterOption(name="qp", type=int), FilterOption(name="mode", type=int)])
    graph.append(filter)
    return graph


def showinfo(graph: Stream, checksum: bool, udu_sei_as_ascii: bool):
    """Show textual information for each video frame.
    :param bool checksum: calculate checksums
    :param bool udu_sei_as_ascii: try to print user data unregistered SEI as ascii character when possible"""
    filter = Filter(command="showinfo", params=[FilterOption(name="checksum", type=bool), FilterOption(name="udu_sei_as_ascii", type=bool)])
    graph.append(filter)
    return graph


def mergeplanes(
    graph: Stream, mapping: int, format: str, map0s: int, map0p: int, map1s: int, map1p: int, map2s: int, map2p: int, map3s: int, map3p: int
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
    :param int map3p: set 4th input to output plane mapping"""
    filter = Filter(
        command="mergeplanes",
        params=[
            FilterOption(name="mapping", type=int),
            FilterOption(name="format", type=str),
            FilterOption(name="map0s", type=int),
            FilterOption(name="map0p", type=int),
            FilterOption(name="map1s", type=int),
            FilterOption(name="map1p", type=int),
            FilterOption(name="map2s", type=int),
            FilterOption(name="map2p", type=int),
            FilterOption(name="map3s", type=int),
            FilterOption(name="map3p", type=int),
        ],
    )
    graph.append(filter)
    return graph


def afwtdn(
    graph: Stream, sigma: float, levels: int, wavet: int, percent: float, profile: bool, adaptive: bool, samples: int, softness: float
):
    """Denoise audio stream using Wavelets.
    :param float sigma: set noise sigma
    :param int levels: set number of wavelet levels
    :param int wavet: set wavelet type
    :param float percent: set percent of full denoising
    :param bool profile: profile noise
    :param bool adaptive: adaptive profiling of noise
    :param int samples: set frame size in number of samples
    :param float softness: set thresholding softness"""
    filter = Filter(
        command="afwtdn",
        params=[
            FilterOption(name="sigma", type=float),
            FilterOption(name="levels", type=int),
            FilterOption(name="wavet", type=int),
            FilterOption(name="percent", type=float),
            FilterOption(name="profile", type=bool),
            FilterOption(name="adaptive", type=bool),
            FilterOption(name="samples", type=int),
            FilterOption(name="softness", type=float),
        ],
    )
    graph.append(filter)
    return graph


def hsvkey(graph: Stream, hue: float, sat: float, val: float, similarity: float, blend: float):
    """Turns a certain HSV range into transparency. Operates on YUV colors.
    :param float hue: set the hue value
    :param float sat: set the saturation value
    :param float val: set the value value
    :param float similarity: set the hsvkey similarity value
    :param float blend: set the hsvkey blend value"""
    filter = Filter(
        command="hsvkey",
        params=[
            FilterOption(name="hue", type=float),
            FilterOption(name="sat", type=float),
            FilterOption(name="val", type=float),
            FilterOption(name="similarity", type=float),
            FilterOption(name="blend", type=float),
        ],
    )
    graph.append(filter)
    return graph


def hsvhold(graph: Stream, hue: float, sat: float, val: float, similarity: float, blend: float):
    """Turns a certain HSV range into gray.
    :param float hue: set the hue value
    :param float sat: set the saturation value
    :param float val: set the value value
    :param float similarity: set the hsvhold similarity value
    :param float blend: set the hsvhold blend value"""
    filter = Filter(
        command="hsvhold",
        params=[
            FilterOption(name="hue", type=float),
            FilterOption(name="sat", type=float),
            FilterOption(name="val", type=float),
            FilterOption(name="similarity", type=float),
            FilterOption(name="blend", type=float),
        ],
    )
    graph.append(filter)
    return graph


def vaguedenoiser(graph: Stream, threshold: float, method: int, nsteps: int, percent: float, planes: int, type: int):
    """Apply a Wavelet based Denoiser.
    :param float threshold: set filtering strength
    :param int method: set filtering method
    :param int nsteps: set number of steps
    :param float percent: set percent of full denoising
    :param int planes: set planes to filter
    :param int type: set threshold type"""
    filter = Filter(
        command="vaguedenoiser",
        params=[
            FilterOption(name="threshold", type=float),
            FilterOption(name="method", type=int),
            FilterOption(name="nsteps", type=int),
            FilterOption(name="percent", type=float),
            FilterOption(name="planes", type=int),
            FilterOption(name="type", type=int),
        ],
    )
    graph.append(filter)
    return graph
