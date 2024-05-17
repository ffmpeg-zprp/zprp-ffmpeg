//imports are ommited
#include "avfilter.h"
#include "libavutil/opt.h"
#include "internal.h"

// the name is infered from ff_vf_xxxxxx
static const AVOption oneoption_options[] = {
    { "oname", "odesc", 4, AV_OPT_TYPE_STRING, {.str="default"}, 0, 0, 0, .unit = "unit123" },
    { NULL }
};

static const AVFilterPad avfilter_oneoption_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = 123,
        .config_props = 345,
    },
};

const AVFilter ff_vf_oneoption = {
    .name          = "oneoption",
    .description   = NULL_IF_CONFIG_SMALL("I have one option"),
    .priv_size     = 123,
    FILTER_INPUTS(avfilter_oneoption_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_QUERY_FUNC(query_formats),
    .flags         = AVFILTER_FLAG_SLICE_THREADS | AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
