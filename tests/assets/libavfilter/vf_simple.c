#include "avfilter.h"
#include "internal.h"

static const AVFilterPad avfilter_simple_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = 123,
        .config_props = 345,
    },
};

const AVFilter ff_vf_simple = {
    .name          = "simple123",
    .description   = NULL_IF_CONFIG_SMALL("simple filter"),
    .priv_size     = 123,
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(avfilter_simple_outputs),
    FILTER_QUERY_FUNC(query_formats),
    .flags         = AVFILTER_FLAG_SLICE_THREADS | AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
