#include "libavutil/opt.h"
#include "avfilter.h"
#include "internal.h"
#include "video.h"

static const AVOption myfilter_options[] = {
    {"color", "select color", 4, AV_OPT_TYPE_INT, {.str = "red"}, 0, 0, 0, .unit = "color_option"},
    {"red", "desc1", 0, AV_OPT_TYPE_CONST, {.str = 1}, 0, 0, 0, .unit = "color_option"},
    {"green", "desc2", 0, AV_OPT_TYPE_CONST, {.str = 2}, 0, 0, 0, .unit = "color_option"},
    {"blue", "desc3", 0, AV_OPT_TYPE_CONST, {.str = 3}, 0, 0, 0, .unit = "color_option"},
    {NULL}};

void init() {};

struct{} bench_class;

const AVFilter ff_vf_myfilter = {
    .name = "myfilter",
    .description = NULL_IF_CONFIG_SMALL("Has option with three colors."),
    .priv_size = 123,
    .init = init,
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    .priv_class = &bench_class,
    .flags = AVFILTER_FLAG_METADATA_ONLY,
};
