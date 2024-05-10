typedef int VADisplay;
typedef int VASurfaceAttrib;
typedef int VASurfaceID;
typedef int VAConfigID;
typedef int VAContextID;
typedef int VARectangle;
typedef int VABufferID;
int VAProcFilterCount=1;
typedef struct {
    int rotation_state;
    int mirror_state;
    } VAProcPipelineParameterBuffer;
typedef int VAHdrMetaDataHDR10;
typedef int VAProcFilterCapDeinterlacing;
typedef struct {
    int rotation_flags;
    int mirror_flags;
    } VAProcPipelineCaps;
typedef int VAStatus;

#define VA_STATUS_SUCCESS 1
#define VA_INVALID_ID 1
#define VA_ROTATION_270 1
#define VA_MIRROR_VERTICAL 1
#define VA_ROTATION_90 1
#define VA_ROTATION_180 1
#define VA_ROTATION_NONE 1
#define VA_MIRROR_NONE 1
#define VA_MIRROR_HORIZONTAL 1
#define VA_MIRROR_NONE 1
