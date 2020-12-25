// MPEG-2 python data loader.
// Part of this implementation is modified from the tutorial at
// https://blog.csdn.net/leixiaohua1020/article/details/50618190
// and FFmpeg extract_mv example.


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <stdio.h>
#include <stdarg.h>

#include <libavformat/avformat.h>


/* ------------------------ Global variables ----------------------------- */

static const char *filename = NULL;

static PyObject *MPEG2Error;


/*------------------------- Macros and constants  -------------------------*/

#define DCT 0
#define MV  1

#define INPUT_BUFFER_SIZE 4096

#define IS_INTERLACED(a) ((a)&MB_TYPE_INTERLACED)
#define IS_16X16(a)      ((a)&MB_TYPE_16x16)
#define IS_16X8(a)       ((a)&MB_TYPE_16x8)
#define IS_8X16(a)       ((a)&MB_TYPE_8x16)
#define IS_8X8(a)        ((a)&MB_TYPE_8x8)
#define USES_LIST(a, list) ((a) & ((MB_TYPE_P0L0|MB_TYPE_P1L0)<<(2*(list))))


/* ------------------------- Data structures ----------------------------- */

typedef struct AVMotionVector {
    /**
     * Where the current macroblock comes from; negative value when it comes
     * from the past, positive value when it comes from the future.
     * XXX: set exact relative ref frame reference instead of a +/- 1 "direction".
     */
    int32_t source;
    /**
     * Width and height of the block.
     */
    uint8_t w, h;
    /**
     * Absolute source position. Can be outside the frame area.
     */
    int16_t src_x, src_y;
    /**
     * Absolute destination position. Can be outside the frame area.
     */
    int16_t dst_x, dst_y;
    /**
     * Extra flag information.
     * Currently unused.
     */
    uint64_t flags;
    /**
     * Motion vector
     * src_x = dst_x + motion_x / motion_scale
     * src_y = dst_y + motion_y / motion_scale
     */
    int32_t motion_x, motion_y;
    uint16_t motion_scale;
} AVMotionVector;


/* -------------------- Local function prototypes ------------------------ */

static int fatal_error(char * fmt, ...);
static void *av_realloc_array(void *ptr, 
                              size_t nmemb, 
                              size_t size);
static int add_mb(AVMotionVector *mb, 
                uint32_t mb_type,
                int dst_x, int dst_y,
                int src_x, int src_y,
                int direction);
static AVMotionVector *av_frame_get_motion_vectors(AVCodecContext *pCodecCtx, 
                                                   AVFrame *pFrame, 
                                                   size_t *size);
static void create_and_load_dctcoeff(AVFrame *pFrame,
                                     PyArrayObject * npy_arr);
static void create_and_load_mv(AVMotionVector *mvs, 
                               size_t size, 
                               PyArrayObject * npy_arr);
static int decode_frame(AVCodecContext *pCodecCtx, 
                        AVFrame *pFrame, 
                        AVPacket *packet);
static int parse_video(PyArrayObject ** npy_arr,
                       int gop_index,
                       int frame_index,
                       int representation_type);
static int count_frames(int *gop_count, 
                        int *frame_count);


/*----------------------------- Routines ----------------------------------*/

static int fatal_error(char * fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    fprintf(stdout, "Error: ");
    vfprintf(stdout, fmt, args);
    fprintf(stdout, "\n");
    va_end(args);

    return -1;
}


static void *av_realloc_array(void *ptr, 
                              size_t nmemb, 
                              size_t size)
{
    if (!size || nmemb >= INT_MAX / size)
        return NULL;
    return av_realloc(ptr, nmemb * size);
}


static int add_mb(AVMotionVector *mb, 
                  uint32_t mb_type,
                  int dst_x, int dst_y,
                  int src_x, int src_y,
                  int direction)
{
    mb->w = IS_8X8(mb_type) || IS_8X16(mb_type) ? 8 : 16;
    mb->h = IS_8X8(mb_type) || IS_16X8(mb_type) ? 8 : 16;
    mb->src_x = src_x;
    mb->src_y = src_y;
    mb->dst_x = dst_x;
    mb->dst_y = dst_y;
    mb->source = direction ? 1 : -1;
    mb->flags = 0; // XXX: does mb_type contain extra information that could be exported here?
    return 1;
}


static AVMotionVector *av_frame_get_motion_vectors(AVCodecContext *pCodecCtx, 
                                                   AVFrame *pFrame, 
                                                   size_t *size)
{
    int mb_width  = (pFrame->width + 15) >> 4;
    int mb_height = (pFrame->height + 15) >> 4;
    int mb_stride = mb_width + 1;
    int mv_sample_log2 = 4 - pFrame->motion_subsample_log2;
    int mv_stride = (mb_width << mv_sample_log2) + (pCodecCtx->codec_id == CODEC_ID_H264 ? 0 : 1);
    int quarter_sample = (pCodecCtx->flags & CODEC_FLAG_QPEL) != 0;
    int shift = 1 + quarter_sample;

    uint32_t *mbtype_table = pFrame->mb_type;
    int mb_x, mb_y, mbcount = 0;

    /* size is width * height * 2 * 4 where 2 is for directions and 4 is
     * for the maximum number of MB (4 MB in case of IS_8x8) */
    AVMotionVector *mvs = av_malloc_array(mb_width * mb_height, 2 * 4 * sizeof(AVMotionVector));
    if (!mvs)
        return NULL;

    for (mb_y = 0; mb_y < mb_height; mb_y++) {
        for (mb_x = 0; mb_x < mb_width; mb_x++) {
            int i, direction, mb_type = mbtype_table[mb_x + mb_y * mb_stride];
            for (direction = 0; direction < 2; direction++) {
                if (!USES_LIST(mb_type, direction))
                    continue;
                if (IS_8X8(mb_type)) {
                    for (i = 0; i < 4; i++) {
                        int sx = mb_x * 16 + 4 + 8 * (i & 1);
                        int sy = mb_y * 16 + 4 + 8 * (i >> 1);
                        int xy = (mb_x * 2 + (i & 1) +
                                 (mb_y * 2 + (i >> 1)) * mv_stride) << (mv_sample_log2 - 1);
                        int mx = (pFrame->motion_val[direction][xy][0] >> shift) + sx;
                        int my = (pFrame->motion_val[direction][xy][1] >> shift) + sy;
                        mbcount += add_mb(mvs + mbcount, mb_type, sx, sy, mx, my, direction);
                    }
                } else if (IS_16X8(mb_type)) {
                    for (i = 0; i < 2; i++) {
                        int sx = mb_x * 16 + 8;
                        int sy = mb_y * 16 + 4 + 8 * i;
                        int xy = (mb_x * 2 + (mb_y * 2 + i) * mv_stride) << (mv_sample_log2 - 1);
                        int mx = (pFrame->motion_val[direction][xy][0] >> shift);
                        int my = (pFrame->motion_val[direction][xy][1] >> shift);

                        if (IS_INTERLACED(mb_type))
                            my *= 2;

                        mbcount += add_mb(mvs + mbcount, mb_type, sx, sy, mx + sx, my + sy, direction);
                    }
                } else if (IS_8X16(mb_type)) {
                    for (i = 0; i < 2; i++) {
                        int sx = mb_x * 16 + 4 + 8 * i;
                        int sy = mb_y * 16 + 8;
                        int xy = (mb_x * 2 + i + mb_y * 2 * mv_stride) << (mv_sample_log2 - 1);
                        int mx = pFrame->motion_val[direction][xy][0] >> shift;
                        int my = pFrame->motion_val[direction][xy][1] >> shift;

                        if (IS_INTERLACED(mb_type))
                            my *= 2;

                        mbcount += add_mb(mvs + mbcount, mb_type, sx, sy, mx + sx, my + sy, direction);
                    }
                } else {
                    int sx = mb_x * 16 + 8;
                    int sy = mb_y * 16 + 8;
                    int xy = (mb_x + mb_y * mv_stride) << mv_sample_log2;
                    int mx = (pFrame->motion_val[direction][xy][0]>>shift) + sx;
                    int my = (pFrame->motion_val[direction][xy][1]>>shift) + sy;
                    mbcount += add_mb(mvs + mbcount, mb_type, sx, sy, mx, my, direction);
                }
            }
        }
    }

    if (mbcount) 
        mvs = av_realloc_array(mvs, mbcount, sizeof(AVMotionVector));
    else
        av_freep(&mvs);

    *size = mbcount * sizeof(AVMotionVector);
    return mvs;
}


static void create_and_load_dctcoeff(AVFrame *pFrame,
                                     PyArrayObject * npy_arr)
{
    if (!pFrame->dct_coeff)
        return;

    int mb_width  = (pFrame->width + 15) >> 4;
    int mb_height = (pFrame->height + 15) >> 4;
    int mb_stride = mb_width + 1;
    int mb_array_size = mb_stride * mb_height;

    int height = mb_height << 4;
    int width  = mb_width << 4;
    int linesize = width * 3;

    int stride_y = linesize;
    int stride_x = 3;

    int16_t *src  = (int16_t*) pFrame->dct_coeff;
    int16_t *dest = (int16_t*) PyArray_DATA(npy_arr);

    for (int mb_y = 0; mb_y < mb_height; mb_y++) {
        for (int mb_x = 0; mb_x < mb_width; mb_x++) {
            int mb_index = mb_x + mb_y * mb_stride;

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    int k = 2 * i + j;

                    int y_start = 2 * mb_y + i;
                    int x_start = 2 * mb_x + j;
                    for (int b_y = 0; b_y < 8; b_y++) {
                        for (int b_x = 0; b_x < 8; b_x++) {
                            int p_y, p_x;
                            p_y = 8 * y_start + b_y;
                            p_x = 8 * x_start + b_x;
                            if (p_y >= 0 && p_y < height && p_x >= 0 && p_x < width) {
                                int xy = p_y * stride_y + p_x * stride_x;
                                dest[xy + 0] = src[6 * 64 * mb_index + k * 64 + b_y * 8 + b_x];
                                dest[xy + 1] = src[6 * 64 * mb_index + 4 * 64 + b_y * 8 + b_x];
                                dest[xy + 2] = src[6 * 64 * mb_index + 5 * 64 + b_y * 8 + b_x];
                            }
                        }
                    }
                }
            }
        }
    }
}


static void create_and_load_mv(AVMotionVector *mvs, 
                               size_t size, 
                               PyArrayObject * npy_arr)
{
    int p_dst_x, p_dst_y, p_src_x, p_src_y;
    int16_t val_x, val_y;

    int ndim = PyArray_NDIM(npy_arr);
    npy_intp * dims = PyArray_DIMS(npy_arr);
    int height = dims[0];
    int width  = dims[1];

    for (int i = 0; i < size / sizeof(*mvs); i++) {
        AVMotionVector *mv = &mvs[i];
        assert(mv->source == -1);

        if (mv->dst_x - mv->src_x != 0 || mv->dst_y - mv->src_y != 0) {

            val_x = mv->dst_x - mv->src_x;
            val_y = mv->dst_y - mv->src_y;

            for (int x_start = (-1 * mv->w / 2); x_start < mv->w / 2; ++x_start) {
                for (int y_start = (-1 * mv->h / 2); y_start < mv->h / 2; ++y_start) {
                    p_dst_x = mv->dst_x + x_start;
                    p_dst_y = mv->dst_y + y_start;

                    p_src_x = mv->src_x + x_start;
                    p_src_y = mv->src_y + y_start;

                    if (p_dst_y >= 0 && p_dst_y < height && 
                        p_dst_x >= 0 && p_dst_x < width &&
                        p_src_y >= 0 && p_src_y < height && 
                        p_src_x >= 0 && p_src_x < width) {

                        // Write MV. 
                        *((int16_t*)PyArray_GETPTR3(npy_arr, p_dst_y, p_dst_x, 0)) = val_x;
                        *((int16_t*)PyArray_GETPTR3(npy_arr, p_dst_y, p_dst_x, 1)) = val_y;
                    }
                }
            }
        }
    }
    av_freep(&mvs);
}


static int decode_frame(AVCodecContext *pCodecCtx, 
                        AVFrame *pFrame, 
                        AVPacket *packet)
{
    int bytesDecoded;
    int frameFinished;

    /* Decode the next chunk of data. */
    bytesDecoded = avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, packet);

    /* Was there an error? */
    if (bytesDecoded < 0)
        return fatal_error("Error while decoding frame.");

    if (packet->data) {
        packet->size -= bytesDecoded;
        packet->data += bytesDecoded;
    }
	
    return frameFinished;
}


static int parse_video(PyArrayObject ** npy_arr,
                       int gop_index,
                       int frame_index,
                       int representation_type)
{
    AVCodec *pCodec = NULL;
    AVCodecContext *pCodecCtx = NULL;  
    AVCodecParserContext *pCodecParserCtx = NULL;  

    FILE *fp;
    AVFrame *pFrame;

    uint8_t input_buffer[INPUT_BUFFER_SIZE + FF_INPUT_BUFFER_PADDING_SIZE];

    uint8_t *data;  
    size_t data_size;
    AVPacket packet;  
    int bytesParsed;

    /* Register all codecs. */
    avcodec_register_all();  

    /* Set end of buffer to 0 (this ensures that no overreading happens for damaged MPEG streams). */
    memset(input_buffer + INPUT_BUFFER_SIZE, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    av_init_packet(&packet);

    /* Find the MPEG-2 video decoder. */  
    pCodec = avcodec_find_decoder(AV_CODEC_ID_MPEG2VIDEO);  
    if (!pCodec)
        return fatal_error("Codec not found.");

    pCodecParserCtx = av_parser_init(AV_CODEC_ID_MPEG2VIDEO);  
    if (!pCodecParserCtx)  
        return fatal_error("Could not allocate video parser context.");  

    pCodecCtx = avcodec_alloc_context3(pCodec);  
    if (!pCodecCtx)
        return fatal_error("Could not allocate video codec context.");  

    /* Debug the DCT coefficients. */  
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "debug", "dct_coeff+mv", 0);

    /* Inform the codec that we can handle truncated bitstreams -- i.e.,
       bitstreams where frame boundaries can fall in the middle of packets.
     */
    if (pCodec->capabilities & CODEC_CAP_TRUNCATED)
        pCodecCtx->flags |= CODEC_FLAG_TRUNCATED;

    /* Open codec. */	      
    if (avcodec_open2(pCodecCtx, pCodec, &opts) < 0)
        return fatal_error("Could not open codec.");  

    /* Allocate video frame. */
    pFrame = avcodec_alloc_frame();

    /* Open input file. */
    fp = fopen(filename, "rb");  
    if (!fp)
        return fatal_error("Could not open file %s.", filename);  

    int gop_count   = -1;
    int frame_count =  0;

    while (!feof(fp)) {
        /* Read raw data from the input file. */
        data_size = fread(input_buffer, 1, INPUT_BUFFER_SIZE, fp);  
        if (!data_size)  
            break;  
		
        /* Use the parser to split the data into frames. */
        data = input_buffer;  
        while (data_size > 0) {   
            bytesParsed = av_parser_parse2(pCodecParserCtx, pCodecCtx,  
                                           &packet.data, &packet.size,  
                                           data, data_size,  
                                           AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);  
            if (bytesParsed < 0)
                return fatal_error("Error while parsing.");
            data      += bytesParsed;  
            data_size -= bytesParsed;

            if (!packet.size)  
                continue;  

            if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I)
                gop_count++;

            if (gop_count == gop_index && frame_count <= frame_index) {
                int ret = decode_frame(pCodecCtx, pFrame, &packet);
                if (ret < 0)
                    return ret;
                if (ret > 0) {

                    // Initialize arrays.
                    if (! (*npy_arr)) {
                        int height = ((pCodecCtx->height + 15) >> 4) << 4;
                        int width  = ((pCodecCtx->width + 15) >> 4) << 4;
                        int depth  = (representation_type == DCT) ? 3 : 2;

                        npy_intp dims[4];
                        dims[0] = height;
                        dims[1] = width;
                        dims[2] = depth;
                        *npy_arr = (PyArrayObject *) PyArray_ZEROS(3, dims, NPY_INT16, 0);
                    }
					
                    if (frame_count == frame_index) {
                        if (representation_type == DCT)
                            create_and_load_dctcoeff(pFrame, *npy_arr);
                        else {
                            size_t size;
                            AVMotionVector *mvs;
                            mvs = av_frame_get_motion_vectors(pCodecCtx, pFrame, &size);
                            if (mvs)
                                create_and_load_mv(mvs, size, *npy_arr);
                        }
                    }					

                    frame_count++;
                }
            }			
        }
    }
  
    /* Some codecs, such as MPEG, transmit the I and P frame with a
       latency of one frame. You must do the following to have a
       chance to get the last frame of the video */
    packet.data = NULL;
    packet.size = 0;
    int ret = decode_frame(pCodecCtx, pFrame, &packet);
    if (ret < 0)
        return ret;
    if (ret > 0) {
        if (gop_count == gop_index && frame_count == frame_index) {
            if (representation_type == DCT)
                create_and_load_dctcoeff(pFrame, *npy_arr);
            else {
                size_t size;
                AVMotionVector *mvs;
                mvs = av_frame_get_motion_vectors(pCodecCtx, pFrame, &size);
                if (mvs)
                    create_and_load_mv(mvs, size, *npy_arr);
            }
        }
		
        frame_count++;
    }
 
    /* Close input file. */
    fclose(fp);

    /* Close the parser. */
    av_parser_close(pCodecParserCtx);  

    /* Free the frame. */
    avcodec_free_frame(&pFrame);  

    /* Close the codec. */
    avcodec_close(pCodecCtx);

    /* Free the codec. */
    av_free(pCodecCtx);
  
    return 0;  
}  


static int count_frames(int *gop_count, 
                        int *frame_count)
{
    AVCodec *pCodec = NULL;
    AVCodecContext *pCodecCtx = NULL;  
    AVCodecParserContext *pCodecParserCtx = NULL;  

    FILE *fp;

    uint8_t input_buffer[INPUT_BUFFER_SIZE + FF_INPUT_BUFFER_PADDING_SIZE];

    uint8_t *data;  
    size_t data_size;
    AVPacket packet;  
    int bytesParsed;

    /* Register all codecs. */
    avcodec_register_all();  

    /* Set end of buffer to 0 (this ensures that no overreading happens for damaged MPEG streams). */
    memset(input_buffer + INPUT_BUFFER_SIZE, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    av_init_packet(&packet);

    /* Find the MPEG-2 video decoder. */  
    pCodec = avcodec_find_decoder(AV_CODEC_ID_MPEG2VIDEO);  
    if (!pCodec)
        return fatal_error("Codec not found.");

    pCodecParserCtx = av_parser_init(AV_CODEC_ID_MPEG2VIDEO);  
    if (!pCodecParserCtx)  
        return fatal_error("Could not allocate video parser context.");  

    pCodecCtx = avcodec_alloc_context3(pCodec);  
    if (!pCodecCtx)
        return fatal_error("Could not allocate video codec context.");  

    /* Debug the DCT coefficients. */  
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "debug", "dct_coeff+mv", 0);

    /* Inform the codec that we can handle truncated bitstreams -- i.e.,
       bitstreams where frame boundaries can fall in the middle of packets.
     */
    if (pCodec->capabilities & CODEC_CAP_TRUNCATED)
        pCodecCtx->flags |= CODEC_FLAG_TRUNCATED;

    /* Open codec. */	      
    if (avcodec_open2(pCodecCtx, pCodec, &opts) < 0)
        return fatal_error("Could not open codec.");  

    /* Open input file. */
    fp = fopen(filename, "rb");  
    if (!fp)
        return fatal_error("Could not open file %s.", filename);  

    *gop_count = 0;
    *frame_count = 0;

    while (!feof(fp)) {
        /* Read raw data from the input file. */
        data_size = fread(input_buffer, 1, INPUT_BUFFER_SIZE, fp);  
        if (!data_size)  
            break;  
		
        /* Use the parser to split the data into frames. */
        data = input_buffer;  
        while (data_size > 0) {   
            bytesParsed = av_parser_parse2(pCodecParserCtx, pCodecCtx,  
                                           &packet.data, &packet.size,  
                                           data, data_size,  
                                           AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);  
            if (bytesParsed < 0)
                return fatal_error("Error while parsing.");
            data      += bytesParsed;  
            data_size -= bytesParsed;

            if (!packet.size)  
                continue;  

            if (pCodecParserCtx->pict_type == AV_PICTURE_TYPE_I)
                (*gop_count)++;
			
            (*frame_count)++;
        }
    }
  
    /* Close input file. */
    fclose(fp);

    /* Close the parser. */
    av_parser_close(pCodecParserCtx);  

    /* Close the codec. */
    avcodec_close(pCodecCtx);

    /* Free the codec. */
    av_free(pCodecCtx);
  
    return 0;
}


static PyObject *get_num_gops(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;

    int gop_count, frame_count;
    if(count_frames(&gop_count, &frame_count) < 0)
        printf("Decoding video failed.\n");
    return Py_BuildValue("i", gop_count);
}


static PyObject *get_num_frames(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;

    int gop_count, frame_count;
    if(count_frames(&gop_count, &frame_count) < 0)
        printf("Decoding video failed.\n");
    return Py_BuildValue("i", frame_count);
}


static PyObject *parse(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *npy_arr = NULL;
    int gop_index = 0;
    int frame_index = 0;
    int representation_type = DCT;

    static char *kwlist[] = {"fname",
                             "gop_index",
                             "frame_index",
                             "representation_type",
                             NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|iii", kwlist,
                                     &filename,
                                     &gop_index,
                                     &frame_index,
                                     &representation_type))
        return NULL;

    if(parse_video(&npy_arr,
                   gop_index,
                   frame_index,
                   representation_type) < 0) {
        printf("Decoding video failed.\n");

        Py_XDECREF(npy_arr);
        return Py_None;
    }

    return (PyObject *) npy_arr;									 
}


static PyMethodDef MPEG2Methods[] = {
    {"parse",  (PyCFunction) parse, METH_VARARGS | METH_KEYWORDS, "Read DCT coefficients or motion vectors of MPEG-2 video."},
    {"get_num_gops",  (PyCFunction) get_num_gops, METH_VARARGS, "Getting number of GOPs in a MPEG-2 video."},
    {"get_num_frames",  (PyCFunction) get_num_frames, METH_VARARGS, "Getting number of frames in a MPEG-2 video."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef mpeg2module = {
    PyModuleDef_HEAD_INIT,
    "mpeg2",  /* name of module */
    NULL,     /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    MPEG2Methods
};


PyMODINIT_FUNC PyInit_mpeg2(void)
{
    PyObject *m;

    m = PyModule_Create(&mpeg2module);
    if (m == NULL)
        return NULL;

    /* IMPORTANT: this must be called */
    import_array();

    MPEG2Error = PyErr_NewException("mpeg2.error", NULL, NULL);
    Py_INCREF(MPEG2Error);
    PyModule_AddObject(m, "error", MPEG2Error);
    return m;
}


int main(int argc, char *argv[])
{
    av_log_set_level(AV_LOG_QUIET);

    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("mpeg2", PyInit_mpeg2);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    PyMem_RawFree(program);
    return 0;
}
