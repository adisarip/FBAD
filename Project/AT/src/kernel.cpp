#include "common.h"
#include "hls_stream.h"


extern "C" {

void AdaptiveThreshold(unsigned short width,
                       unsigned short height,
                       unsigned short size,
                       int *src ,
                       unsigned char *dst)
{
    assert(width <= MAX_IMAGE_WIDTH);
    assert (height <= MAX_IMAGE_HEIGHT);
    assert(size <= MAX_SIZE);
    uint arr[MAX_SIZE][ MAX_IMAGE_WIDTH] ;

    #pragma HLS ARRAY_PARTITION variable=arr type=complete dim=2
    Load_data:
    for(int i=0; i < size*width; i++)
    {
        #pragma HLS PIPELINE
        arr[i/width][i%width] = src[i] ;
    }
}

} // extern C
