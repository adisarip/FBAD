#include "common.h"
#include "hls_stream.h"



extern "C"{
	void AdaptiveThreshold(
			unsigned short width,
			unsigned short height,
			unsigned short size,
			int *src ,
			unsigned char *dst
			) {
		assert(width <= MAX_IMAGE_WIDTH) ;
		assert (height <= MAX_IMAGE_HEIGHT) ;
		assert(size <= MAX_SIZE) ;
		uint arr[MAX_SIZE][ MAX_IMAGE_WIDTH] ;
#pragma HLS ARRAY_PARTITION variable=arr type=complete dim=2

		arr[0][0] =src[0] ;
		Read_f_line:for (int i =1 ; i < width  ; i++){
#pragma HLS pipeline
			arr[0][i] = src[i] + arr[0][i-1];
		}

	}
}
