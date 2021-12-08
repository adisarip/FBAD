#include "at_common.h"
#include <assert.h>

#define MAX_SIZE 137

void compute(uint intImage[MAX_SIZE][MAX_IMAGE_WIDTH +1 ],unsigned char *src  ,uint height,uint width, int i , uint size , unsigned char ans[MAX_IMAGE_WIDTH]){
	uint x1,  x2 ,y1 , y2, area , sum ,  T = 15 ;
    
	for (uint j = 0 ; j < width; j++){
#pragma HLS UNROLL factor=512
#pragma HLS dependence variable=intImage type=inter dependent=false
#pragma HLS dependence variable=ans type=inter dependent=false

		x1 = 0> j-size/2? 0: j-size/2;
		x2= width-1< j+size /2 ? width-1 : j+ size/2  ;
		y1 = 0> i-size/2? 0: i-size/2;
		y2= height-1< i+size /2 ? height-1 : i+ size/2  ;
        y1 = (0 == y1) ? y1 : (y1-1);

		area = (x2-x1)*(y2-y1) ;
        x1 = (0 == x1) ? x1 : (x1-1);
        sum = intImage[y2%size][x2 ]
			-intImage[y1%size][x2 ]
			- intImage[y2%size][x1 ]
			+ intImage[y1%size][x1 ];
		if (src[i*width + j]*area < sum* (100 - T) /100 ){
			ans[j]  = 0 ;
		}else {
			ans[j] = 255;
		}

	}
}

void intergralimage(uint intImage[MAX_SIZE][MAX_IMAGE_WIDTH+1 ], uint width ,int i ,uint size ){
#pragma HLS PIPELINE
	intImage[i%size][0] += intImage[(i-1)%size ][0] ;// split in two sum and above addition .
	Row:for (int j = 1 ; j < width ; j++){
		intImage[i%size][j] += intImage[(i-1)%size][j] -intImage[(i-1)%size][j-1];
	}
}

extern "C"{
	void adaptiveThresholdingKernel(
			uint width,
			uint height,
			uint size,
			unsigned char *src ,
			unsigned char *dst
			) {
    #pragma HLS INTERFACE m_axi port=srcImage max_read_burst_length=64  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=intImage max_write_burst_length=64 offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=dstImage max_write_burst_length=64 offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port = srcImage bundle = control
    #pragma HLS INTERFACE s_axilite port = intImage bundle = control
    #pragma HLS INTERFACE s_axilite port = dstImage bundle = control
    #pragma HLS INTERFACE s_axilite port = width    bundle = control
    #pragma HLS INTERFACE s_axilite port = height   bundle = control
    #pragma HLS INTERFACE s_axilite port = size     bundle = control
    #pragma HLS INTERFACE s_axilite port = return   bundle = control
		assert(width <= MAX_IMAGE_WIDTH) ;
		assert (height <= MAX_IMAGE_HEIGHT) ;
		assert(size <= MAX_SIZE) ;
		uint intImage[MAX_SIZE][ MAX_IMAGE_WIDTH +1 ] ;
        
		unsigned char ans[MAX_IMAGE_WIDTH] ;
#pragma HLS intImageAY_PARTITION variable=intImage type=complete dim=2

		Load_data:for(uint i =0 ; i< size*width ; i++){// break into i j.
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
			intImage[i/width][i%width] = src[i] ;
		}

		First_II:for (uint i =1 ; i < width; i ++){
			intImage[0][i] +=intImage[0][i-1];
		}

		Rest_II:for (uint i=1 ;i < size; i++ ){
			intergralimage(intImage, width, i, size);
		}

		Starting_loop:for (uint i= 0; i< size/2;i++){
			compute(intImage, height, width, i, size, ans);
			Transfer_out:for (uint j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				dst[i*width+j] = ans[j] ;
			}
		}
		Main_loop:for (uint i= size/2; i< height;i++){

			Transfer_in2:for (uint j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				intImage[i%size][j] = src[i*width + j ] ;
			}

			intergralimage(intImage, width, i, size);
			compute(intImage, height, width, i, size, ans);

			Transfer_out2:for (uint j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				dst[i*width+j] = ans[j] ;
			}
		}

	}
}
