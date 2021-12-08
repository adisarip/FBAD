#include "at_common.h"
#include <assert.h>

#define MAX_SIZE 137

void compute(int *intImage ,unsigned char *src  ,int height,int width, int i , int size , unsigned char ans[MAX_IMAGE_WIDTH]){
	int x1,  x2 ,y1 , y2, area , sum ,  T = 15 ;
    
	for (int j = 0 ; j < width; j++){
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

        sum = intImage[(y2%size)*width+ x2 ]
			- intImage[(y1%size)*width+ x2 ]
			- intImage[(y2%size)*width+ x1 ]
			+ intImage[(y1%size)*width+ x1 ];
		if (src[i*width + j]*area < sum* (100 - T) /100 ){
			ans[j]  = 0 ;
		}else {
			ans[j] = 255;
		}

	}
}

void intergralimage(int *intImage , int width ,int i ,int size ){
#pragma HLS PIPELINE
	intImage[(i%size)*width ] += intImage[((i-1)%size)*width ] ;// split in two sum and above addition .
	Row:for (int j = 1 ; j < width ; j++){
		intImage[(i%size)*width +  j] += intImage[((i-1)%size) *width+j] -intImage[((i-1)%size) *width+j-1];
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
    #pragma HLS INTERFACE m_axi port=src max_read_burst_length=64  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=dst max_write_burst_length=64 offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port = src bundle = control
    #pragma HLS INTERFACE s_axilite port = dst bundle = control
    #pragma HLS INTERFACE s_axilite port = width    bundle = control
    #pragma HLS INTERFACE s_axilite port = height   bundle = control
    #pragma HLS INTERFACE s_axilite port = size     bundle = control
    #pragma HLS INTERFACE s_axilite port = return   bundle = control
		assert(width <= MAX_IMAGE_WIDTH) ;
		assert (height <= MAX_IMAGE_HEIGHT) ;
		assert(size <= MAX_SIZE) ;
		int intImage[MAX_SIZE*( MAX_IMAGE_WIDTH +1) ] ;
        
		unsigned char ans[MAX_IMAGE_WIDTH] ;
#pragma HLS intImageAY_PARTITION variable=intImage type=complete dim=2

		Load_data:for(int i =0 ; i< size*width ; i++){// break into i j.
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
			intImage[i] = src[i] ;
		}

		First_II:for (int i =1 ; i < width; i ++){
			intImage[i] +=intImage[ i-1];
		}

		Rest_II:for (int i=1 ;i < size; i++ ){
			intergralimage(intImage, width, i, size);
		}

		Starting_loop:for (int i= 0; i< size/2;i++){
			compute(intImage,src, height, width, i, size, ans);
			Transfer_out:for (int j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				dst[i*width+j] = ans[j] ;
			}
		}
		Main_loop:for (int i= size/2; i< height;i++){

			Transfer_in2:for (int j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				intImage[(i%size)*width +j] = src[i*width + j ] ;
			}

			intergralimage(intImage, width, i, size);
			compute(intImage,src, height, width, i, size, ans);

			Transfer_out2:for (int j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				dst[i*width+j] = ans[j] ;
			}
		}

	}
}
