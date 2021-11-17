#include "common.h"
#include "hls_stream.h"

void compute(uint arr[MAX_SIZE][MAX_IMAGE_WIDTH] ,uint height,uint width, int i , uint size , unsigned char ans[MAX_IMAGE_WIDTH]){
	uint x1,  x2 ,y1 , y2, area;
	for (uint j = 0 ; j < width; j++){
#pragma HLS UNROLL factor=512
#pragma HLS dependence variable=arr type=inter dependent=false
#pragma HLS dependence variable=ans type=inter dependent=false

		x1 = 0> j-size/2? 0: j-size/2;
		x2= width-1< j+size /2 ? width-1 : j+ size/2  ;
		y1 = 0> i-size/2? 0: i-size/2;
		y2= height-1< i+size /2 ? height-1 : i+ size/2  ;
		area = (x2-x1)*(y2-y1)/2 ;
		ans[j]=(arr[x2%size][y2] - (y1 > 0? arr[x2%size][y1-1]:0) - (x1>0?arr[(x1-1)%size][y2] : 0) + ((y1 > 0) & ( x1>0)?arr[(x1-1)%size][y1-1]:0)) > area? 255 : 0 ;

	}
}

void intergralimage(uint arr[MAX_SIZE][MAX_IMAGE_WIDTH], uint width ,int i ,uint size ){
#pragma HLS PIPELINE
	arr[i%size][0] += arr[(i-1)%size ][0] ;// split in two sum and above addition .
	Row:for (int j = 1 ; j < width ; j++){
		arr[i%size][j] += arr[(i-1)%size][j] -arr[(i-1)%size][j-1];
	}
}

extern "C"{
	void AdaptiveThreshold(
			uint width,
			uint height,
			uint size,
			unsigned char *src ,
			unsigned char *dst
			) {
		assert(width <= MAX_IMAGE_WIDTH) ;
		assert (height <= MAX_IMAGE_HEIGHT) ;
		assert(size <= MAX_SIZE) ;
		uint arr[MAX_SIZE][ MAX_IMAGE_WIDTH] ;
		unsigned char ans[MAX_IMAGE_WIDTH] ;
#pragma HLS ARRAY_PARTITION variable=arr type=complete dim=2

		Load_data:for(uint i =0 ; i< size*width ; i++){// break into i j.
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
			arr[i/width][i%width] = src[i] ;
		}

		First_II:for (uint i =1 ; i < width; i ++){
			arr[0][i] +=arr[0][i-1];
		}

		Rest_II:for (uint i=1 ;i < size; i++ ){
			intergralimage(arr, width, i, size);
		}

		Starting_loop:for (uint i= 0; i< size/2;i++){
#pragma HLS dataflow
			compute(arr, height, width, i, size, ans);
			Transfer_out:for (uint j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				dst[i*width+j] = ans[j] ;
			}
		}
		Main_loop:for (uint i= size/2; i< height;i++){
#pragma HLS dataflow
			Transfer_in2:for (uint j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				arr[i%size][j] = src[i*width + j ] ;
			}
			intergralimage(arr, width, i, size);
			compute(arr, height, width, i, size, ans);
			Transfer_out2:for (uint j= 0; j < width; j++ ){
#pragma HLS PIPELINE
#pragma HLS unroll factor=64
				dst[i*width+j] = ans[j] ;
			}
		}

	}
}
