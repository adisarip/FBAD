#include "at_common.h"
#include <assert.h>
#include <iostream>
#define MAX_FILTER_SIZE 137 // (1080 / 8) approx

extern "C"
{

void adaptiveThresholdingKernel(uint width,
                                uint height,
                                int size,
                                unsigned char *src,
                                unsigned char *dst)
{
    #pragma HLS INTERFACE m_axi port=src max_read_burst_length=64  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=dst max_write_burst_length=64 offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=src    bundle=control
    #pragma HLS INTERFACE s_axilite port=dst    bundle=control
    #pragma HLS INTERFACE s_axilite port=width  bundle=control
    #pragma HLS INTERFACE s_axilite port=height bundle=control
    #pragma HLS INTERFACE s_axilite port=size   bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    assert(width <= MAX_IMAGE_WIDTH);
    assert (height <= MAX_IMAGE_HEIGHT);
    assert(size <= MAX_FILTER_SIZE);

    int intImage[(2 + MAX_FILTER_SIZE) * (MAX_IMAGE_WIDTH + 1)];
    unsigned char ans[MAX_IMAGE_WIDTH];

    load_data:
    for(int i =0 ; i< size*width ; i++)
    {
        #pragma HLS PIPELINE
        #pragma HLS unroll factor=64
        intImage[i] = src[i] ;
    }

    first_ii:
    for (int i=1; i < width; i++)
    {
        intImage[i] +=intImage[ i-1];
    }

    rest_ii:
    for (int i=1; i < size; i++)
    {
        intImage[(i%(size+2)) * width] += intImage[((i-1)%(size+2))*width];
        compute_row_ii:
        for (int j=1; j < width; j++)
        {
            intImage[(i%(size+2))*width + j] += intImage[((i-1)%(size+2)) * width + j]
                                              + intImage[((i)%(size+2)) * width + (j-1)]
                                              - intImage[((i-1)%(size+2)) * width + (j-1)];
        }
    }

    starting_loop:
    for (int i= 0; i< size/2;i++)
    {
        compute_1:
        for (int j = 0 ; j < width; j++)
        {
            #pragma HLS UNROLL factor=1024
            #pragma HLS dependence variable=intImage type=inter dependent=false
            #pragma HLS dependence variable=ans type=inter dependent=false

            int x1, x2, y1, y2, area, sum;
            T = 15;
            x1 = (0 > j-size/2) ? 0 : (j-size/2);
            x2 = (width-1 < j+size/2) ? (width-1) : (j+size/2);
            y1 = (0 > i-size/2) ? 0 : (i-size/2);
            y2 = (height-1 < i+size/2) ? (height-1) : (i+size/2);

            y1 = (0 == y1) ? y1 : (y1-1);
            area = (x2-x1) * (y2-y1);
            x1 = (0 == x1) ? x1 : (x1-1);

            sum = intImage[(y2 % (size+2)) * width + x2]
                - intImage[(y1 % (size+2)) * width + x2]
                - intImage[(y2 % (size+2)) * width + x1]
                + intImage[(y1 % (size+2)) * width + x1];

            ans[j] = ((src[i*width + j] * area) < (sum * (100-T)/100)) ? 0 : 255;
        }

        transfer_out:
        for (int j=0; j < width; j++)
        {
            #pragma HLS PIPELINE
            #pragma HLS unroll factor=64
            dst[i*width+j] = ans[j] ;
        }
    }

    main_loop:
    for (int i=size/2; i < height; i++)
    {
        if ((i + (size+1)/2) < height)
        {
            transfer_in_2:
            for (int j=0; j < width; j++)
            {
                #pragma HLS PIPELINE
                #pragma HLS unroll factor=64
                intImage[((i + (size+1)/2)%(size+2)) * width + j] = src[(i+ (size+1)/2)*width + j];
            }

            intImage[((i + (size+1)/2)%(size+2)) * width] += intImage[(((i+(size+1)/2)-1)%(size+2)) * width];

            compute_row_ii_2:for (int j=1; j < width; j++)
            {
                intImage[((i + (size+1)/2)%(size+2)) * width + j] += intImage[(((i+(size+1)/2)-1)%(size+2)) * width + j]
                                                                   + intImage[((i + (size+1)/2)%(size+2)) * width + (j-1)]
                                                                   - intImage[(((i + (size+1)/2)-1)%(size+2)) * width + (j-1)];
            }
        }

        compute_2:
        for (int j = 0 ; j < width; j++)
        {
            #pragma HLS UNROLL factor=1024
            #pragma HLS dependence variable=intImage type=inter dependent=false
            #pragma HLS dependence variable=ans type=inter dependent=false

            int x1, x2, y1, y2, area, sum;
            T = 15;
            x1 = (0 > j-size/2) ? 0 : (j-size/2);
            x2 = (width-1 < j+size/2) ? (width-1) : (j+size/2);
            y1 = (0 > i-size/2) ? 0 : (i-size/2);
            y2 = (height-1 < i+size/2) ? (height-1) : (i+size/2);

            y1 = (0 >= y1) ? 0 : (y1-1);
            area = (x2-x1)*(y2-y1);
            x1 = (0 >= x1) ? 0 : (x1-1);

            sum = intImage[(y2 % (size+2)) * width + x2]
                - intImage[(y1 % (size+2)) * width + x2]
                - intImage[(y2 % (size+2)) * width + x1]
                + intImage[(y1 % (size+2)) * width + x1];

            ans[j] = ((src[i*width + j] * area) < (sum * (100-T)/100)) ? 0 : 255;
        }

        transfer_out_2:
        for (int j=0; j < width; j++)
        {
            #pragma HLS PIPELINE
            #pragma HLS unroll factor=64
            dst[i*width+j] = ans[j];
        }
    }
}

} // extern "C"
