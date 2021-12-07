#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "at_common.h"
#include <iostream>
using namespace std;

void performAdaptiveThresholding(IN  uint width,
                                 IN  uint height,
                                 IN  uint filter_size,
                                 IN  const uchar* srcImage,
                                 IN  uint* intImage,
                                 OUT uchar* dstImage)
{
    int x1, x2, y1, y2, area, sum;
    int T = 15; // Threshold value for comparison

    for (uint i = 0; i < height; i++)
    {
        #pragma HLS PIPELINE
        #pragma HLS dependence variable=srcImage type=inter dependent=false
        #pragma HLS dependence variable=intImage type=inter dependent=false
        #pragma HLS dependence variable=dstImage type=inter dependent=false
        // initializing y-coordinates of compute area
        y1 = i - filter_size/2;
        y2 = i + filter_size/2;
        // checking boundaries
        y1 = (y1 < 0) ? 0 : y1;
        y2 = (y2 >= height) ? (height-1) : y2;
        y1 = (0 == y1) ? y1 : (y1-1);

        for (uint j=0; j < width; j++)
        {
            #pragma HLS UNROLL factor=64
            // initializing x-coordinates of compute area
            x1 = j - filter_size/2;
            x2 = j + filter_size/2;
            // checking boundaries
            x1 = (x1 < 0) ? 0 : x1;
            x2 = (x2 >= width) ? (width-1) : x2;

            // compute area of the rectangular region
            area = (x2-x1) * (y2-y1);

            // Computing the integral image for the rectangular region
            // I(x,y) = s(x2,y2) - s(x2,y1-1) - s(x1-1,y2) + s(x1-1,y1-1)
            x1 = (0 == x1) ? x1 : (x1-1);
            sum = (int)intImage[y2*(width+1) + x2]
                - (int)intImage[y1*(width+1) + x2]
                - (int)intImage[y2*(width+1) + x1]
                + (int)intImage[y1*(width+1) + x1];

            if ((int)(srcImage[i*width + j] * area) < (sum * (100 - T)/100))
            {
                dstImage[i*width + j] = 0;
            }
            else
            {
                dstImage[i*width + j] = 255;
            }
        }
    }
}

void computeIntegralImage(IN  uint width,
                          IN  uint height,
                          IN  const uchar* srcImage,
                          OUT uint* intImage)
{
    uint sum = 0;
    compute_ii:
    for (uint y=1; y<=height; y++)
    {
        #pragma HLS PIPELINE
        sum = 0;
        // padded column will be all zeros
        compute_paded_ii:
        for (uint x=1; x<=width; x++)
        {
            #pragma HLS UNROLL factor=64
            sum += srcImage[(y-1)*width + (x-1)];
            intImage[y*(width+1) + x] = intImage[(y-1)*(width+1) + x] + sum;
        }
    }
}

extern "C"
{

void adaptiveThresholdingKernel(IN  uint width,
                                IN  uint height,
                                IN  uint size, // filter size of image
                                IN  uchar* srcImage,
                                IN  uint*  intImage,
                                OUT uchar* dstImage)
{
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

    assert(width <= MAX_IMAGE_WIDTH);
    assert(height <= MAX_IMAGE_HEIGHT);

    computeIntegralImage(IN  width,
                         IN  height,
                         IN  srcImage,
                         OUT intImage);

    performAdaptiveThresholding(IN  width,
                                IN  height,
                                IN  size,
                                IN  srcImage,
                                IN  intImage,
                                OUT dstImage);
}

} // extern "C"
