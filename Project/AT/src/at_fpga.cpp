#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "at_common.h"
#include "hls_stream.h"

void performAdaptiveThresholding(IN  uint width,
                                 IN  uint height,
                                 IN  uint filter_size,
                                 IN  const uint8* srcImage,
                                 IN  uint8* integralImage,
                                 OUT uint8* dstImage)
{
    int x1, x2, y1, y2, area, sum;
    float T = 0.15; // Threshold value for comparison

    for (uint i = 0; i < height; i++)
    {
        #pragma HLS PIPELINE
        #pragma HLS UNROLL factor=64
        #pragma HLS dependence variable=integralImage type=inter dependent=false
        #pragma HLS dependence variable=dstImage type=inter dependent=false
        // initializing y-coordinates of compute area
        y1 = i - filter_size/2;
        y2 = i + filter_size/2;
        // checking boundaries
        y1 = (y1 < 0) ? 0: y1;
        y2 = (y2 > height-1) ? (height-1) : y2;

        for (uint j=0; j < width; j++)
        {
            #pragma HLS UNROLL factor=64
            // initializing x-coordinates of compute area
            x1 = j - filter_size/2;
            x2 = j + filter_size/2;
            // checking boundaries
            x1 = (x1 < 0) ? 0 : x1;
            x2 = (x2 > width-1) ? (width-1) : x2;

            // compute area of the rectangular region
            area = (x2-x1) * (y2-y1);

            // Computing the integral image for the rectangular region
            // I(x,y) = s(x2,y2) - s(x2,y1-1) - s(x1-1,y2) + s(x1-1,y1-1)
            x1 = (0 == x1) ? x1 : (x1-1);
            y1 = (0 == x1) ? y1 : (y1-1);
            sum = integralImage[x2*width + y2]
                - integralImage[x2*width + y1]
                - integralImage[x1*width + y2]
                + integralImage[x1*width + y1];

            dstImage[i*width + j] = ((srcImage[i*width + j] * area) < (sum * (1.0 - T))) ? 0 : 255;
        }
    }
}

void computeIntegralImage(IN  uint width,
                          IN  uint height,
                          IN  const uint8* srcImage,
                          OUT uint8* integralImage)
{
    integralImage[0] = srcImage[0];
    first_row_ii:
    for (uint8 j=1 ; j < width; j++)
    {
        #pragma HLS UNROLL factor=64
        integralImage[j] += integralImage[j-1] + srcImage[j];
    }

    int sum[IMAGE_WIDTH];
    remaining_rows_ii:
    for (uint8 i=1; i < height; i++)
    {
        #pragma HLS PIPELINE

        // Compute all the individual row sums
        sum[0] = srcImage[i*width];
        compute_sum_vector:
        for (uint8 k=1; k<width; k++)
        {
            #pragma HLS UNROLL factor=64
            sum[k] = sum[k-1] + srcImage[i*width + k];
        }

        // Compute the Integral Image using the row sums
        compute_ii:
        for (uint8 j=0 ; j < width; j++)
        {
            #pragma HLS UNROLL factor=512
            integralImage[i*width + j] = integralImage[(i-1)*width + j] + sum[j];
        }
    }
}

extern "C"
{

void adaptiveThresholdingKernel(IN  uint width,
                                IN  uint height,
                                IN  uint size, // filter size of image
                                IN  uint8* srcImage,
                                OUT uint8* dstImage)
{
    assert(width <= IMAGE_WIDTH);
    assert (height <= IMAGE_HEIGHT);
    assert(size <= FILTER_SIZE);

    #pragma HLS DATAFLOW
    uint8 integralImage[IMAGE_WIDTH * IMAGE_HEIGHT];
    computeIntegralImage(IN  width,
                         IN  height,
                         IN  srcImage,
                         OUT integralImage);

    performAdaptiveThresholding(IN  width,
                                IN  height,
                                IN  size,
                                IN  srcImage,
                                IN  integralImage,
                                OUT dstImage);
}

} // extern "C"
