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
                                 IN  uint* integralImage,
                                 OUT uchar* dstImage)
{
    int x1, x2, y1, y2, area, sum;
    int T = 15; // Threshold value for comparison

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
        y1 = (y1 < 0) ? 0 : y1;
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
            y1 = (0 == y1) ? y1 : (y1-1);
            sum = (int)integralImage[y2*width + x2]
                - (int)integralImage[y1*width + x2]
                - (int)integralImage[y2*width + x1]
                + (int)integralImage[y1*width + x1];

            //cout << "(x1,y1)<>(x2,y2):" << "(" << x1 << "," << y1 << ")" << "<>(" << x2 << "," << y2 << ")" << endl;
            if ((int)(srcImage[i*width + j] * area) < (sum * (100 - T)/100))
            {
                dstImage[i*width + j] = 0;
                //cout << "[" << i*width + j << "] 0.A:" << (int)(srcImage[i*width + j] * area) << " | 0.B:" << (int)(sum * (1.0 - T)) << endl;
                if ((i*width + j) >= 20486 && (i*width + j) <= 20491)
                {
                    cout << "FPGA[" << i*width + j << "]"
                         << " | 0.A:" << (int)(srcImage[i*width + j] * area)
                         << " | 0.B:" << (sum * (100 - T)/100)
                         << " | sum=" << sum
                         //<< " | (x1,y1)<>(x2,y2):" << "(" << x1 << "," << y1 << ")" << "<>(" << x2 << "," << y2 << ")" << endl;
                         << " | II(x2,y2)=" << (int)integralImage[y2*width + x2]
                         << " | II(x2,y1)=" << (int)integralImage[y1*width + x2]
                         << " | II(x1,y2)=" << (int)integralImage[y2*width + x1]
                         << " | II(x1,y1)=" << (int)integralImage[y1*width + x1] << endl;
                }
            }
            else
            {
                dstImage[i*width + j] = 255;
                //cout << "[" << i*width + j << "] 255.A:" << (int)(srcImage[i*width + j] * area) << " | 255.B:" << (int)(sum * (1.0 - T)) << endl;
                if ((i*width + j) >= 20486 && (i*width + j) <= 20491)
                {
                    cout << "FPGA[" << i*width + j << "]"
                         << " | 255.A:" << (int)(srcImage[i*width + j] * area)
                         << " | 255.B:" << (sum * (100 - T)/100)
                         << " | sum=" << sum
                         //<< " | (x1,y1)<>(x2,y2):" << "(" << x1 << "," << y1 << ")" << "<>(" << x2 << "," << y2 << ")" << endl;
                         << " | II(x2,y2)=" << (int)integralImage[y2*width + x2]
                         << " | II(x2,y1)=" << (int)integralImage[y1*width + x2]
                         << " | II(x1,y2)=" << (int)integralImage[y2*width + x1]
                         << " | II(x1,y1)=" << (int)integralImage[y1*width + x1] << endl;
                }
            }
        }
    }
}

void computeIntegralImage(IN  uint width,
                          IN  uint height,
                          IN  const uchar* srcImage,
                          OUT uint* integralImage)
{
    integralImage[0] = srcImage[0];
    first_row_ii:
    for (uint j=1 ; j < width; j++)
    {
        #pragma HLS UNROLL factor=64
        integralImage[j] += integralImage[j-1] + srcImage[j];
    }

    remaining_rows_ii:
    uint sum = 0;
    for (uint i=1; i < height; i++)
    {
        #pragma HLS PIPELINE
        /*uint sum[MAX_IMAGE_WIDTH] = {0};
        // Compute all the individual row sums
        sum[0] = srcImage[i*width];
        compute_sum_vector:
        for (uint k=1; k<width; k++)
        {
            #pragma HLS UNROLL factor=64
            sum[k] = sum[k-1] + srcImage[i*width + k];
        }*/

        sum = 0;
        // Compute the Integral Image using the row sums
        compute_ii:
        for (uint j=0 ; j < width; j++)
        {
            #pragma HLS UNROLL factor=512
            sum = sum + srcImage[i*width + j];
            integralImage[i*width + j] = integralImage[(i-1)*width + j] + sum;
        }
        /*for (int x=0; x < height; x++)
        {
            cout << "FPGA:II["<<x<<"][76]: " << integralImage[x*width + 76] << endl;
        }*/
    }
}

extern "C"
{

void adaptiveThresholdingKernel(IN  uint width,
                                IN  uint height,
                                IN  uint size, // filter size of image
                                IN  uchar* srcImage,
                                OUT uchar* dstImage)
{
    assert(width <= MAX_IMAGE_WIDTH);
    assert (height <= MAX_IMAGE_HEIGHT);

    #pragma HLS DATAFLOW
    uint integralImage[MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT];
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
