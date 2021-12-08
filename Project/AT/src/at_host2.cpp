
// Xilinx OpenCL and XRT includes
#include "xilinx_ocl.h"
#include "event_timer.h"
#include "at_common.h"
#include <iostream>
#include <memory>
#include <string>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;

void adaptiveThresholdingHost(cv::Mat &inputMat,
                              cv::Mat &outputMat)
{
    // accept only char type matrices
    CV_Assert(!inputMat.empty());
    CV_Assert(inputMat.depth() == CV_8U);
    CV_Assert(inputMat.channels() == 1);

    CV_Assert(!outputMat.empty());
    CV_Assert(outputMat.depth() == CV_8U);
    CV_Assert(outputMat.channels() == 1);

    // no of rows == height == y-coordinate
    // no of cols == width  == x-coordinate
    int nRows = inputMat.rows;
    int nCols = inputMat.cols;

    // Create the integral image
    cv::Mat integralMat;
    cv::integral(inputMat, integralMat);

    CV_Assert(integralMat.depth() == CV_32S);
    CV_Assert(sizeof(int) == 4);

    // Values for image filter/kernel sizes & Threshold size.
    // Values taken based on the paper.
    int S = MAX(nRows, nCols) / 8;
    int T = 15;

    // Perform adaptive thresholding
    int s2 = S / 2;
    int x1, y1, x2, y2, area, sum;
    int *p_y1, *p_y2;
    uchar *p_inputMat, *p_outputMat;

    for (int i = 0; i < nRows; ++i)
    {
        y1 = i - s2;
        y2 = i + s2;
        y1 = (y1 < 0) ? 0 : y1;
        y2 = (y2 >= nRows) ? (nRows - 1) : y2;

        y1 = (0 == y1) ? y1 : (y1-1);
        p_y1 = integralMat.ptr<int>(y1);
        p_y2 = integralMat.ptr<int>(y2);
        p_inputMat  = inputMat.ptr<uchar>(i);
        p_outputMat = outputMat.ptr<uchar>(i);

        for (int j = 0; j < nCols; ++j)
        {
            // Set the SxS region
            x1 = j - s2;
            x2 = j + s2;
            x1 = (x1 < 0) ? 0 : x1;
            x2 = (x2 >= nCols) ? (nCols-1) : x2;

            // compute the area of the SxS rectangular region
            area = (x2 - x1) * (y2 - y1);
            // Computing the integral image for the rectangular region (x1,y1) to (x2,y2)
            // I(x,y) = s(x2,y2) - s(x2,y1-1) - s(x1-1,y2) + s(x1-1,y1-1)
            x1 = (0 == x1) ? x1 : (x1-1);
            sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];

            if ((int)(p_inputMat[j] * area) < (sum * (100 - T)/100))
            {
                p_outputMat[j] = 0;
            }
            else
            {
                p_outputMat[j] = 255;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    // Check if the binary file & input image file are passed as arguments
    if (argc != 3)
    {
      cout << "[INFO] Usage: " << argv[0] << " <XCLBIN File> <input_image_file>" << endl;
      return EXIT_FAILURE;
    }

    // Copy xclbin binary filename
    char* binaryName = argv[1];

    // Load the image
    cv::Mat src;
    try
    {
        src = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    }
    catch(cl::Error &err)
    {
        cout << "[ERROR] " << err.what() << endl;
        cout << "[ERROR] Invalid argument in position 2 (" << argv[2] << ") program expects an image file" << endl;
        return EXIT_FAILURE;
    }

    // Check if image is loaded fine
    if (src.empty())
    {
        cout << "[ERROR] Problem loading image!!!" << endl;
        return EXIT_FAILURE;
    }

    // Transform source image to gray
    cv::Mat grayed_image;
    if (src.channels() == 3)
    {
        // save the gray image
        cv::cvtColor(src, grayed_image, cv::COLOR_BGR2GRAY);
        cv::imwrite("image_grayed.jpg", grayed_image);
    }
    else
    {
        grayed_image = src;
    }

    cv::Mat fpga_grayed_image = grayed_image.clone();

    // This application will use the first Xilinx device found in the system
    swm::XilinxOcl xocl;
    xocl.initialize(binaryName);

    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl = xocl.get_kernel("adaptiveThresholdingKernel");

    // Initialize an event timer we'll use for monitoring the application
    EventTimer et;

    try
    {
        // HOST: Computing the adaptive threshold of the image
        et.add("[ET] HOST: Perform Adaptive Thresholding");
        cv::Mat host_at_image = cv::Mat::zeros(grayed_image.size(), CV_8UC1);
        adaptiveThresholdingHost(grayed_image, host_at_image);
        cv::imwrite("host_at_image.jpg", host_at_image);
        et.finish();

        // FPGA: computing the adaptive threshold of the image
        // Map our user-allocated buffers as OpenCL buffers
        uint height = grayed_image.rows;
        uint width  = grayed_image.cols;
        uint filter_size = MAX(width, height) / 8;

        cl_mem_ext_ptr_t bank0_ext = {0};
        cl_mem_ext_ptr_t bank2_ext = {0};
        bank0_ext.flags = 0 | XCL_MEM_TOPOLOGY;
        bank0_ext.obj   = NULL;
        bank0_ext.param = NULL;
        bank2_ext.flags = 2 | XCL_MEM_TOPOLOGY;
        bank2_ext.obj   = NULL;
        bank2_ext.param = NULL;

        cl::Buffer src_image_buf(xocl.get_context(),
                                 static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX),
                                 width * height * sizeof(uchar),
                                 &bank0_ext,
                                 NULL);
        /*cl::Buffer int_image_buf(xocl.get_context(),
                                 static_cast<cl_mem_flags>(CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX),
                                 (width+1) * (height+1) * sizeof(uint),
                                 &bank2_ext,
                                 NULL);*/
        cl::Buffer dst_image_buf(xocl.get_context(),
                                 static_cast<cl_mem_flags>(CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX),
                                 width * height * sizeof(uchar),
                                 &bank2_ext,
                                 NULL);

        // Although we'll change these later, we'll set the buffers as kernel
        // arguments prior to mapping so that XRT can resolve the physical memory
        // in which they need to be allocated
        krnl.setArg(0, width);
        krnl.setArg(1, height);
        krnl.setArg(2, filter_size);
        krnl.setArg(3, src_image_buf);
        //krnl.setArg(4, int_image_buf);
        krnl.setArg(4, dst_image_buf);

        uchar* src_image = (uchar*)q.enqueueMapBuffer(src_image_buf,
                                                      CL_TRUE,
                                                      CL_MAP_WRITE,
                                                      0,
                                                      width * height * sizeof(uchar));
        /*uchar* int_image = (uchar*)q.enqueueMapBuffer(int_image_buf,
                                                      CL_TRUE,
                                                      CL_MAP_WRITE,
                                                      0,
                                                      (width+1) * (height+1) * sizeof(uint));*/
        // Now load the input image into src_image as a byte-stream
        memcpy(src_image, fpga_grayed_image.data, height * width);
        /*memset(int_image, 0, (width+1) * (height+1) * sizeof(uint));*/

        // Send the image buffers down to the FPGA card
        et.add("[ET] Memory object migration enqueue");
        cl::Event event_sp;
        q.enqueueMigrateMemObjects({src_image_buf, dst_image_buf}, 0, NULL, &event_sp);
        clWaitForEvents(1, (const cl_event*)&event_sp);

        et.add("[ET] OCL Enqueue task");
        q.enqueueTask(krnl, NULL, &event_sp);
        et.add("[ET] Wait for kernel to complete");
        clWaitForEvents(1, (const cl_event*)&event_sp);
        et.finish();

        uchar* dst_image = (uchar*)q.enqueueMapBuffer(dst_image_buf,
                                                      CL_TRUE,
                                                      CL_MAP_READ,
                                                      0,
                                                      width * height * sizeof(uchar));

        // Now write the output byte-stream as an jpeg image
        cv::Mat fpga_at_image(height, width, CV_8UC1, &dst_image[0]);
        cv::imwrite("fpga_at_image.jpg", fpga_at_image);

        q.enqueueUnmapMemObject(src_image_buf, src_image);
        q.enqueueUnmapMemObject(dst_image_buf, dst_image);
        q.finish();

        cout << "--------------- Key execution times ---------------" << endl;
        et.print();
    }
    catch(cl::Error &err)
    {
        cout << "[ERROR] " << err.what() << endl;
        return EXIT_FAILURE;
    }

    return 0;
}
