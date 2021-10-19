/**********
Copyright (c) 2020, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/


#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <iomanip>
#include <chrono>
#include "CL/cl_ext_xilinx.h" // This extension file is required for stream APIs
#include "xcl2.hpp"           // This file is required for OpenCL C++ wrapper APIs
using namespace std;

void vectors_init(int *buffer_a,
                  int *buffer_b,
                  int *sw_results,
                  int *hw_results,
                  unsigned int num_elements)
{
    // Fill the input vectors with random data
    for (size_t i = 0; i < num_elements; i++)
    {
        buffer_a[i] = rand() * ((rand() % 2) ? 1 : -1);
        buffer_b[i] = rand() * ((rand() % 2) ? 1 : -1);
        hw_results[i] = 0;
    }
}

bool verify(int *sw_results,
            int *hw_results,
            int num_elements)
{
    bool match = true;
    for (int i = 0; i < num_elements; i++)
    {
        if (sw_results[i] != hw_results[i])
        {
            match = false;
            break;
        }
    }
    cout << "TEST " << (match ? "PASSED" : "FAILED") << endl;
    return match;
}

int main(int argc, char **argv)
{
    // Check input arguments
    if (argc < 2 || argc > 4)
    {
        cout << "Usage: " << argv[0] << " <XCLBIN File> <#elements(optional)> <debug(optional)>" << endl;
        return EXIT_FAILURE;
    }

    // Read FPGA binary file
    auto binaryFile = argv[1];
    unsigned int num_elements = 4096;
    bool user_size = false;
    // Check if the user defined the # of elements
    if (argc >= 3)
    {
        user_size = true;
        unsigned int val;
        try
        {
            val = stoi(argv[2]);
        }
        catch (const invalid_argument val)
        {
            cerr << "Invalid argument in position 2 (" << argv[2] << ") program expects an integer as number of elements" << endl;
            return EXIT_FAILURE;
        }
        catch (const out_of_range val)
        {
            cerr << "Number of elements out of range, try with a number lower than 2147483648" << endl;
            return EXIT_FAILURE;
        }
        num_elements = val;
        cout << "User number of elements enabled" << endl;
    }

    bool debug = false;
    // Check if the user defined debug
    if (argc == 4)
    {
        string debug_arg = argv[3];
        if(debug_arg.compare("debug") == 0)
        {
            debug = true;
        }
        cout << "Debug enabled" << endl;
    }

    if (!user_size)
    {
        // Define number of num_elements
        if (xcl::is_hw_emulation())
        {
            num_elements = 4096;
        }
        else if (xcl::is_emulation())
        {
            num_elements = 4096 * 8;
        }
        else
        {
            num_elements = 4096 * 4096;
        }
    }

    // I/O Data Vectors
    vector<int, aligned_allocator<int>> buffer_a(num_elements);
    vector<int, aligned_allocator<int>> buffer_b(num_elements);
    vector<int, aligned_allocator<int>> hw_results(num_elements);
    vector<int, aligned_allocator<int>> sw_results(num_elements);

    // OpenCL Host Code Begins.
    // OpenCL objects
    cl::Device device;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    cl::Kernel krnl_vadd;
    cl_int err;

    // get_xil_devices() is a utility API which will find the Xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++)
    {
      device = devices[i];
      // Creating Context and Command Queue for selected Device
      OCL_CHECK(err, context = cl::Context(device,
                                           NULL,
                                           NULL,
                                           NULL,
                                           &err));

      OCL_CHECK(err, q = cl::CommandQueue(context,
                                          device,
                                          CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                          &err));

      cout << "Trying to program device[" << i << "]: "
                << device.getInfo<CL_DEVICE_NAME>() << endl;

      cl::Program program(context,
                          {device},
                          bins,
                          NULL,
                          &err);

        if (err != CL_SUCCESS)
        {
            cout << "Failed to program device[" << i << "] with xclbin file!\n";
        }
        else
        {
            cout << "Device[" << i << "]: program successful!\n";
            // Creating Kernel
            OCL_CHECK(err, krnl_vadd  = cl::Kernel(program,
                                                   "krnl_vadd",
                                                   &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }

    if (!valid_device)
    {
        cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    cout << "Running Vector add with " << num_elements << " elements" << endl;

    // Initialize the data vectors
    vectors_init(buffer_a.data(),
                 buffer_b.data(),
                 sw_results.data(),
                 hw_results.data(),
                 num_elements);

    // perform vector addition on the CPU
    auto sw_start = chrono::steady_clock::now();
    for (size_t i = 0; i < num_elements; i++)
    {
        sw_results[i] = buffer_a[i] * buffer_b[i]; // changed to multiplication temporarily
    }
    auto sw_end = chrono::steady_clock::now();

    // Now running it on the kernel
    unsigned int size_bytes  = num_elements * sizeof(int);

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_input1(context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            size_bytes,
                                            buffer_a.data(),
                                            &err));

    OCL_CHECK(err, cl::Buffer buffer_input2(context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                            size_bytes,
                                            buffer_b.data(),
                                            &err));

    OCL_CHECK(err, cl::Buffer buffer_output(context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                            size_bytes,
                                            hw_results.data(),
                                            &err));

    // Setting Kernel Arguments krnl_vadd
    OCL_CHECK(err, err = krnl_vadd.setArg(0, buffer_input1));
    OCL_CHECK(err, err = krnl_vadd.setArg(1, buffer_input2));
    OCL_CHECK(err, err = krnl_vadd.setArg(2, buffer_output));
    OCL_CHECK(err, err = krnl_vadd.setArg(3, num_elements));

    // Copy input data to device global memory
    auto comm_us_start = chrono::steady_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input1, buffer_input2}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q.finish());
    auto comm_us_end = chrono::steady_clock::now();

    // Launching the Kernels
    cout << "Launching Hardware Kernel..." << endl;
    auto hw_start = chrono::steady_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_vadd));
    // wait for the kernel to finish their operations
    OCL_CHECK(err, err = q.finish());
    auto hw_end = chrono::steady_clock::now();


    // Copy Result from Device Global Memory to Host Local Memory
    cout << "Getting Hardware Results..." << endl;
    auto comm_ds_start = chrono::steady_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());
    auto comm_ds_end = chrono::steady_clock::now();

    // OpenCL Host Code Ends

    // Compare the device results with software results
    bool match = verify(sw_results.data(),
                        hw_results.data(),
                        num_elements);

    cout << "CPU Execution Time: "
         << chrono::duration_cast<chrono::microseconds>(sw_end - sw_start).count()
         << "µs" << endl;

    cout << "Kernel Execution Time: "
         << chrono::duration_cast<chrono::microseconds>(hw_end - hw_start).count()
         << "µs" << endl;

    cout << "Data Transfer Time: "
         << chrono::duration_cast<chrono::milliseconds>(comm_us_end - comm_us_start + comm_ds_end - comm_ds_start).count()
         << "ms" << endl;

    if (debug)
    {
        for (unsigned int i = 0 ; i < num_elements; i++)
        {
            cout << "Idx [" << setw(6) << i << "]" << setw(14) << buffer_a[i] << " + ";
            cout << setw(14) << buffer_b[i] <<"\tsw result" << setw(14);
            cout << sw_results[i] << "\thw result" << setw(14) << hw_results[i];
            cout << "\tequal "<< ((hw_results[i] == sw_results[i]) ? "True" : "False") << endl;
        }
    }

    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
