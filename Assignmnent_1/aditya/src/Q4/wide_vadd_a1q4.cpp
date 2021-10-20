/**********
Copyright (c) 2019-2020, Xilinx, Inc.
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


#include "event_timer.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <cstdlib>

// OpenMP
#include <omp.h>

// Xilinx OpenCL and XRT includes
#include "xilinx_ocl.hpp"
using namespace std;

// page size in bytes
#define PAGE_SIZE 4096
#define NUM_BUFS  10

#define SUCCESS  0
#define FAILURE -1

// Software vadd
void vadd_sw(uint32_t *a,
             uint32_t *b,
             uint32_t *c,
             uint32_t size)
{
    #pragma omp parallel for
    for (uint32_t i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int subdivide_buffer(std::vector<cl::Buffer> &divided_buf,
                     cl::Buffer buf_in,
                     cl_mem_flags flags,
                     unsigned int num_divisions)
{
    // Get the size of the buffer
    size_t size;
    size = buf_in.getInfo<CL_MEM_SIZE>();

    if (size  <= (num_divisions * PAGE_SIZE))
    {
        return FAILURE;
    }
    //cout << "Size : " << size << " > " << (num_divisions * PAGE_SIZE) << endl;

    int num_pages = size / PAGE_SIZE;
    unsigned int num_pages_per_buffer = ((num_pages-1)/10) + 1;
    unsigned int sub_buffer_size = (num_pages_per_buffer * PAGE_SIZE);
    unsigned int num_divs = (size-1) / (sub_buffer_size) + 1;
    //cout << "num_pages = " << num_pages << " | c = " << num_pages_per_buffer
    //     << " | buffer_size = " << buffer_size << " | divs = " << num_divs << endl;

    cl_buffer_region region;

    int err;
    region.origin = 0;
    region.size   = sub_buffer_size;

    for (unsigned int i = 0; i < num_divs; i++)
    {
        if (i == num_divs-1)
        {
            region.size = size - region.origin;
        }
        //cout << "[" << i << "]"  "region.origin = " << region.origin << " | " << region.size << endl;
        cl::Buffer buf = buf_in.createSubBuffer(flags,
                                                CL_BUFFER_CREATE_TYPE_REGION,
                                                &region,
                                                &err);
        if (err != CL_SUCCESS)
        {
            return err;
        }
        divided_buf.push_back(buf);
        region.origin += region.size;
    }
    return SUCCESS;
}

int enqueue_subbuf_vadd(cl::CommandQueue &q,
                        cl::Kernel &krnl,
                        cl::Event &event,
                        cl::Buffer a,
                        cl::Buffer b,
                        cl::Buffer c)
{
    // Get the size of the buffer
    cl::Event k_event, m_event;
    std::vector<cl::Event> krnl_events;

    static std::vector<cl::Event> tx_events, rx_events;
    size_t size;
    size = a.getInfo<CL_MEM_SIZE>();

    std::vector<cl::Memory> in_vec;
    in_vec.push_back(a);
    in_vec.push_back(b);

    q.enqueueMigrateMemObjects(in_vec, 0, &tx_events, &m_event);
    krnl_events.push_back(m_event);
    tx_events.push_back(m_event);

    if (tx_events.size() > 1)
    {
        tx_events[0] = tx_events[1];
        tx_events.pop_back();
    }

    krnl.setArg(0, a);
    krnl.setArg(1, b);
    krnl.setArg(2, c);
    krnl.setArg(3, (uint32_t)(size / sizeof(uint32_t)));

    q.enqueueTask(krnl, &krnl_events, &k_event);
    krnl_events.push_back(k_event);

    if (rx_events.size() == 1)
    {
        krnl_events.push_back(rx_events[0]);
        rx_events.pop_back();
    }

    std::vector<cl::Memory> c_vec;
    c_vec.push_back(c);
    q.enqueueMigrateMemObjects(c_vec,
                               CL_MIGRATE_MEM_OBJECT_HOST,
                               &krnl_events,
                               &event);
    rx_events.push_back(event);

    return 0;
}

int main(int argc, char *argv[])
{
    // Initialize an event timer we'll use for monitoring the application
    EventTimer et;

    // Check if the binary file & no of elements are passed as arguments
    if (argc != 3)
    {
      cout << "Usage: " << argv[0] << " <XCLBIN File> <num_elements>" << endl;
      return EXIT_FAILURE;
    }

    // Copy binary name
    char* binaryName = argv[1];

    uint32_t num_elements = 4096; // default
    try
    {
        num_elements = stoi(argv[2]);
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

    cout << "-- Parallelizing the Data Path --" << endl << endl;

    // Initialize the runtime (including a command queue) and load the FPGA image
    cout << "Loading " << binaryName << " to program the board." << endl << endl;

    // This application will use the first Xilinx device found in the system
    swm::XilinxOcl xocl;
    xocl.initialize(binaryName);

    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl = xocl.get_kernel("wide_vadd_a1q4_krnl");

    /// New code for example 01
    cout << "Running kernel test XRT-allocated buffers and wide data path:" << endl << endl;

    try
    {
        // Map our user-allocated buffers as OpenCL buffers using a shared host pointer
        cl_mem_ext_ptr_t bank0_ext = {0};
        cl_mem_ext_ptr_t bank2_ext = {0};
        bank0_ext.flags = 0 | XCL_MEM_TOPOLOGY;
        bank0_ext.obj   = NULL;
        bank0_ext.param = NULL;
        bank2_ext.flags = 2 | XCL_MEM_TOPOLOGY;
        bank2_ext.obj   = NULL;
        bank2_ext.param = NULL;

        cl::Buffer a_buf(xocl.get_context(),
                         static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX),
                         num_elements * sizeof(uint32_t),
                         &bank0_ext,
                         NULL);
        cl::Buffer b_buf(xocl.get_context(),
                         static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX),
                         num_elements * sizeof(uint32_t),
                         &bank2_ext,
                         NULL);
        cl::Buffer c_buf(xocl.get_context(),
                         static_cast<cl_mem_flags>(CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX),
                         num_elements * sizeof(uint32_t),
                         &bank0_ext,
                         NULL);

        uint32_t *d = new uint32_t[num_elements];

        // Although we'll change these later, we'll set the buffers as kernel
        // arguments prior to mapping so that XRT can resolve the physical memory
        // in which they need to be allocated
        krnl.setArg(0, a_buf);
        krnl.setArg(1, b_buf);
        krnl.setArg(2, c_buf);

        uint32_t *a = (uint32_t *)q.enqueueMapBuffer(a_buf,
                                                     CL_TRUE,
                                                     CL_MAP_WRITE,
                                                     0,
                                                     num_elements * sizeof(uint32_t));
        uint32_t *b = (uint32_t *)q.enqueueMapBuffer(b_buf,
                                                     CL_TRUE,
                                                     CL_MAP_WRITE,
                                                     0,
                                                     num_elements * sizeof(uint32_t));
        uint32_t *c = (uint32_t *)q.enqueueMapBuffer(c_buf,
                                                     CL_TRUE,
                                                     CL_MAP_READ,
                                                     0,
                                                     num_elements * sizeof(uint32_t));

        for (uint32_t i = 0; i < num_elements; i++)
        {
            a[i] = i;
            b[i] = 2 * i;
        }

        // For comparison, let's have the CPU calculate the result
        et.add("Software VADD run");
        vadd_sw(a, b, d, num_elements);
        et.finish();

        q.enqueueUnmapMemObject(a_buf, a);
        q.enqueueUnmapMemObject(b_buf, b);

        // Subdividing the buffers and executing the kernel
        int sRc = SUCCESS;
        std::vector<cl::Buffer> a_bufs, b_bufs, c_bufs;
        et.add("Subdividing Buffers");
        sRc = subdivide_buffer(a_bufs, a_buf, CL_MEM_READ_ONLY, NUM_BUFS);
        sRc |= subdivide_buffer(b_bufs, b_buf, CL_MEM_READ_ONLY, NUM_BUFS);
        sRc |= subdivide_buffer(c_bufs, c_buf, CL_MEM_WRITE_ONLY, NUM_BUFS);
        et.finish();

        unsigned int num_buffers = a_bufs.size();

        if (num_buffers == 0)
        {
            // the buffer size is too small to divide into sub buffers
            et.add("Send/Execute/Receive Sub buffers");
            cl::Event kernel_event;
            enqueue_subbuf_vadd(q,
                                krnl,
                                kernel_event,
                                a_buf,
                                b_buf,
                                c_buf);

            et.add("Wait for the kernel to complete");
            clWaitForEvents(1, (const cl_event *)&kernel_event);
            et.finish();
        }
        else
        {
            et.add("Send/Execute/Receive Sub buffers");
            //std::array<cl::Event, NUM_BUFS> kernel_events;
            cl::Event kernel_events[num_buffers];
            for (unsigned int i = 0; i < num_buffers; i++)
            {
                //std::array<cl::Event, NUM_BUFS+1> kernel_events;
                enqueue_subbuf_vadd(q,
                                    krnl,
                                    kernel_events[i],
                                    a_bufs[i],
                                    b_bufs[i],
                                    c_bufs[i]);
            }
            et.add("Wait for kernels to complete");
            clWaitForEvents(num_buffers, (const cl_event *)&kernel_events);
            et.finish();
        }

        // Verify the results
        bool verified = true;
        for (uint32_t i = 0; i < num_elements; i++)
        {
            if (c[i] != d[i])
            {
                verified = false;
                cout << "ERROR: software and hardware vadd do not match: "
                     << c[i] << "!=" << d[i] << " at position " << i << endl;
                break;
            }
        }

        if (verified)
        {
            cout << endl << "OCL-mapped contiguous buffer example complete successfully!" << endl << endl;
        }
        else
        {
            cout << endl << "OCL-mapped contiguous buffer example complete! (with errors)" << endl << endl;
        }

        cout << "--------------- Key execution times ---------------" << endl;

        q.enqueueUnmapMemObject(c_buf, c);
        free(d);
        q.finish();

        et.print();

    }
    catch(cl::Error &err)
    {
        cout << "ERROR: " << err.what() << endl;
        return EXIT_FAILURE;
    }
}
