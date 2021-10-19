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
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include "xcl2.hpp"
#include <climits>
#include <sys/stat.h>
#include <unistd.h>
using namespace std;

namespace xcl
{
    vector<cl::Device> get_devices(const string &vendor_name)
    {
        size_t i;
        cl_int err;
        vector<cl::Platform> platforms;
        OCL_CHECK(err, err = cl::Platform::get(&platforms));
        cl::Platform platform;

        for (i = 0; i < platforms.size(); i++)
        {
            platform = platforms[i];
            OCL_CHECK(err, string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
            if (platformName == vendor_name)
            {
                cout << "Found Platform" << endl;
                cout << "Platform Name: " << platformName.c_str() << endl;
                break;
            }
        }

        if (i == platforms.size())
        {
            cout << "Error: Failed to find Xilinx platform" << endl;
            exit(EXIT_FAILURE);
        }

        // Getting ACCELERATOR Devices and selecting 1st such device
        vector<cl::Device> devices;
        OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
        return devices;
    }

    vector<cl::Device> get_xil_devices()
    {
        return get_devices("Xilinx");
    }

    vector<unsigned char> read_binary_file(const string &xclbin_file_name)
    {
        cout << "INFO: Reading " << xclbin_file_name << endl;
        FILE *fp;
        if ((fp = fopen(xclbin_file_name.c_str(), "r")) == NULL)
        {
            printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
            exit(EXIT_FAILURE);
        }

        // Loading XCL Bin into char buffer
        cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";

        ifstream bin_file(xclbin_file_name.c_str(), ifstream::binary);
        bin_file.seekg(0, bin_file.end);
        auto nb = bin_file.tellg();
        bin_file.seekg(0, bin_file.beg);
        vector<unsigned char> buf;
        buf.resize(nb);
        bin_file.read(reinterpret_cast<char *>(buf.data()), nb);
        return buf;
    }

    bool is_emulation()
    {
        bool ret = false;
        char *xcl_mode = getenv("XCL_EMULATION_MODE");
        if (xcl_mode != NULL)
        {
            ret = true;
        }
        return ret;
    }

    bool is_hw_emulation()
    {
        bool ret = false;
        char *xcl_mode = getenv("XCL_EMULATION_MODE");
        if ((xcl_mode != NULL) && !strcmp(xcl_mode, "hw_emu"))
        {
            ret = true;
        }
        return ret;
    }

    bool is_xpr_device(const char *device_name)
    {
        const char *output = strstr(device_name, "xpr");
        if (output == NULL)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

}; // namespace xcl
