#include <vector>
#include <random>
#include "xcl2.hpp"
#include <algorithm>
// #include <cstring>
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 4)
    {
        cout << "Incorrect format" << endl;
        return EXIT_FAILURE;
    }
    char *binaryFile = argv[1];
    int N  ;
    if (argc == 3)
    {
        try
        {
            N = stoi(argv[2]);
        }
        catch (invalid_argument val)
        {
            cerr << "Invalid argument" << endl;
        }
    }
    else
    {
        N = 256;
    }
    
    vector<float, aligned_allocator<float>> data(N * 256);
    vector<float, aligned_allocator<float>> query(256);
    
    vector<int, aligned_allocator<int>> res(1);

    cl::Device device;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    cl::Kernel krnl_fir;
    cl_int err;

    auto devices = xcl::get_xil_devices();

    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    // bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++)
    {
        device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(context, device,
                                       CL_QUEUE_PROFILING_ENABLE |
                                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                       &err));
        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        }
        else
        {
            std::cout << "Device[" << i << "]: program successful!\n";
            // Creating Kernel
            OCL_CHECK(err, krnl_fir = cl::Kernel(program, "matrix_multiply", &err));
            // valid_device = true;
            break; // we break because we found a valid device
        }
    }

    // Filling the data
    float temp_sum = 0;
    for (int i = 0; i < N; i++)
    {
        temp_sum = 0 ; 
        for (int j = 0; j < 256; j++)
        {
            data[i * 256 + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            temp_sum += data[i * 256 + j] * data[i * 256 + j];
        }
        temp_sum = sqrt(temp_sum);
        for (int j = 0; j < 256; j++)
        {
            data[i * 256 + j] /= temp_sum;
        }
    }
    temp_sum = 0 ; 
    for (int j = 0; j < 256; j++)
    {
        query[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        temp_sum +=query[j]*query[j];
    }
    temp_sum = sqrt(temp_sum);
    for (int j = 0; j < 256; j++)
    {
        data[j] /= temp_sum;
    }

    OCL_CHECK(err, cl::Buffer buffer_input1(context,
                                           CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           N*256*sizeof(float), data.data(), &err));
    
    OCL_CHECK(err, cl::Buffer buffer_input2 (context,
                                           CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           Q * 256 * sizeof(float), query.data(), &err));
    
    OCL_CHECK(err, cl::Buffer buffer_output (context,
                                            CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                            Q*sizeof(int), res.data(), &err));
    OCL_CHECK(err, err = krnl_fir.setArg( 0 ,buffer_input1 ) ) ; 
    OCL_CHECK(err, err = krnl_fir.setArg(1 , buffer_input2 )) ; 
    OCL_CHECK(err , err = krnl_fir.setArg(2 , N) ) ; 
    OCL_CHECK(err , err = krnl_fir.setArg(3 , buffer_output)) ; 
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input1, buffer_input2}, 0 /* 0 means from host*/));
    OCL_CHECK(err, err = q.finish());
    // Launching the Kernels
    OCL_CHECK(err, err = q.enqueueTask(krnl_fir));
    // wait for the kernel to finish their operations
    OCL_CHECK(err, err = q.finish());
    
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());
    
    cout<< << "Finish \n" ; 
    return 0  ; 
}