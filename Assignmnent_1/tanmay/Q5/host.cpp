#include <vector>
#include <random>
#include "xcl2.hpp"
#include <algorithm>
// #include <cstring>
#include <string>
#include <iostream>
#include <chrono>

using namespace std;

int getans(float *data , float *query , int N ){
	int ans = 0;
	float val = 0 ;
	std::cout<< N << std::endl ;
	for (int i =0 ; i< N ;i++){
		float tem = 0 ;
		for (int j= 0 ; j < 256 ; j++){
			tem += data[i* 256 + j ] * query[j] ;
		}
		if (tem  >  val){
			val = tem;
			ans = i ;
		}
		//std::cout << tem << std::endl;
	}
	return ans ;
}

int subdivide_buffer(std::vector<cl::Buffer> &data_bufs, cl::Buffer data_in,
		std::vector<cl::Buffer>&res_bufs , cl::Buffer res_in , uint &num_div , int N){
	num_div = (N +511)/512;
	cl_buffer_region r1, r2 ;
	int err;
	r1.origin = 0  ;
	r2.origin = 0 ;

	r1.size = 256 * 4*16;
	r2.size = 4 *16 ;
	for(uint i =0 ;i< num_div  ; i++){
		if (i == num_div -1 ){
			r1.size =data_in.getInfo<CL_MEM_SIZE>() - i*16*4*256;
			r2.size =res_in.getInfo<CL_MEM_SIZE>() - i*16*4;
		}
		data_bufs.push_back(data_in.createSubBuffer(static_cast<cl_mem_flags>(CL_MEM_READ_ONLY) , CL_BUFFER_CREATE_TYPE_REGION, &r1, &err));
		if (err != CL_SUCCESS){
			exit(-1) ;
		}
		res_bufs.push_back(res_in.createSubBuffer(static_cast<cl_mem_flags>(CL_MEM_WRITE_ONLY), CL_BUFFER_CREATE_TYPE_REGION, &r2, &err));
		if (err != CL_SUCCESS){
			exit(-1) ;
		}
		r1.origin += r1.size;
		r2.origin += r2.size ;
	}
	return 0 ;
}

int enqueue_subbuf_vadd(cl::CommandQueue &q, cl::Kernel &krnl,
		cl::Event &event, cl::Buffer a, cl::Buffer b, cl::Buffer c)
{
    // Get the size of the buffer
    cl::Event k_event, m_event;
    std::vector<cl::Event> krnl_events;

    static std::vector<cl::Event> tx_events, rx_events;
    std::vector<cl::Memory> c_vec;
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
    krnl.setArg(3, c);
    krnl.setArg(2, (int)(size / (4 * 256)));
    q.enqueueTask(krnl, &krnl_events, &k_event);
    krnl_events.push_back(k_event);
    if (rx_events.size() == 1)
    {
        krnl_events.push_back(rx_events[0]);
        rx_events.pop_back();
    }
    c_vec.push_back(c);
    q.enqueueMigrateMemObjects(c_vec, CL_MIGRATE_MEM_OBJECT_HOST, &krnl_events, &event);
    rx_events.push_back(event);

    return 0;
}

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
            return -1;
        }
    }
    else
    {
        N = 256;
    }


    cl::Device device;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    cl::Kernel krnl;
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
            OCL_CHECK(err, krnl = cl::Kernel(program, "matrix_multiply", &err));
            // valid_device = true;
            break; // we break because we found a valid device
        }
    }

    cl::Buffer data_buf(context,
					 static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					 N*256 * sizeof(float),
					 NULL,
					 NULL);
	cl::Buffer query_buf(context,
					 static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					 256 * sizeof(float),
					 NULL,
					 NULL);
	cl::Buffer res_buf(context,
					 static_cast<cl_mem_flags>(CL_MEM_READ_WRITE),
					 N* sizeof(float),
					 NULL,
					 NULL);
    krnl.setArg(0, data_buf);
	krnl.setArg(1, query_buf);
	krnl.setArg(2 ,N)  ;
	krnl.setArg(3, res_buf);
    float *data = (float *)q.enqueueMapBuffer(data_buf,
												 CL_TRUE,
												 CL_MAP_WRITE,
												 0,
												 N*256 * sizeof(float));
	float *query = (float *)q.enqueueMapBuffer(query_buf,
												 CL_TRUE,
												 CL_MAP_WRITE,
												 0,
												 256* sizeof(float));


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
    	query[j] /= temp_sum;
    }

    std::chrono::high_resolution_clock::time_point st = std::chrono::high_resolution_clock::now() ;
    int ans_val = getans(data , query , N);
    std::chrono::duration<float, std::milli> duration = std::chrono::high_resolution_clock::now() - st;


    std::cout << "CPU time:\t" << duration.count()<< " ms "<<endl;
    q.enqueueUnmapMemObject(data_buf, data);
	q.enqueueUnmapMemObject(query_buf,query);

	std::vector<cl::Buffer> data_bufs , res_bufs;
	uint num_div;
	subdivide_buffer(data_bufs, data_buf, res_bufs, res_buf, num_div, N);
	cl::Event  kernel_events[num_div];

	for (uint i = 0; i < num_div; i++)
	{
		cl::Buffer t2 =data_bufs[i];
		cl::Buffer te = res_bufs[i]  ;
		enqueue_subbuf_vadd(q, krnl, kernel_events[i], t2 , query_buf, te);
	}
	st = std::chrono::high_resolution_clock::now() ;
	clWaitForEvents(num_div, (const cl_event *)&kernel_events);
	duration = std::chrono::high_resolution_clock::now() - st;


	    std::cout << "FPGA time:\t" << duration.count()<< " ms "<<endl;
	int ker_ans =0 , tem_ans =0  ;
	float *res = (float *)q.enqueueMapBuffer(res_buf,
												 CL_TRUE,
												 CL_MAP_READ,
												 0,
												 N* sizeof(float));
	//calculate ans
	for (int i= 0 ; i< N ; i++){
		if (tem_ans < res[i]){
			tem_ans = res[i] ;
			ker_ans = i ;
		}
	}

	q.enqueueUnmapMemObject(res_buf , res) ;

    std::cout<< "Finish \n" ;
    return 0  ; 
}
