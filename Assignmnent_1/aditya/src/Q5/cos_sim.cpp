
#include <iostream>
#include <random>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <memory>
#include <string>
#include <omp.h>

// Xilinx OpenCL and XRT includes
#include "xilinx_ocl.hpp"
#include "event_timer.hpp"

using namespace std;

#define VECTOR_SIZE 256
#define PAGE_SIZE   4096
#define NUM_BUFS    10

#define SUCCESS  0
#define FAILURE -1

void create_data_set(uint32_t n_vecs,
                     float* q_vec,
                     float* d_vecs)
{
    std::default_random_engine engine;
    std::uniform_real_distribution<float> dist(10, 25); // range 1 - 5
    srand(time(0));

    // create random normalized data vectors
    for (uint32_t i=0; i < n_vecs; i++)
    {
        float norm = 0.0;
        uint32_t offset = i * VECTOR_SIZE;
        for (uint32_t j=0; j < VECTOR_SIZE; j++)
        {
            d_vecs[offset + j] = dist(engine) * ((rand() % 2) ? 1 : -1);
            norm += d_vecs[offset + j] * d_vecs[offset + j];
        }
        // normalize
        for (uint32_t j=0; j < VECTOR_SIZE; j++)
        {
            d_vecs[offset + j] /= sqrt(norm);
        }
    }

    // create a random normalized query vector
    float norm = 0.0;
    for (uint32_t k=0; k<VECTOR_SIZE; k++)
    {
        q_vec[k] = dist(engine) * ((rand() % 2) ? 1 : -1);
        norm += q_vec[k] * q_vec[k];

    }
    // normalize
    for (uint32_t k=0; k<VECTOR_SIZE; k++)
    {
        q_vec[k] /= sqrt(norm);
    }
}

uint32_t compute_cosine_similarity(uint32_t n_vecs,
                                   float* q_vec,
                                   float* d_vecs,
                                   float* c_vals)
{
    // computing the cosine similarity values for each vector with the query vector
    // return the index of the vector close enough with the query vector
    float cosine_value = 0.0;
    uint32_t index = 0;
    #pragma omp parallel for
    for (uint32_t i=0; i < n_vecs; i++)
    {
        float dot_product = 0.0;
        for (uint32_t j=0; j < VECTOR_SIZE; j++)
        {
            dot_product += q_vec[j] * d_vecs[i*VECTOR_SIZE + j];
        }

        // save the index of the max cosine value as the last value
        // in the results data
        if (i==0 || cosine_value < dot_product)
        {
            index = i;
            cosine_value = dot_product;
        }
        c_vals[i] = dot_product;
    }
    return index;
}

// subdivide_data_buffer() should be called before this function
// It will set the num_divisions appropriately. We need to divide
// the cosine buffer into same number of divisions to maintain sync
// between both the buffers
int subdivide_cosine_buffer(std::vector<cl::Buffer> &divided_buf,
                            cl::Buffer buf_in,
                            cl_mem_flags flags,
                            uint32_t num_buffers,
                            uint32_t num_vectors_per_buffer)
{
    if (num_buffers == 0)
    {
        return FAILURE;
    }

    // Get the size of the buffer
    size_t size = buf_in.getInfo<CL_MEM_SIZE>();
    size_t element_size = sizeof(float); // 4 bytes
    uint32_t sub_buffer_size = (num_vectors_per_buffer * element_size);

    cl_buffer_region region;
    region.origin = 0;
    region.size = sub_buffer_size;

    for (uint32_t i = 0; i < num_buffers; i++)
    {
        if (i == num_buffers-1)
        {
            region.size = size - region.origin;
        }
        int err;
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

int subdivide_data_buffer(std::vector<cl::Buffer> &divided_buf,
                          cl::Buffer buf_in,
                          cl_mem_flags flags,
                          uint32_t num_divisions,
                          uint32_t& num_vectors_per_buffer)
{
    // Get the size of the buffer
    size_t size = buf_in.getInfo<CL_MEM_SIZE>();
    if (size  <= (num_divisions * PAGE_SIZE))
    {
        return FAILURE;
    }

    uint32_t num_pages = size / PAGE_SIZE;
    uint32_t num_pages_per_buffer = ((num_pages-1)/num_divisions) + 1;
    uint32_t sub_buffer_size = (num_pages_per_buffer * PAGE_SIZE);
    uint32_t num_divs = (size-1) / (sub_buffer_size) + 1;
    num_vectors_per_buffer = sub_buffer_size / (VECTOR_SIZE * sizeof(float));

    cl_buffer_region region;
    region.origin = 0;
    region.size = sub_buffer_size;

    for (uint32_t i = 0; i < num_divs; i++)
    {
        if (i == num_divs-1)
        {
            region.size = size - region.origin;
        }
        int err;
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
                        cl::Buffer q_buf,
                        cl::Buffer d_buf,
                        cl::Buffer c_buf)
{
    // Get the size of the buffer
    cl::Event k_event, m_event;
    std::vector<cl::Event> krnl_events;

    static std::vector<cl::Event> tx_events, rx_events;

    std::vector<cl::Memory> in_vec;
    in_vec.push_back(q_buf);
    in_vec.push_back(d_buf);

    q.enqueueMigrateMemObjects(in_vec, 0, &tx_events, &m_event);
    krnl_events.push_back(m_event);
    tx_events.push_back(m_event);

    if (tx_events.size() > 1)
    {
        tx_events[0] = tx_events[1];
        tx_events.pop_back();
    }

    size_t size;
    size = c_buf.getInfo<CL_MEM_SIZE>();
    // No of Vectors is 1 less than the c_buf size. The reason being we added
    // that one additional last entry to store the index of the matched query.
    uint32_t n_vectors = size / sizeof(float);
    krnl.setArg(0, q_buf);
    krnl.setArg(1, d_buf);
    krnl.setArg(2, c_buf);
    krnl.setArg(3, n_vectors);

    q.enqueueTask(krnl, &krnl_events, &k_event);
    krnl_events.push_back(k_event);

    if (rx_events.size() == 1)
    {
        krnl_events.push_back(rx_events[0]);
        rx_events.pop_back();
    }

    std::vector<cl::Memory> c_vec;
    c_vec.push_back(c_buf);
    q.enqueueMigrateMemObjects(c_vec,
                               CL_MIGRATE_MEM_OBJECT_HOST,
                               &krnl_events,
                               &event);
    rx_events.push_back(event);

    return 0;
}

int main(int argc, char* argv[])
{
    // Check if the binary file & no of elements are passed as arguments
    if (argc != 3)
    {
      cout << "[INFO] Usage: " << argv[0] << " <XCLBIN File> <num_elements>" << endl;
      return EXIT_FAILURE;
    }

    // Copy xclbin binary filename
    char* binaryName = argv[1];
    // default values
    uint32_t num_vectors = 0;
    uint32_t num_elements = 0;
    try
    {
        num_vectors = stoi(argv[2]);
        num_elements = num_vectors * VECTOR_SIZE;
    }
    catch (const invalid_argument val)
    {
        cerr << "[ERROR] Invalid argument in position 2 (" << argv[2] << ") program expects an integer as number of elements" << endl;
        return EXIT_FAILURE;
    }
    catch (const out_of_range val)
    {
        cerr << "[ERROR] Number of elements out of range, try with a number lower than 2147483648" << endl;
        return EXIT_FAILURE;
    }

    cout << "[INFO] No Of Vectors: " << num_vectors << endl;
    cout << "[INFO] No Of Elements: " << num_elements << endl;

    // Initialize the runtime (including a command queue) and load the FPGA image
    cout << "[INFO] Loading " << binaryName << " to program the board." << endl;

    // This application will use the first Xilinx device found in the system
    swm::XilinxOcl xocl;
    xocl.initialize(binaryName);

    cl::CommandQueue q = xocl.get_command_queue();
    cl::Kernel krnl = xocl.get_kernel("cos_sim_krnl");

    // Initialize an event timer we'll use for monitoring the application
    EventTimer et;

    try
    {
        // Map our user-allocated buffers as OpenCL buffers
        cl_mem_ext_ptr_t bank0_ext = {0};
        cl_mem_ext_ptr_t bank2_ext = {0};
        bank0_ext.flags = 0 | XCL_MEM_TOPOLOGY;
        bank0_ext.obj   = NULL;
        bank0_ext.param = NULL;
        bank2_ext.flags = 2 | XCL_MEM_TOPOLOGY;
        bank2_ext.obj   = NULL;
        bank2_ext.param = NULL;

        cl::Buffer q_vec_buf(xocl.get_context(),
                             static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX),
                             VECTOR_SIZE * sizeof(float),
                             &bank0_ext,
                             NULL);
        cl::Buffer d_vec_buf(xocl.get_context(),
                             static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX),
                             num_elements * sizeof(float),
                             &bank2_ext,
                             NULL);
        cl::Buffer c_val_buf(xocl.get_context(),
                             static_cast<cl_mem_flags>(CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX),
                             num_vectors * sizeof(float),
                             &bank0_ext,
                             NULL);

        // Although we'll change these later, we'll set the buffers as kernel
        // arguments prior to mapping so that XRT can resolve the physical memory
        // in which they need to be allocated
        krnl.setArg(0, q_vec_buf);
        krnl.setArg(1, d_vec_buf);
        krnl.setArg(2, c_val_buf);

        float* query_vector = (float*)q.enqueueMapBuffer(q_vec_buf,
                                                         CL_TRUE,
                                                         CL_MAP_WRITE,
                                                         0,
                                                         VECTOR_SIZE * sizeof(float));
        float* data_vectors = (float*)q.enqueueMapBuffer(d_vec_buf,
                                                         CL_TRUE,
                                                         CL_MAP_WRITE,
                                                         0,
                                                         num_elements * sizeof(float));
        float* hw_cosine_values = (float*)q.enqueueMapBuffer(c_val_buf,
                                                             CL_TRUE,
                                                             CL_MAP_READ,
                                                             0,
                                                             num_vectors * sizeof(float));

        et.add("[ET] Create Data Set");
        create_data_set(num_vectors,
                        query_vector,
                        data_vectors);

        float* sw_cosine_values = new float[num_vectors];
        et.add("[ET] Compute Cosine Similarity on CPU");
        uint32_t matched_vector_index = compute_cosine_similarity(num_vectors,
                                                                  query_vector,
                                                                  data_vectors,
                                                                  sw_cosine_values);
        et.finish();

        // Subdividing the buffers and executing the kernel
        int sRc = SUCCESS;
        uint32_t num_buffers = 0;
        uint32_t num_vectors_per_buffer = 0;
        std::vector<cl::Buffer> d_vec_bufs, c_val_bufs;

        et.add("[ET] Subdividing Buffers");
        sRc = subdivide_data_buffer(d_vec_bufs,
                                    d_vec_buf,
                                    CL_MEM_READ_ONLY,
                                    NUM_BUFS,
                                    num_vectors_per_buffer);
        if (SUCCESS == sRc)
        {
            num_buffers = d_vec_bufs.size();
            sRc = subdivide_cosine_buffer(c_val_bufs,
                                          c_val_buf,
                                          CL_MEM_WRITE_ONLY,
                                          num_buffers,
                                          num_vectors_per_buffer);
        }
        et.finish();

        if (num_buffers == 0)
        {
            // the buffer size is too small to divide into sub buffers
            et.add("[ET] Send/Execute/Receive buffers");
            cl::Event kernel_event;
            enqueue_subbuf_vadd(q,
                                krnl,
                                kernel_event,
                                q_vec_buf,
                                d_vec_buf,
                                c_val_buf);

            et.add("[ET] Wait for the kernel to complete");
            clWaitForEvents(1, (const cl_event *)&kernel_event);
            et.finish();
        }
        else
        {
            et.add("[ET] Send/Execute/Receive sub buffers");
            //std::array<cl::Event, NUM_BUFS> kernel_events;
            cl::Event kernel_events[num_buffers];
            for (uint32_t i = 0; i < num_buffers; i++)
            {
                enqueue_subbuf_vadd(q,
                                    krnl,
                                    kernel_events[i],
                                    q_vec_buf,
                                    d_vec_bufs[i],
                                    c_val_bufs[i]);
            }
            et.add("[ET] Wait for kernels to complete");
            clWaitForEvents(num_buffers, (const cl_event *)&kernel_events);
            et.finish();
        }

        // Verify the results
        bool verified = true;
        for (uint32_t i = 0; i < num_vectors; i++)
        {
            if (sw_cosine_values[i] != hw_cosine_values[i])
            {
                verified = false;
                cout << "ERROR: software and hardware cosine values do not match: "
                     << sw_cosine_values[i] << "!=" << hw_cosine_values[i]
                     << " at position " << i << endl;
                break;
            }
        }

        if (verified)
        {
            cout << "[INFO] Cosine Similarity experiment completed successfully!" << endl;
            cout << "[INFO] Max cosine value [SW]: [" << matched_vector_index <<  "] : "
                        << sw_cosine_values[matched_vector_index] << endl;
            cout << "[INFO] Max cosine value [HW]: [" << matched_vector_index <<  "] : "
                        << hw_cosine_values[matched_vector_index] << endl;
        }
        else
        {
            cout << "[ERROR] Cosine Similarity experiment completed with errors!" << endl;
        }

        cout << "--------------- Key execution times ---------------" << endl;

        q.enqueueUnmapMemObject(q_vec_buf, query_vector);
        q.enqueueUnmapMemObject(d_vec_buf, data_vectors);
        q.enqueueUnmapMemObject(c_val_buf, hw_cosine_values);
        free(sw_cosine_values);
        q.finish();
        et.print();
    }
    catch(cl::Error &err)
    {
        cout << "[ERROR] " << err.what() << endl;
        return EXIT_FAILURE;
    }


    return 0;
}
