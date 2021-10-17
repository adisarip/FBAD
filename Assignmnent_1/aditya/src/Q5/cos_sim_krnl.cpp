

#include <stdio.h>
#include <string.h>

#define VECTOR_SIZE 256

/*
    Kernel implementing cosine similarity
    Arguments:
        in1   (input)     --> Input Vector1
        in2   (input)     --> Input Vector2
        out   (output)    --> Output Vector
        size  (input)     --> Size of Vector in Integer
   */
extern "C"
{
    void cos_sim_krnl(float* q_vec,        // Query Vector : size = 256
                      float* d_vecs,       // Data Vectors : size = batch-size
                      float* c_vals,       // cosine values
                      unsigned int n_vecs) // number of vectors
    {
        #pragma HLS INTERFACE m_axi port=q_vec max_read_burst_length=32 offset=slave bundle=gmem0
        #pragma HLS INTERFACE m_axi port=q_vec max_read_burst_length=32 offset=slave bundle=gmem1
        #pragma HLS INTERFACE m_axi port=c_vals max_write_burst_length=32 offset=slave bundle=gmem0
        #pragma HLS INTERFACE s_axilite port = q_vec bundle = control
        #pragma HLS INTERFACE s_axilite port = d_vecs bundle = control
        #pragma HLS INTERFACE s_axilite port = c_vals bundle = control
        #pragma HLS INTERFACE s_axilite port = n_vecs bundle = control
        #pragma HLS INTERFACE s_axilite port = return bundle = control

        float query_buffer[VECTOR_SIZE];
        float vector_buffer[VECTOR_SIZE];

        // Burst reading the query vector into local memory
        read_query_vector:
        for (unsigned int j = 0; j < VECTOR_SIZE; j++)
        {
            #pragma HLS UNROLL
            query_buffer[j] = q_vec[j];
        }

        //Per iteration of this loop perform VECTOR_SIZE vector MAC operations
        for (unsigned int i = 0; i < n_vecs; i++)
        {
            #pragma HLS DATAFLOW
            float cosine_value = 0;

            read_data_vector:
            for (unsigned int j = 0; j < VECTOR_SIZE; j++)
            {
                #pragma HLS UNROLL
                vector_buffer[j] = d_vecs[i*VECTOR_SIZE + j];
            }

            compute_dot_product:
            for (unsigned int j = 0; j < VECTOR_SIZE; j++)
            {
                #pragma HLS PIPELINE
                cosine_value += query_buffer[j] * vector_buffer[j];
            }
            c_vals[i] = cosine_value;
        }
    }
}
