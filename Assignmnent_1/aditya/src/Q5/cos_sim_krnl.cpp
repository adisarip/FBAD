

#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

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
        #pragma HLS INTERFACE m_axi port=d_vecs max_read_burst_length=32 offset=slave bundle=gmem2
        #pragma HLS INTERFACE m_axi port=c_vals max_write_burst_length=32 offset=slave bundle=gmem0
        #pragma HLS INTERFACE s_axilite port = q_vec bundle = control
        #pragma HLS INTERFACE s_axilite port = d_vecs bundle = control
        #pragma HLS INTERFACE s_axilite port = c_vals bundle = control
        #pragma HLS INTERFACE s_axilite port = n_vecs bundle = control
        #pragma HLS INTERFACE s_axilite port = return bundle = control

        float query_buffer[VECTOR_SIZE];

        // Burst reading the query vector into local memory
        // we need to do this only once
        read_query_vector:
        for (unsigned int j = 0; j < VECTOR_SIZE; j++)
        {
            #pragma HLS UNROLL
            query_buffer[j] = q_vec[j];
        }

        //Per iteration of this loop perform VECTOR_SIZE vector MAC operations
        for (unsigned int i = 0; i < n_vecs; i++)
        {
            #pragma HLS PIPELINE
            float dot_product = 0.0;
            compute_dot_product:
            for (unsigned int j = 0; j < VECTOR_SIZE; j++)
            {
                #pragma HLS PIPELINE
                dot_product += query_buffer[j] * d_vecs[i*VECTOR_SIZE +j];
            }
            c_vals[i] = dot_product;
        }
    }
}
