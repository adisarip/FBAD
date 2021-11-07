/**********
Copyright (c) 2018-2020, Xilinx, Inc.
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

/*******************************************************************************
Description:
    Wide Memory Access Example using ap_uint<Width> datatype
    Description: This is vector addition example to demonstrate Wide Memory
    access of 512bit Datawidth using ap_uint<> datatype which is defined inside
    'ap_int.h' file.
*******************************************************************************/

#include <stdio.h>
#include <string.h>

#define BUFFER_SIZE 64

/*
    Vector Addition Kernel Implementation using uint512_dt datatype
    Arguments:
        in1   (input)     --> Input Vector1
        in2   (input)     --> Input Vector2
        out   (output)    --> Output Vector
        size  (input)     --> Size of Vector in Integer
   */
extern "C"
{
    void wide_vmul(const float* in1,  // Read-Only Vector 1
                   const float* in2,  // Read-Only Vector 2
                   float* out,        // Output Result
                   unsigned int size) // Size in integer
    {
        #pragma HLS INTERFACE m_axi port = in1 max_read_burst_length = 32  offset = slave bundle = gmem0
        #pragma HLS INTERFACE m_axi port = in2 max_read_burst_length = 32  offset = slave bundle = gmem1
        #pragma HLS INTERFACE m_axi port = out max_write_burst_length = 32 offset = slave bundle = gmem2
        #pragma HLS INTERFACE s_axilite port = in1 bundle = control
        #pragma HLS INTERFACE s_axilite port = in2 bundle = control
        #pragma HLS INTERFACE s_axilite port = out bundle = control
        #pragma HLS INTERFACE s_axilite port = size bundle = control
        #pragma HLS INTERFACE s_axilite port = return bundle = control

        float v1_local[BUFFER_SIZE]; // Local memory to store vector1
        float v2_local[BUFFER_SIZE]; // Local memory to store vector2

        //Per iteration of this loop perform BUFFER_SIZE vector addition
        for (unsigned int i = 0; i < size; i += BUFFER_SIZE)
        {
            //#pragma HLS PIPELINE
            #pragma HLS DATAFLOW
            #pragma HLS stream variable = v1_local depth = 64
            #pragma HLS stream variable = v2_local depth = 64

            unsigned int chunk_size = ((i + BUFFER_SIZE) > size) ? (size - i) : BUFFER_SIZE;

            v1_rd:
            for (unsigned int j = 0; j < chunk_size; j++)
            {
                #pragma HLS PIPELINE
                v1_local[j] = in1[i + j];
                v2_local[j] = in2[i + j];
            }

            v2_rd_mul:
            for (int j = 0; j < chunk_size; j++)
            {
                #pragma HLS PIPELINE
                out[i + j] = v1_local[j] * v2_local[j];
            }
        }
    }
}
