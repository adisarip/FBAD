#include <stdio.h>
#include <string.h>
#include<hls_stream.h>

extern "C"
{
    void matrix_multiply(float *data, float *query, int N, int *res)
    {
#pragma HLS INTERFACE m_axi port = data max_read_burst_length = 32 offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = query max_read_burst_length = 32 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = res max_write_burst_length = 32 offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = query bundle = control
#pragma HLS INTERFACE s_axilite port = res bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = Q bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
        float query_local[256]; // 4 queries
        float query_cal, query_a_val  = 0 ;
        int query_ans;
        // float data_val[256];
        hls::stream<float, 256> data_val;
        int n = N, q;
        Load_Query: for (int i = 0; i < 256; i++)
        {
            #pragma HLS pipeline
            query_local[i] = query[i] ; 
        }
        
    Main:
        for (int j = 0; j < n; j++)
        {
#pragma HLS DATAFLOW
#pragma HLS LOOP_TRIPCOUNT min = 256 max = 1024

        query_cal =0 ; 

        M_rd:
            for (int k = 0; k < 256; k++)
            {
#pragma HLS pipeline
#pragma HLS LOOP_TRIPCOUNT min = 2 max = 4
                data_val.write(data[k + j * 256]);
            }

        M_cp:
            for (int k = 0; k < 256; k++)
            {
#pragma HLS pipeline
                float temp = data_val.read();
                query_cal = query_cal + temp * query_local[k] ;
            }
        
            if (query_cal > query_a_val)
            {
                query_a_val = query_cal;
                query_ans = j;
            }
        }
        res[0] = query_ans ; 
    }
}