#include <stdio.h>
#include <string.h>

extern "C"
{
    void matrix_multiply(float *data, float *query, int N, float *res)
    {
#pragma HLS INTERFACE m_axi port = data max_read_burst_length = 32 offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = query max_read_burst_length = 32 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = res max_write_burst_length = 32 offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = query bundle = control
#pragma HLS INTERFACE s_axilite port = res bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
        float query_local[256]; // 4 queries
        float query_cal;
        // float data_val[256];
        int n = N, q;
        Load_Query: for (int i = 0; i < 256; i++)
        {
            #pragma HLS unroll
            query_local[i] = query[i] ;
        }
        
    Main:
        for (int j = 0; j < n; j++)
        {
//#pragma HLS pipeline
#pragma HLS LOOP_TRIPCOUNT min = 16 max = 1024

        query_cal =0 ;

        M_cp:
            for (int k = 0; k < 256; k++)
            {
				#pragma HLS pipeline
            	#pragma HLS unroll factor=16

                query_cal += data[k + j * 256] * query_local[k] ;
            }

            res[j] =query_cal ;
        }
    }
}
