#include <stdio.h>
#include <string.h>

extern "C"{
    void matrix_multiply(float *data  , float *query , int N , int  Q , int *res ){
#pragma HLS INTERFACE m_axi port = data max_read_burst_length = 64  offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = query max_read_burst_length = 64  offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = res max_write_burst_length = 64 offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = query bundle = control
#pragma HLS INTERFACE s_axilite port = res bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = Q bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
        float query_local [1024]; // 4 queries 
        float query_cal[4] , query_a_val[4] ;
        int query_ans[4]; 
        float data_val[256]; 
        int n = N  , q  ; 
        
        for (int i = 0 ; i < Q +3 ; i+= 4){
            #pragma HLS LOOP_TRIPCOUNT min=4 max = 32
            if( Q < i +  4){
                q = Q;
            }else {
                q = i + 4 ; 
            }
            ReadQuery: for (int j = i*256  ;j < q*256 ; j++){
                #pragma HLS LOOP_TRIPCOUNT min=256 max = 1024 
                #pragma HLS pipeline
                #pragma HLS unroll factor=16
                query_local[j -i]  = query [j];
            }
            INI:for(int j = 0 ; j < q-i ; j++){
                #pragma HLS LOOP_TRIPCOUNT min=2 max = 4 
                #pragma HLS unroll factor=4
                query_a_val[j]= 0;
            }
            Main:for(int j = 0  ; j< n ; j++){
                #pragma HLS DATAFLOW
                #pragma HLS stream variable=data_val depth=256
                #pragma HLS LOOP_TRIPCOUNT min=256 max = 4096
                
                query_cal[0] = 0 ;
                query_cal[1] = 0 ;
                query_cal[2] = 0 ;
                query_cal[3] = 0 ;

                M_rd:for (int k = 0; k < 256; k++)
                {
                    #pragma HLS pipeline
                    #pragma HLS LOOP_TRIPCOUNT min=2 max = 4 
                    data_val[k] = data[k + j*256];
                }

                M_cp:for (int k = 0; k < 256; k++)
                {
                    #pragma HLS pipeline
                    for (int  l = 0; l < q-i; l++)
                    {
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=4
                        #pragma HLS unroll factor=4
                        query_cal[l] += data_val[k] * query_local[l*256+k];
                    }
                }

                M_uans:for (int k = 0; k < q-i; k++)
                {
                    #pragma HLS LOOP_TRIPCOUNT min=2 max=4
                    #pragma HLS unroll factor=4
                    if(query_cal[k] > query_a_val[k]){
                        query_a_val[k] = query_cal[k];
                        query_ans[k] = j;
                    }
                }
            } 
            Write_ans:for (int j = i; j < q; j++)
            {
                #pragma HLS unroll factor=4
                #pragma HLS LOOP_TRIPCOUNT min=2 max = 4 
                res[j]= query_ans[j-i]; 
            }
        }
    }
}