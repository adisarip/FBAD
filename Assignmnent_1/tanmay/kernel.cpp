#include <stdio.h>
#include <string.h>
#include<hls_stream.h> 

extern "C"{
    void matrix_multiply(float *data  , float *query , int N , int  Q , int *res ){
#pragma HLS INTERFACE m_axi port = data max_read_burst_length = 32  offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = query max_read_burst_length = 32  offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = res max_write_burst_length = 32 offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = query bundle = control
#pragma HLS INTERFACE s_axilite port = res bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = Q bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
        float query_local [1024]; // 4 queries 
        float query_cal[4] , query_a_val[4] ;
        int query_ans[4]; 
        // float data_val[256]; 
        hls::stream <float , 256> data_val ; 
        int n = N  , q  ; 
        
        for (int i = 0 ; i < Q +3 ; i+= 4){
            #pragma HLS LOOP_TRIPCOUNT min=4 max = 32
            if( Q < i +  4){
                q = Q - i ;
            }else {
                q = 4 ; 
            }
            ReadQuery: for (int j = 0  ;j < q ; j++){
                #pragma HLS LOOP_TRIPCOUNT min=2 max = 4  
                #pragma HLS pipeline
                for (int k = 0; k < 256; k++)
                {
                    #pragma HLS pipeline
                    #pragma HLS unroll factor=16
                    query_local[j*256 + k]  = query [ (i+ j) *256 + k];
                }
                
                
            }
            Set_0_global:for(int j = 0 ; j < q ; j++){
                #pragma HLS LOOP_TRIPCOUNT min=2 max = 4 
                #pragma HLS unroll factor=4
                query_a_val[j]= 0;
            }
            Main:for(int j = 0  ; j< n ; j++){
                #pragma HLS DATAFLOW
                #pragma HLS LOOP_TRIPCOUNT min=256 max = 4096
                
                Set_0_local:for(int j = 0 ; j < q ; j++){
                    #pragma HLS LOOP_TRIPCOUNT min=2 max = 4 
                    #pragma HLS unroll factor=4
                    query_cal[j]= 0;
                }
                
                M_rd:for (int k = 0; k < 256; k++)
                {
                    #pragma HLS pipeline
                    #pragma HLS LOOP_TRIPCOUNT min=2 max = 4 
                    data_val.write( data[k + j*256]);
                }

                M_cp:for (int k = 0; k < 256; k++)
                {
                    #pragma HLS pipeline
                    float temp = data_val.read() ; 
                    M_cp_assign:for (int  l = 0; l < q; l++)
                    {
                        #pragma HLS LOOP_TRIPCOUNT min=2 max=4
                        #pragma HLS unroll factor=4
                        query_cal[l] = query_cal [l]+  temp * query_local[l*256+k];
                    }
                }

                M_uans:for (int k = 0; k < q; k++)
                {
                    #pragma HLS LOOP_TRIPCOUNT min=2 max=4
                    #pragma HLS unroll factor=4
                    if(query_cal[k] > query_a_val[k]){
                        query_a_val[k] = query_cal[k];
                        query_ans[k] = j;
                    }
                }
            } 
            Write_ans:for (int j = 0; j < q; j++)
            {
                #pragma HLS unroll factor=4
                #pragma HLS LOOP_TRIPCOUNT min=2 max = 4 
                res[i+ j]= query_ans[j]; 
            }
        }
    }
}