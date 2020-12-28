#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <sys/time.h>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <chrono>
#include <assert.h>

#include "cnrt.h"
#include "product_quantization_mlu.h"

extern "C" {
    void MLUUnion4ProductQuantization();
    void topk_entry();
}

template <typename T>
void Range(T *src, T start, T end, T stride) {
  int i = 0;
  for (T v = start; v < end; v += stride) {
        src[i] = v;
	i++;
  }
}

template <typename T>
void readData(const char* path, T* pointer, int vec_num){
    
    std::ifstream in(path);
    std::string line;

    if(in){
        int i = 0;
        T x;
        while(std::getline(in, line)){
            std::stringstream ss(line);
            ss >> x;
            pointer[i] = x;
            i++;
        }
    }
    else{
        std::cout<<"no such file: "<<path<<std::endl;
    }
}

template <typename T>
void readLib(const char* path, T* pointer, int vec_num){
    
    std::ifstream in(path);
    std::string line;

    if(in){
        int i = 0;
        int x;
        while(std::getline(in, line)){
            std::stringstream ss(line);
            ss >> x;
            pointer[i] = (T)x;
            //printf("%hhu\n",pointer[i]);
            //cout << pointer[i]<<endl;
            i++;
        }
    }
    else{
        std::cout<<"no such file: "<<std::endl;
    }
}

template <typename T>
void CopyToMlu(T *src, int size){

}

void SaveTrainResults(float *level1_centroids,
                    float *level2_residual_centroids,
                    uint8_t *lib,
                    int nlist,
                    int dim,
                    uint64_t n, 
                    int m,
                    int k){            
    FILE* file1 = fopen("./level1_centroids", "wb+");
    FILE* file2 = fopen("./level2_residual_centroids", "wb+");
    FILE* file3 = fopen("./codes", "wb+");

    for(int i = 0; i < nlist; i++){
        for (int j = 0; j < dim; j++)
        fprintf(file1, "%f\n", level1_centroids[i * dim + j]);
    }

    for(int i = 0; i < k * dim; i++){
        fprintf(file2, "%f\n", level2_residual_centroids[i]);
    }

    for(int i = 0; i < n * m; i++){
        fprintf(file3, "%hhu\n", lib[i]);
    }

}

template <typename T>
void TransLevel2Centroids(T* src, T* dst, int N, int m, int dim){
    assert(dim%m == 0);
    int dsub = dim/m;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < m; j++){
            int boundary = dsub * N * j + i * dsub;
            int dst_left = i * dim + dsub * j;
            memcpy(dst + dst_left, src + boundary, dsub * sizeof(T));
        }
    }
}

void PQEncoder(uint8_t *lib, 
        float *long_lib, 
        float *code, 
        uint64_t N, 
        int D, 
        int M){

    int dsub = D / M ;
    int ksub = 256;

    std::vector<float> distances(ksub);
    for (uint64_t n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float mindis = 1e20;
            int idxm = 0;
            for (int i = 0; i < ksub; i++) {
                // get subvector m  and the centroids associated with subvector m
                const float * x = long_lib + n * D + dsub * m; //N * D 
                const float * y = code + i * D + dsub * m;	//k_sub * D
                for (int j = 0; j < dsub; j++) {
                    distances[i] +=  pow(x[j] - y[j], 2);
                }
            }
            /* Find best centroid */
            for (int i = 0; i < ksub; i++) {
                float dis = distances[i];
                if (dis < mindis) {
                    mindis = dis;
                    idxm = i;
                } 
                distances[i] = 0; 
            }
            lib[m * N + n]	= (uint8_t)idxm; //N * M in common, but mlu will be friendly to M * N, do transpose 
        }
    }
}

void InitMluDevice(int device_id){
    cnrtInit(0);
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, device_id);
    cnrtSetCurrentDevice(dev);
}

void getTempMemForMlu(size_t requested){
    CNRT_CHECK(cnrtMalloc(&mlu_dev, cnrtDataTypeSize(CNRT_UINT8) * requested));
}

cnrtRet_t MluTopk(void *input, 
        void *input_index,
        void *output,
        void *output_index,
        int n, 
        int c, 
        int k, 
        cnrtDataType_t src_dt, 
        cnrtDataType_t index_dt){

    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION4;
    k_dim.x = 16;
    k_dim.y = 1;
    k_dim.z = 1;

    cnrtQueue_t queue;
    cnrtCreateQueue(&queue);

    cnrtKernelInitParam_t init_param;
    cnrtCreateKernelInitParam(&init_param);
    cnrtInitKernelMemory(reinterpret_cast<void *>(&topk_entry),init_param);
    struct timeval start, end;
    double time = 0;
    gettimeofday(&start, NULL);

    cnrtKernelParamsBuffer_t params;
    cnrtGetKernelParamsBuffer(&params);
    cnrtKernelParamsBufferAddParam(params, (void *)&input, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&input_index, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&output, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&output_index, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&n, sizeof(int));
    cnrtKernelParamsBufferAddParam(params, (void *)&c, sizeof(int));
    cnrtKernelParamsBufferAddParam(params, (void *)&k, sizeof(int));
    cnrtKernelParamsBufferAddParam(params, (void *)&src_dt, sizeof(cnrtDataType_t));
    cnrtKernelParamsBufferAddParam(params, (void *)&index_dt, sizeof(cnrtDataType_t));

    cnrtInvokeKernel_V3(reinterpret_cast<void *>(&topk_entry),init_param, k_dim, params, k_type, queue, NULL);

    cnrtSyncQueue(queue);

    gettimeofday(&end, NULL);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    std::cout<<"----------------mlu invokekernel time: "<<time<<"ms"<<std::endl;

    cnrtDestroyQueue(queue);

    cnrtDestroyKernelParamsBuffer(params);
    cnrtDestroyKernelInitParamAndMemory(init_param);

    return CNRT_RET_SUCCESS;
}

cnrtRet_t MluProductQuantization(
        void *query_vec,
        void *code_vec,
        void *lut,
        void *lib_vec,
        void *output,
        cnrtDataType_t query_vec_dt,
        cnrtDataType_t code_vec_dt,
        cnrtDataType_t lib_vec_dt,
        cnrtDataType_t out_dt,
        int batch,
        int m,
        int k,
        int D,
        int n){

    if (k != 256)
    {
        std::cout << "Only support k == 256";
        abort();
    }
    if (D > 1024)
    {
        std::cout << "Only support D <= 1024";
        abort();
    }
    if(D % m != 0)
    {
        std::cout << "Only support D % m == 0";
        abort();
    }

    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION4;
    k_dim.x = 16;
    k_dim.y = 1;
    k_dim.z = 1;

    cnrtQueue_t queue;
    cnrtCreateQueue(&queue);

    cnrtKernelInitParam_t init_param;
    cnrtCreateKernelInitParam(&init_param);
    cnrtInitKernelMemory(reinterpret_cast<void *>(&MLUUnion4ProductQuantization),init_param);

    struct timeval start, end;
    double time = 0;
    gettimeofday(&start, NULL);

    cnrtKernelParamsBuffer_t params;
    cnrtGetKernelParamsBuffer(&params);
    cnrtKernelParamsBufferAddParam(params, (void *)&query_vec, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&code_vec, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&lut, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&lib_vec, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&output, sizeof(void *));
    cnrtKernelParamsBufferAddParam(params, (void *)&query_vec_dt, sizeof(cnrtDataType_t));
    cnrtKernelParamsBufferAddParam(params, (void *)&code_vec_dt, sizeof(cnrtDataType_t));
    cnrtKernelParamsBufferAddParam(params, (void *)&lib_vec_dt, sizeof(cnrtDataType_t));
    cnrtKernelParamsBufferAddParam(params, (void *)&out_dt, sizeof(cnrtDataType_t));
    cnrtKernelParamsBufferAddParam(params, (void *)&batch, sizeof(int));
    cnrtKernelParamsBufferAddParam(params, (void *)&m, sizeof(int));
    cnrtKernelParamsBufferAddParam(params, (void *)&k, sizeof(int));
    cnrtKernelParamsBufferAddParam(params, (void *)&D, sizeof(int));
    cnrtKernelParamsBufferAddParam(params, (void *)&n, sizeof(int));

    cnrtInvokeKernel_V3(reinterpret_cast<void *>(&MLUUnion4ProductQuantization),init_param, k_dim, params, k_type, queue, NULL);

    cnrtSyncQueue(queue);

    gettimeofday(&end, NULL);
    time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
    std::cout<<"----------------mlu invokekernel time: "<<time<<"ms"<<std::endl;

    cnrtDestroyQueue(queue);


    cnrtDestroyKernelParamsBuffer(params);
    cnrtDestroyKernelInitParamAndMemory(init_param);

    return CNRT_RET_SUCCESS;
}

void MLUPQSearch(float *query, 
                float *level1_centroids,
                float *code, // level2_centroids
                uint8_t *lib,// 底库短码, 需要transorder到 m * n
                const int &batch,
                const int &m,
                const int &k,
                const int &nlist,
                const int &D,
                const uint64_t &n,
                const int &topk){
    // cnrtInit(0);
    // cnrtDev_t dev;
    // cnrtGetDeviceHandle(&dev, 0);
    // cnrtSetCurrentDevice(dev);

    const cnrtDataType_t query_dt = CNRT_FLOAT32;
    const cnrtDataType_t lib_dt = CNRT_UINT8;
    cnrtDataType_t code_dt = query_dt;
    cnrtDataType_t out_dt = query_dt;

    int query_vec_num, code_vec_num, act_tbl_num;
    query_vec_num = batch * D;
    code_vec_num = k * D;
    act_tbl_num = batch * m * 128/*64k and 64b*/ * 9/*multi segs*/;
    uint64_t lib_vec_num = m * n;
    uint64_t output_num = batch * n;
    int topk_num = batch * topk;

    float *code_transorder = (float*)malloc(sizeof(float) * code_vec_num);
    uint8_t *lib_transorder = (uint8_t *)malloc(sizeof(uint8_t) * lib_vec_num);
    uint32_t *idx = (uint32_t *)malloc(sizeof(uint32_t) * batch * n);
    
    // n * m ---> m * n
    int dimValues[] = {1, 1, (int)n, m};
    int dimOrder[] = {0, 1, 3, 2};
    cnrtTransDataOrder(lib, lib_dt, lib_transorder, 4, dimValues, dimOrder);

    // (m * ksub) * dsub --> ksub * (m * dsub)
    TransLevel2Centroids(code, code_transorder, k, m, D);

    // do resiudal  
    for (int i = 0; i < batch; i++){
        for (int j = 0; j < D; j++){
            query[i * D + j] -= level1_centroids[j];
        }
    }

    Range<uint32_t>(idx, 0, n, 1);

    /*****************************************************************/
    /*			        MluProductQuantization	            		 */
    /*****************************************************************/
    void *query_vec_dev;
    void *code_vec_dev;
    void *act_tbl_dev;
    void *lib_vec_dev;
    void *output_dev;

    cnrtMalloc(&query_vec_dev, cnrtDataTypeSize(query_dt) * query_vec_num);
    cnrtMemcpy(query_vec_dev, query, cnrtDataTypeSize(query_dt) * query_vec_num, CNRT_MEM_TRANS_DIR_HOST2DEV);    
    cnrtMalloc(&code_vec_dev, cnrtDataTypeSize(code_dt) * code_vec_num);
    cnrtMemcpy(code_vec_dev, code_transorder, cnrtDataTypeSize(code_dt) * code_vec_num, CNRT_MEM_TRANS_DIR_HOST2DEV);   
    cnrtMalloc(&output_dev, cnrtDataTypeSize(out_dt) * output_num);
    cnrtMalloc(&act_tbl_dev, cnrtDataTypeSize(out_dt) * act_tbl_num);

    cnrtMalloc(&lib_vec_dev, cnrtDataTypeSize(lib_dt) * lib_vec_num);
    cnrtMemcpy(lib_vec_dev, lib_transorder, cnrtDataTypeSize(lib_dt) * lib_vec_num , CNRT_MEM_TRANS_DIR_HOST2DEV);

    void *idx_dev;
    void *topk_out_dev;
    void *topk_index_dev;

    cnrtMalloc(&idx_dev, sizeof(uint32_t) * batch * n);
    cnrtMemcpy(idx_dev, idx, sizeof(uint32_t) * batch * n, CNRT_MEM_TRANS_DIR_HOST2DEV);   
    cnrtMalloc(&topk_out_dev, sizeof(float) * topk_num);
    cnrtMalloc(&topk_index_dev, sizeof(uint32_t) * topk_num);

    cnrtDataType_t src_dt = query_dt;
    cnrtDataType_t index_dt = CNRT_INT32;

    MluProductQuantization(
            query_vec_dev,
            code_vec_dev,
            act_tbl_dev,
            lib_vec_dev,
            output_dev,
            query_dt,
            code_dt,
            lib_dt,
            out_dt,
            batch,
            m,
            k,
            D,
            n);

    MluTopk(output_dev,
            idx_dev,
            topk_out_dev,
            topk_index_dev,
            batch,
            n,
            topk,
            src_dt, 
            index_dt
           );

    float *topk_out_mlu = (float *)malloc(sizeof(float) * topk_num);
    uint32_t *topk_index_mlu = (uint32_t *)malloc(sizeof(uint32_t) * topk_num);

    cnrtMemcpy(topk_out_mlu, topk_out_dev, sizeof(float) * topk_num, CNRT_MEM_TRANS_DIR_DEV2HOST);
    cnrtMemcpy(topk_index_mlu, topk_index_dev, sizeof(uint32_t) * topk_num, CNRT_MEM_TRANS_DIR_DEV2HOST);

    // print query results 
    printf("final_indexs:\n");
    for (int i = 0; i < batch; i++){
        for (int j = 0; j < topk; j++){
            printf("%d ", topk_index_mlu[i * topk + j]);
        }
        printf("\n");
    }
    printf("\nfinal_distances:\n");
    for (int i = 0; i < batch; i++){
        for (int j = 0; j < topk; j++){
            printf("%f ", topk_out_mlu[i * topk + j]);
        }
        printf("\n");
    }	
    printf("\n");
    
    // free  mlu && cpu resources
    cnrtFree(query_vec_dev);
    cnrtFree(code_vec_dev);
    cnrtFree(lib_vec_dev);
    cnrtFree(output_dev);
    cnrtFree(act_tbl_dev);
    cnrtFree(idx_dev);
    cnrtFree(topk_out_dev);
    cnrtFree(topk_index_dev);
    cnrtFree(mlu_dev);

    free(lib_transorder);
    free(code_transorder);
    free(topk_out_mlu);
    free(topk_index_mlu);
}


