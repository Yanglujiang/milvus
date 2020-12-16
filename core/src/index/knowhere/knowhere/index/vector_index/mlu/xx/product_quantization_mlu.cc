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
#include <typeinfo>

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

//template <typename T>
//void readData(const char* path, T* pointer, int vec_num){
//    
//    std::ifstream in(path);
//    std::string line;
//
//    if(in){
//        int i = 0;
//        T x;
//        while(std::getline(in, line)){
//            std::stringstream ss(line);
//            ss >> x;
//            pointer[i] = x;
//            i++;
//        }
//    }
//    else{
//        std::cout<<"no such file: "<<path<<std::endl;
//    }
//}
//
//template <typename T>
//void readLib(const char* path, T* pointer, int vec_num){
//    
//    std::ifstream in(path);
//    std::string line;
//
//    if(in){
//        int i = 0;
//        int x;
//        while(std::getline(in, line)){
//            std::stringstream ss(line);
//            ss >> x;
//            pointer[i] = (T)x;
//            //printf("%hhu\n",pointer[i]);
//            //cout << pointer[i]<<endl;
//            i++;
//        }
//    }
//    else{
//        std::cout<<"no such file: "<<std::endl;
//    }
//}

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

void AdaptDataForBase(float *level2_centroids,
                     float *level2_centroids_transorder,
                     uint8_t *lib,
                     uint8_t *lib_transorder,
                     uint64_t nb,
                     int ksub,
                     int m,
                     int dim){
    // (m * ksub) * dsub --> ksub * (m * dsub)
    TransLevel2Centroids(level2_centroids, level2_centroids_transorder, ksub, m, dim);

    // n * m ---> m * n
    const cnrtDataType_t lib_dt = CNRT_UINT8;
    int dimValues[] = {1, 1, (int)nb, m};
    int dimOrder[] = {0, 1, 3, 2};
    cnrtTransDataOrder(lib, lib_dt, lib_transorder, 4, dimValues, dimOrder);
}


void AdaptDataForQuery(float *query, 
                     float *level1_centroids,
                     float *level2_centroids,
                     float *level2_centroids_transorder,
                     uint8_t *lib,
                     uint8_t *lib_transorder,
                     uint64_t nb,
                     int nq,
                     int ksub,
                     int m,
                     int dim){
    // do resiudal: query - level1_centroids 
    for (int i = 0; i < nq; i++){
        for (int j = 0; j < dim; j++){
            query[i * dim + j] -= level1_centroids[j];
        }
    }

    // (m * ksub) * dsub --> ksub * (m * dsub)
    TransLevel2Centroids(level2_centroids, level2_centroids_transorder, ksub, m, dim);

    // n * m ---> m * n
    const cnrtDataType_t lib_dt = CNRT_UINT8;
    int dimValues[] = {1, 1, (int)nb, m};
    int dimOrder[] = {0, 1, 3, 2};
    cnrtTransDataOrder(lib, lib_dt, lib_transorder, 4, dimValues, dimOrder);
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

void CopyToMlu(float *src, int size){
    CNRT_CHECK(cnrtMemcpy(mlu_dev + offset, src, cnrtDataTypeSize(CNRT_FLOAT32) * size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    offset += cnrtDataTypeSize(CNRT_FLOAT32) * size;
    //std::cout<<"size: "<<size<<"   "<<"offset: "<<offset<<std::endl;
}

void CopyToMlu(uint32_t *src, int size){
    CNRT_CHECK(cnrtMemcpy(mlu_dev + offset, src, cnrtDataTypeSize(CNRT_UINT32) * size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    offset += cnrtDataTypeSize(CNRT_FLOAT32) * size;
    //std::cout<<"size: "<<size<<"   "<<"offset: "<<offset<<std::endl;
}


void CopyToMlu(uint8_t *src, int size){
    CNRT_CHECK(cnrtMemcpy(mlu_dev + offset, src, cnrtDataTypeSize(CNRT_UINT8) * size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    offset += cnrtDataTypeSize(CNRT_UINT8) * size;
    //std::cout<<"size: "<<size<<"   "<<"offset: "<<offset<<std::endl;
}

void CopyToCpu(int32_t *dst, int size){
    CNRT_CHECK(cnrtMemcpy(dst, mlu_dev + topk_index_dev_offset, sizeof(int32_t) * size, CNRT_MEM_TRANS_DIR_DEV2HOST));
}

void CopyToCpu(float *dst, int size){
    CNRT_CHECK(cnrtMemcpy(dst, mlu_dev + topk_out_dev_offset, sizeof(float) * size, CNRT_MEM_TRANS_DIR_DEV2HOST));
}

void ConvertInt32ToInt64(int32_t *src, int64_t *dst, int length){
    for (int i = 0; i < length; i++){
        dst[i] = src[i];
    }
}

void ConvertInt64ToInt32(int64_t *src, int32_t *dst, int length){
    for (int i = 0; i < length; i++){
        dst[i] = src[i];
    }
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
    std::cout<<"---------------- mlu invokekernel time: "<<time<<"ms"<<std::endl;

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

void MLUPQSearch(const int &batch,
                const int &m,
                const int &k,
                const int &D,
                const uint64_t &n,
                const int &topk){

    const cnrtDataType_t query_dt = CNRT_FLOAT32;
    const cnrtDataType_t lib_dt = CNRT_UINT8;
    cnrtDataType_t code_dt = query_dt;
    cnrtDataType_t out_dt = query_dt;
    cnrtDataType_t src_dt = query_dt;
    cnrtDataType_t index_dt = CNRT_INT32;

    int act_tbl_num = batch * m * 128/*64k and 64b*/ * 9/*multi segs*/;

    // calculate memory offsets
    //size_t query_offset = 0;
    //size_t level2_centroids_offset = query_offset + cnrtDataTypeSize(query_dt) * batch * D;
    size_t level2_centroids_offset = 0;
    size_t lib_offset = level2_centroids_offset + cnrtDataTypeSize(query_dt) * k * D;
    size_t idx_dev_offset = lib_offset + cnrtDataTypeSize(lib_dt) * ((int)n) * m;
    size_t query_offset = idx_dev_offset + cnrtDataTypeSize(CNRT_UINT32) * ((int)n);
    size_t act_tbl_offset = query_offset + cnrtDataTypeSize(query_dt) * batch * D;
    size_t output_dev_offset = act_tbl_offset + cnrtDataTypeSize(out_dt) * act_tbl_num;
    //size_t act_tbl_offset = lib_offset + cnrtDataTypeSize(lib_dt) * ((int)n) * m;
    //size_t idx_dev_offset = act_tbl_offset + cnrtDataTypeSize(out_dt) * act_tbl_num;
    //size_t output_dev_offset = idx_dev_offset + cnrtDataTypeSize(CNRT_UINT32) * batch * ((int)n);
    topk_out_dev_offset = output_dev_offset + cnrtDataTypeSize(out_dt) * batch * ((int)n);  
    topk_index_dev_offset = topk_out_dev_offset + cnrtDataTypeSize(out_dt) * batch * topk;

    std::cout<<"level2_centroids_offset:        "<<level2_centroids_offset<<std::endl; 
    std::cout<<"lib_offset:                     "<<lib_offset<<std::endl;   
    std::cout<<"idx_dev_offset:                 "<<idx_dev_offset<<std::endl;    
    std::cout<<"query_offset:                   "<<query_offset<<std::endl;    
    std::cout<<"act_tbl_offset:                 "<<act_tbl_offset<<std::endl;    
    std::cout<<"output_dev_offset:              "<<output_dev_offset<<std::endl;   
    std::cout<<"output_distances_dev_offset:    "<<topk_out_dev_offset<<std::endl;
    std::cout<<"topk_index_dev_offset:          "<<topk_index_dev_offset<<std::endl;

    /*****************************************************************/
    /*			        MluProductQuantization	            		 */
    /*****************************************************************/
//    uint32_t *idx = (uint32_t *)malloc(sizeof(uint32_t) /* batch*/ * n);
//    Range<uint32_t>(idx, 0, n, 1);
//    CNRT_CHECK(cnrtMemcpy(mlu_dev + idx_dev_offset, idx, sizeof(uint32_t) /* batch*/ * n, CNRT_MEM_TRANS_DIR_HOST2DEV));   
//
    MluProductQuantization(
            mlu_dev + query_offset,
            mlu_dev + level2_centroids_offset,
            mlu_dev + act_tbl_offset,
            mlu_dev + lib_offset,
            mlu_dev + output_dev_offset,
            query_dt,
            code_dt,
            lib_dt,
            out_dt,
            batch,
            m,
            k,
            D,
            n);

    MluTopk(mlu_dev + output_dev_offset,
            mlu_dev + idx_dev_offset,
            mlu_dev + topk_out_dev_offset, //topk_out_dev,
            mlu_dev + topk_index_dev_offset,//topk_index_dev,
            batch,
            n,
            topk,
            src_dt, 
            index_dt
           );
}


