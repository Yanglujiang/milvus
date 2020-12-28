/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*************************************************************************/
#ifndef MLU_PRODUCT_QUANTIZATION_H_
#define MLU_PRODUCT_QUANTIZATION_H_

#include "cnrt.h"

void *mlu_dev;                   // mlu memory space
size_t offset = 0;                   // mlu memory offset for copy query_vec, level2_centroids, libcodes to mlu
size_t topk_out_dev_offset;      // topk distances mlu mem offset
size_t topk_index_dev_offset;    // topk indexs mlu mem offset

/****************************
     mlu_dev storage layout:
      --------------------
      |       query       |  
      |-------------------|
      |  level2_centroids |
      |-------------------| 
      |      libcodes     |
      |-------------------| 
      |   act_tbl_dev     |
      |-------------------| 
      |     index_dev     |
      |-------------------| 
      |    output_dev     |
      |-------------------| 
      |   topk_out_dev    |
      |-------------------|
      |  topk_index_dev   |
      ---------------------
****************************/

template <typename T>
void Range(T *src,
         T start, 
         T end, 
         T stride);

template <typename T>
void readData(const char *path, 
        T *pointer, 
        int vec_num );

template <typename T>
void readLib(const char *path, 
        T *pointer, 
        int vec_num );

void CopyToMlu(float *src, 
        int size);

void CopyToMlu(uint8_t *src,
        int size);

void CopyToCpu(uint32_t *dst, 
        int size);

void CopyToCpu(float *dst,
        int size);

void ConvertUint32ToInt64(uint32_t *src, 
                        int64_t *dst, int length);

template <typename T>
void TransLevel2Centroids(T *src,
                         T *dst, 
                         int N, 
                         int m, 
                         int dim);
                
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
                     int dim);

void SaveTrainResults(float *level1_centroids,
                    float *level2_residual_centroids,
                    uint8_t *lib,
                    int nlist,
                    int dim,
                    uint64_t n, 
                    int m,
                    int k);

void PQEncoder(uint8_t *lib, 
        float *long_lib, 
        float *code, 
        uint64_t N, 
        int D, 
        int M);

void InitMluDevice(int device_id);

cnrtRet_t FreeMluResoures(){
        CNRT_CHECK(cnrtFree(mlu_dev));
        return CNRT_RET_SUCCESS;
}

void getTempMemForMlu(size_t requested);

cnrtRet_t MluTopk(void *input, 
        void *input_index,
        void *output,
        void *output_index,
        int n, 
        int c, 
        int k, 
        cnrtDataType_t src_dt, 
        cnrtDataType_t index_dt);

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
        int n);
        
// void* mlu_dev
void MLUPQSearch(
        /* float *query,
        float *level2_centroids,
        uint8_t *lib, */
        const int &batch,
        const int &m, 
        const int &k,
        const int &D,
        const uint64_t &n,
        const int &topk);

#endif  //MLU_PRODUCT_QUANTIZATION_H_
