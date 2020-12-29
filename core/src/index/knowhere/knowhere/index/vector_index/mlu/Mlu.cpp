// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include "knowhere/index/vector_index/mlu/Mlu.h"
#include <algorithm>
//namespace Mlu {
namespace milvus {
namespace knowhere {


MluInterface::MluInterface(int64_t& device_id, ResWPtr& res_, faiss::IndexIVFPQ* index) : 
        mlu_id_((int)device_id), mlu_res_(res_), index_ptr(index){ 
}

MluInterface::~MluInterface() {
}

void
MluInterface::GetDevAddr() {
    //level1_centroids_offset = 0;
    level2_centroids_offset = 0;
    codes_offset = level2_centroids_offset + sizeof(float) * ksub * d;
    ids_offset = codes_offset + sizeof(uint8_t) * m * nb;    
    //query_offset = ids_offset + sizeof(int32_t) * nb;    
}

void ConvertInt64ToInt32(int64_t *src, int32_t *dst, int length){
    for (int i = 0; i < length; i++){
        dst[i] = src[i];
    }
}

void ConvertUint32ToInt64(uint32_t *src, int64_t *dst, int length){
    for (int i = 0; i < length; i++){
        dst[i] = src[i];
    }
}


void TransLevel2Centroids(float* src, float* dst, int N, int m, int dim){
    assert(dim%m == 0);
    int dsub = dim/m;
    for (int i = 0; i < N; i++){
        for (int j = 0; j < m; j++){
            int boundary = dsub * N * j + i * dsub;
            int dst_left = i * dim + dsub * j;
            memcpy(dst + dst_left, src + boundary, dsub * sizeof(float));
        }
    }
}

void AdaptDataForBase(float *level2_centroids,
                     float *level2_centroids_transorder,
                     uint8_t *lib,
                     uint8_t *lib_transorder,
                     int nb,
                     int ksub,
                     int m,
                     int d){
    // (m * ksub) * dsub --> ksub * (m * dsub)

    TransLevel2Centroids(level2_centroids, level2_centroids_transorder, ksub, m, d);

    // n * m ---> m * n
    const cnrtDataType_t lib_dt = CNRT_UINT8;
    int dimValues[] = {1, 1, nb, m};
    int dimOrder[] = {0, 1, 3, 2};
    cnrtTransDataOrder(lib, lib_dt, lib_transorder, 4, dimValues, dimOrder);
}


void
MluInterface::CopyIndexToMLU() {
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, mlu_id_);
    cnrtSetCurrentDevice(dev);

    int index_request = sizeof(float) * ksub * d + sizeof(uint8_t) * m * nb + sizeof(int32_t) * nb; 
    CNRT_CHECK(cnrtMalloc(&index_dev, cnrtDataTypeSize(CNRT_UINT8) * index_request));

    float *level2_centroids = &(index_ptr->pq.centroids[0]);                        // level2_cluster centers
    uint8_t *codes = const_cast<uint8_t *>(index_ptr->invlists->get_codes(0));        // lib short codes
    int64_t *ids_64 = const_cast<int64_t *>(index_ptr->invlists->get_ids(0));        //lib index 

    //std::vector<float> level1_centroids_transorder; // level1_cluster_trans centers
    std::vector<float> level2_centroids_transorder(ksub * d); // level2_cluster_trans centers
    std::vector<uint8_t> codes_transorder(m * nb);    // lib_transorder codes
    std::vector<int32_t> ids_32(nb);    // lib_transorder codes

    AdaptDataForBase(
                level2_centroids, 
                level2_centroids_transorder.data(), 
                codes, 
                codes_transorder.data(),
                nb, 
                ksub, 
                m, 
                d);

    ConvertInt64ToInt32(ids_64, ids_32.data(), nb);
    
    cnrtMemcpy((char *)index_dev + level2_centroids_offset, 
            level2_centroids_transorder.data(), sizeof(float) * ksub * d, CNRT_MEM_TRANS_DIR_HOST2DEV);

    cnrtMemcpy((char *)index_dev + codes_offset, 
            codes_transorder.data(), sizeof(uint8_t) * m * nb, CNRT_MEM_TRANS_DIR_HOST2DEV);

//    only support  search by auto ids
//    cnrtMemcpy((char *)index_dev + ids_offset, 
//            ids_32.data(), sizeof(int32_t) * nb, CNRT_MEM_TRANS_DIR_HOST2DEV);

}

void
MluInterface::Search(int64_t num_query, const float *data, int64_t k, float *distance, int64_t *labels) {
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, mlu_id_);
    cnrtSetCurrentDevice(dev);

    nq = num_query;
    topk = k;

    // if query size > 64 we search by blocks to avoid malloc issue
    const int64_t block_size = 16;
    int64_t search_size = (nq > block_size) ? block_size : nq;
    int64_t query_request = sizeof(float) * search_size * d + sizeof(float) * search_size * m * 128 * 9 + sizeof(float) * search_size * nb + sizeof(float) * search_size * topk +sizeof(int32_t) * search_size * topk;
    CNRT_CHECK(cnrtMalloc(&query_dev, cnrtDataTypeSize(CNRT_UINT8) * query_request));

    for (int64_t query_loop = 0; query_loop < nq; query_loop += block_size) {
        search_size = (nq - query_loop > block_size) ? block_size : (nq - query_loop);

        int query_offset = 0;
        int act_tbl_offset = query_offset + sizeof(float) * search_size * d;
        int output_offset = act_tbl_offset + sizeof(float) * search_size * m * 128 * 9;
        int topk_out_offset = output_offset + sizeof(float) * search_size * nb;  
        int topk_out_ids_offset = topk_out_offset + sizeof(float) * search_size * topk;

        const cnrtDataType_t query_dt = CNRT_FLOAT32;
        const cnrtDataType_t lib_dt = CNRT_UINT8;
        cnrtDataType_t code_dt = query_dt;
        cnrtDataType_t out_dt = query_dt;
        cnrtDataType_t src_dt = query_dt;
        cnrtDataType_t index_dt = CNRT_INT32;

        float *level1_centroids = &((faiss::IndexFlatL2 *)index_ptr->quantizer)->xb[0]; // level1_cluster centers
        
        std::vector<float> query(search_size * d); 
        // do resiudal: query - level1_centroids 
        for (int i = 0; i < search_size; i++){
            for (int j = 0; j < d; j++){
                query[i * d + j] = data[(query_loop + i) * d + j] - level1_centroids[j];
            }
        }

        cnrtMemcpy((char *)query_dev + query_offset, 
                query.data(), sizeof(float) * search_size * d, CNRT_MEM_TRANS_DIR_HOST2DEV);
 
        MluProductQuantization(
                (char *)query_dev + query_offset,
                (char *)index_dev + level2_centroids_offset,
                (char *)query_dev + act_tbl_offset,
                (char *)index_dev + codes_offset,
                (char *)query_dev + output_offset,
                query_dt,
                code_dt,
                lib_dt,
                out_dt,
                search_size,
                m,
                ksub,
                d,
                nb);
        //for(int i = 0; i < search_size; i++){
        //    std::vector<int64_t> ss_ids(nb); 
        //    std::vector<float> ss_dis(nb); 
        //    typedef std::pair<int64_t, float> Pair;
        //    std::vector<Pair> pq_search_out_dis_ids;
        //    
        //    cnrtMemcpy(ss_dis.data(), (char *)query_dev + output_offset + nb * i * sizeof(float),  sizeof(float) * nb, CNRT_MEM_TRANS_DIR_DEV2HOST);

        //    int64_t *ids_64 = const_cast<int64_t *>(index_ptr->invlists->get_ids(0));        //lib index 
        //    for(int j = 0; j < nb; j++){
        //        ss_ids[j] = ids_64[j];
        //        pq_search_out_dis_ids.push_back(std::make_pair(ss_ids[j], ss_dis[j]));
        //    }
        //    std::partial_sort(pq_search_out_dis_ids.begin(), pq_search_out_dis_ids.begin() + k, pq_search_out_dis_ids.end(),
        //    		                      [](Pair &x, Pair &y) -> bool { return x.second < y.second; });
        //    
        //    for(int j = 0; j < topk; j++){
        //    	distance[(query_loop + i) * topk + j] = pq_search_out_dis_ids[j].second;
        //    	labels[(query_loop + i) * topk + j] = pq_search_out_dis_ids[j].first;
        //    }
        //    pq_search_out_dis_ids.clear();
        //}
        MluTopk(
                (char *)query_dev + output_offset,
                (char *)query_dev + topk_out_offset, 
                (char *)query_dev + topk_out_ids_offset,
                topk,
              	search_size,
		1,
                nb,
		0,
                src_dt);

        cnrtMemcpy(distance, (char *)query_dev + topk_out_offset,  sizeof(float) * search_size * topk, CNRT_MEM_TRANS_DIR_DEV2HOST);

        std::vector<uint32_t> ids_32(search_size * topk); 
        cnrtMemcpy(ids_32.data(), (char *)query_dev + topk_out_ids_offset, sizeof(uint32_t) * search_size * topk, CNRT_MEM_TRANS_DIR_DEV2HOST);

        ConvertUint32ToInt64(ids_32.data(), labels + query_loop * topk, search_size * topk);
    }
    CNRT_CHECK(cnrtFree(index_dev));
    CNRT_CHECK(cnrtFree(query_dev));

}
}  // namespace knowhere
}  // namespace milvus

//}  // namespace Mlu
