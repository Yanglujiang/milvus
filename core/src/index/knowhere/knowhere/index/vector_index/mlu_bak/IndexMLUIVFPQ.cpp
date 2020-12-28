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

#include "knowhere/index/vector_index/mlu/IndexMLUIVFPQ.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/IndexIVFPQ.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

#include <iostream>
#include <vector>

#include "cnrt.h"
#include "product_quantization_mlu.h"

namespace milvus {
namespace knowhere {

void
MLUIVFPQ::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)
    std::cout<<"MLU IVFPQ TRAIN only support L2 metric_type"<<std::endl;
    //LOG_KNOWHERE_WARNING_ << "MLU IVFPQ TRAIN only support L2 metric_type";
    //faiss::MetricType metric_type = GetMetricType(config[Metric::TYPE].get<std::string>());
    faiss::Index* coarse_quantizer = new faiss::IndexFlatL2(dim);
    index_ = std::shared_ptr<faiss::Index>(new faiss::IndexIVFPQ(
        coarse_quantizer, dim, config[IndexParams::nlist].get<int64_t>(), config[IndexParams::m].get<int64_t>(),
        config[IndexParams::nbits].get<int64_t>()/*, metric_type*/));

    index_->train(rows, reinterpret_cast<const float*>(p_data));
}

/*
void
IVF::Add(const DatasetPtr& dataset_ptr, const Config& config) {
    std::cout<<"MLU IVFPQ Add do not support add_with_id"<<std::endl;
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    std::cout<<"MLU IVFPQ Add only support add_without_id"<<std::endl;
    //LOG_KNOWHERE_WARNING_ << "MLU IVFPQ TRAIN only support add_with_id"
    std::lock_guard<std::mutex> lk(mutex_);

    GET_TENSOR_DATA(dataset_ptr)
    index_->add(rows, (float*)p_data);  
}
*/

void
MLUIVFPQ::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    std::cout<<"MLU IVFPQ Add only support add_without_id"<<std::endl;
    //LOG_KNOWHERE_WARNING_ << "MLU IVFPQ TRAIN only support add_with_id"
    std::lock_guard<std::mutex> lk(mutex_);

    GET_TENSOR_DATA(dataset_ptr)
    index_->add(rows, (float*)p_data);  
}

template <typename T>
void MLUIVFPQ::CopyCpuToMlu(T *src, int size){
    CopyToMlu(src, size);
}

void 
MLUIVFPQ::InitDevice(int device_ID){
    InitMluDevice(device_ID);
}

void 
MLUIVFPQ::alloTemMemForMlu(size_t total){
    getTempMemForMlu(total);
}

void
MLUIVFPQ::QueryImpl(int64_t nq, const float* data, int64_t k, float* distances, int64_t* labels, const Config& config,
               const faiss::ConcurrentBitsetPtr& bitset) {
    //auto params = GenParams(config);
    std::cout<<"----------------  MLU IVFPQ  Query  ----------------"<<std::endl;
    auto ivfpq_index = dynamic_cast<faiss::IndexIVFPQ*>(index_.get());
    //ivfpq_index->nprobe = std::min(params->nprobe, ivfpq_index->invlists->nlist);
    /* -----------  train&search args  ----------- */
    int dim = ivfpq_index->d; //dim
    uint64_t nb = ivfpq_index->ntotal; //nb

    int ksub = ivfpq_index->pq.ksub; // k: cluster numbers
    int m = ivfpq_index->invlists->code_size; // m
    int nlist = ivfpq_index->invlists->nlist;   // nlist

    /* -----------  search data  ------------ */
    // query_dataset: const float* data.
    // std::vector<float> level1_centroids = ((faiss::IndexFlatL2*)ivfpq_index->quantizer)->xb; // level1 cluster centers: nlist x d
    // std::vector<float> level2_residual_centroids = ivfpq_index->pq.centroids; // level2 residual cluster centers: ksub x d
    // const uint8_t * codes_0 = ivfpq_index->invlists->get_codes(0); // codes, nb * m

    float *level1_centroids = &((faiss::IndexFlatL2*)ivfpq_index->quantizer)->xb[0]; // level1_cluster centers
    float *level2_centroids = &(ivfpq_index->pq.centroids[0]);                       // level2_cluster centers
    uint8_t *lib = const_cast<uint8_t*>(ivfpq_index->invlists->get_codes(0));        // lib short codes
    
    std::cout<<"MLU IVFPQ  codes.size: "<<m<<std::endl;
    std::cout<<"MLU IVFPQ  nlist.size: "<<nlist<<std::endl;
    std::cout<<"MLU IVFPQ  list0_size: "<<nb<<std::endl;
    std::cout<<"MLU IVFPQ  nb:         "<<nb<<std::endl;
    std::cout<<"MLU IVFPQ  m:          "<<m<<std::endl;
    std::cout<<"MLU IVFPQ  ksub:       "<<ksub<<std::endl;

    // SaveTrainResults(level1_centroids, level2_centroids, lib, nlist, dim, nb, m, ksub);
    
    int mlu_device_id = 0;
    InitDevice(mlu_device_id);

    alloTemMemForMlu(nq * nb * 2 * 4);

    // CopyToMlu() --- copy query - level1_centroids to MLU
    // CopyToMlu() --- copy level2_centroids to MLU(after trans_level2_centroids)
    // CopyToMLU() --- copy libcodes to MLU(after transorder)

    MLUPQSearch(const_cast<float*>(data), 
            level1_centroids,
            level2_centroids,
            lib, 
            nq, 
            m, 
            ksub, 
            nlist,
            dim, 
            nb, 
            k); // topk

    // CopyToCpu() --- copy topk_indexs to CPU
    // CopyToCpu() --- copy topk_distances to CPU
}
}  // namespace knowhere
}  // namespace milvus
