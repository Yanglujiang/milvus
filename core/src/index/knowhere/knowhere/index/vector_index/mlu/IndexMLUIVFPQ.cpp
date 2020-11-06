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
#include <string>
#include <vector>
#include <chrono>

#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/IndexIVFPQ.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"

#include<iostream>

namespace milvus {
namespace knowhere {

void
MLUIVFPQ::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)
    std::cout<<"MLU IVFPQ TRAIN only support L2 metric_type"<<std::endl;
    //LOG_KNOWHERE_WARNING_ << "MLU IVFPQ TRAIN only support L2 metric_type";
    //faiss::MetricType metric_type = GetMetricType(config[Metric::TYPE].get<std::string>());
    //faiss::Index* coarse_quantizer = new faiss::IndexFlat(dim, metric_type);
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

void
MLUIVFPQ::QueryImpl(int64_t n, const float* data, int64_t k, float* distances, int64_t* labels, const Config& config,
               const faiss::ConcurrentBitsetPtr& bitset) {
    std::cout<<"MLU IVFPQ  Query"<<std::endl;
    auto params = GenParams(config);
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->nprobe = std::min(params->nprobe, ivf_index->invlists->nlist);
    //stdclock::time_point before = stdclock::now();
    if (params->nprobe > 1 && n <= 4) {
        ivf_index->parallel_mode = 1;
    } else {
        ivf_index->parallel_mode = 0;
    }
    ivf_index->search(n, data, k, distances, labels, bitset);
    //stdclock::time_point after = stdclock::now();
    //double search_cost = (std::chrono::duration<double, std::micro>(after - before)).count();
    //LOG_KNOWHERE_DEBUG_ << "IVF search cost: " << search_cost
    //                    << ", quantization cost: " << faiss::indexIVF_stats.quantization_time
    //                    << ", data search cost: " << faiss::indexIVF_stats.search_time;
    faiss::indexIVF_stats.quantization_time = 0;
    faiss::indexIVF_stats.search_time = 0;
}

}  // namespace knowhere
}  // namespace milvus
