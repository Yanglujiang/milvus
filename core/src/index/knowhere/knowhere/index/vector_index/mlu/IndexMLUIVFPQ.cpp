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

#include "knowhere/index/vector_index/ConfAdapter.h"


#include <iostream>
#include <vector>
#include <cstdint>

#include "cnrt.h"
//#include "product_quantization_mlu.h"

namespace milvus {
namespace knowhere {

using stdclock = std::chrono::high_resolution_clock;

//void
//MLUIVFPQ::Train(const DatasetPtr& dataset_ptr, const Config& config) {
//    GET_TENSOR_DATA_DIM(dataset_ptr)
//    std::cout<<"MLU IVFPQ TRAIN only support L2 metric_type"<<std::endl;
//    //LOG_KNOWHERE_WARNING_ << "MLU IVFPQ TRAIN only support L2 metric_type";
//    //faiss::MetricType metric_type = GetMetricType(config[Metric::TYPE].get<std::string>());
//    faiss::Index* coarse_quantizer = new faiss::IndexFlatL2(dim);
//    index_ = std::shared_ptr<faiss::Index>(new faiss::IndexIVFPQ(
//        coarse_quantizer, dim, config[IndexParams::nlist].get<int64_t>(), config[IndexParams::m].get<int64_t>(),
//        config[IndexParams::nbits].get<int64_t>()/*, metric_type*/));
//
//    index_->train(rows, reinterpret_cast<const float*>(p_data));
//}
//
//
//void
//MLUIVFPQ::Add(const DatasetPtr& dataset_ptr, const Config& config) {
//    std::cout<<"MLU IVFPQ  add_with_id"<<std::endl;
//    if (!index_ || !index_->is_trained) {
//        KNOWHERE_THROW_MSG("index not initialize or trained");
//    }
//    //LOG_KNOWHERE_WARNING_ << "MLU IVFPQ TRAIN only support add_with_id"
//    std::lock_guard<std::mutex> lk(mutex_);
//
//    GET_TENSOR_DATA(dataset_ptr)
//    index_->add(rows, (float*)p_data);  
//}
//
//
//void
//MLUIVFPQ::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
//    if (!index_ || !index_->is_trained) {
//        KNOWHERE_THROW_MSG("index not initialize or trained");
//    }
//    std::cout<<"MLU IVFPQ  add_without_id"<<std::endl;
//    //LOG_KNOWHERE_WARNING_ << "MLU IVFPQ TRAIN only support add_with_id"
//    std::lock_guard<std::mutex> lk(mutex_);
//
//    GET_TENSOR_DATA(dataset_ptr)
//    index_->add(rows, (float*)p_data);  
//}

void 
MLUIVFPQ::CopyIndexCpuToMlu(){
    std::lock_guard<std::mutex> lk(mutex_);
    try {
        auto ivf_index = static_cast<faiss::IndexIVFPQ*>(index_.get());
        stdclock::time_point before = stdclock::now();
        Mlu = std::make_shared<MluInterface>(mlu_id_, res_, ivf_index);
        ivf_index->make_direct_map();
        Mlu->setIndex(ivf_index);
        Mlu->CopyIndexToMLU();
        stdclock::time_point after = stdclock::now();
        double search_cost = (std::chrono::duration<double, std::micro>(after - before)).count();
        std::cout<< "MLU PCIE cost: " << search_cost<<" us"<<std::endl;
    } catch(...) {
    }
}

void
MLUIVFPQ::QueryImpl(int64_t nq, const float *data, int64_t k, float *distances, int64_t *labels, const Config& config,
               const faiss::ConcurrentBitsetPtr& bitset) {
    std::lock_guard<std::mutex> lk(mutex_);
    try {
        auto ivf_index = static_cast<faiss::IndexIVFPQ*>(index_.get());
        stdclock::time_point before = stdclock::now();
        Mlu->Search(nq, data, k, distances, labels);         
        stdclock::time_point after = stdclock::now();
        double search_cost = (std::chrono::duration<double, std::micro>(after - before)).count();
        std::cout<< "MLU search cost: " << search_cost<<" us"<<std::endl;
    } catch(...) {
    }

}
}  // namespace knowhere
}  // namespace milvus
