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

#pragma once

#include <memory>
#include <utility>

#include "knowhere/index/vector_index/IndexIVFPQ.h"


namespace milvus {
namespace knowhere {

class MLUIVFPQ : public IVFPQ  {
 public:
    //faiss::Index* coarse_quantizer;
    MLUIVFPQ(const int& device_id) : IVFPQ() {
        index_type_ = IndexEnum::INDEX_FAISS_IVFPQ;
    }
    explicit MLUIVFPQ(std::shared_ptr<faiss::Index> index, const int64_t device_id) : IVFPQ(std::move(index)) {
        index_type_ = IndexEnum::INDEX_FAISS_IVFPQ;
    }

    void
    Train(const DatasetPtr&, const Config&) override;
 
    /*
    void
    Add(const DatasetPtr&, const Config&) override;
    */

    void InitDevice(int device_ID);

    void alloTemMemForMlu(size_t total);

    template <typename T>
    void CopyCpuToMlu(T* src, int size);

/*    cpuTomLU

   mlutoCpu */


    void
    AddWithoutIds(const DatasetPtr&, const Config&) override;

 
 protected:
    void
    QueryImpl(int64_t, const float*, int64_t, float*, int64_t*, const Config&,
              const faiss::ConcurrentBitsetPtr& bitset) override;


};

using MLUIVFPQPtr = std::shared_ptr<MLUIVFPQ>;
}  // namespace knowhere
}  // namespace milvus
