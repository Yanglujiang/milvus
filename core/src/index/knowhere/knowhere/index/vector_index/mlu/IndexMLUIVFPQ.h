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
#include "knowhere/index/vector_index/mlu/MLUIndex.h"
#include "knowhere/index/vector_index/mlu/MluInst.h"

namespace milvus {
namespace knowhere {


class MLUIVFPQ : public IVFPQ , public MLUIndex {
 public:
    explicit MLUIVFPQ(const int& device_id) : IVFPQ() , MLUIndex(device_id) {
        index_mode_ = IndexMode::MODE_MLU;
        index_type_ = IndexEnum::INDEX_FAISS_IVFPQ;
    }
    MLUIVFPQ(std::shared_ptr<faiss::Index> index, const int64_t device_id, ResPtr& res) : IVFPQ(std::move(index)) , MLUIndex(device_id, res) {
        index_mode_ = IndexMode::MODE_MLU;
        index_type_ = IndexEnum::INDEX_FAISS_IVFPQ;
    }

    //void
    //Train(const DatasetPtr&, const Config&) override;
 
    //void
    //Add(const DatasetPtr&, const Config&) override;
 
    //void
    //AddWithoutIds(const DatasetPtr&, const Config&) override;

    void 
    CopyIndexCpuToMlu();

 protected:
    void
    QueryImpl(int64_t, const float*, int64_t, float*, int64_t*, const Config&,
              const faiss::ConcurrentBitsetPtr& bitset) override;

    MluInterfacePtr Mlu;
};

using MLUIVFPQPtr = std::shared_ptr<MLUIVFPQ>;
}  // namespace knowhere
}  // namespace milvus
