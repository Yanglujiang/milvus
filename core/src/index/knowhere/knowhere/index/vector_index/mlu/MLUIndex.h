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

#include "knowhere/index/vector_index/VecIndex.h"
#include "knowhere/index/vector_index/helpers/FaissMluResourceMgr.h"

namespace milvus {
namespace knowhere {

class MLUIndex {
 public:
    explicit MLUIndex(const int& device_id) : mlu_id_(device_id) {
    }

    MLUIndex(const int& device_id, const ResPtr& resource) : mlu_id_(device_id), res_(resource) {
    }

    void
    SetMluDevice(const int& mlu_id) {
        mlu_id_ = mlu_id;
    }

    const int64_t
    GetMluDevice() {
        return mlu_id_;
    }

 protected:
    int64_t mlu_id_;
    ResWPtr res_;
};

}  // namespace knowhere
}  // namespace milvus
