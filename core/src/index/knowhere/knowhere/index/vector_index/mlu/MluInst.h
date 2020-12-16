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

#ifndef MLU_INST_H
#define MLU_INST_H
#include <memory>
#include <mutex>
#include "Mlu.h"

//namespace Mlu {
namespace milvus {
namespace knowhere {

class MluInst {
 public:
    static MluInterfacePtr
    GetInstance(int64_t device_id, ResWPtr res_, faiss::IndexIVFPQ* index) {
        if (instance == nullptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (instance == nullptr) {
                instance = std::make_shared<MluInterface>(device_id, res_, index);
            }
        }
        return instance;
    }

 private:
    static MluInterfacePtr instance;
    static std::mutex mutex_;
};
}  // namespace knowhere
}  // namespace milvus

//}  // namespace Mlu
#endif
