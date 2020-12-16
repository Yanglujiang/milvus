// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "cache/MluCacheMgr.h"
#include "config/ServerConfig.h"
#include "utils/Log.h"

#include <fiu/fiu-local.h>
#include <sstream>
#include <utility>

namespace milvus {
namespace cache {

#ifdef MILVUS_MLU_VERSION
std::mutex MluCacheMgr::global_mutex_;
std::unordered_map<int64_t, MluCacheMgrPtr> MluCacheMgr::instance_;

MluCacheMgr::MluCacheMgr(int64_t mlu_id) : mlu_id_(mlu_id) {
    std::string header = "[CACHE MLU" + std::to_string(mlu_id) + "]";
    cache_ = std::make_shared<Cache<DataObjPtr>>(config.mlu.cache_size(), 1UL << 32, header);

    if (config.mlu.cache_threshold() > 0.0) {
        cache_->set_freemem_percent(config.mlu.cache_threshold());
    }
    ConfigMgr::GetInstance().Attach("mlu.cache_threshold", this);
}

MluCacheMgr::~MluCacheMgr() {
    ConfigMgr::GetInstance().Detach("mlu.cache_threshold", this);
}

MluCacheMgrPtr
MluCacheMgr::GetInstance(int64_t mlu_id) {
    if (instance_.find(mlu_id) == instance_.end()) {
        std::lock_guard<std::mutex> lock(global_mutex_);
        if (instance_.find(mlu_id) == instance_.end()) {
            instance_[mlu_id] = std::make_shared<MluCacheMgr>(mlu_id);
        }
    }
    return instance_[mlu_id];
}

void
MluCacheMgr::ConfigUpdate(const std::string& name) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    for (auto& it : instance_) {
        it.second->SetCapacity(config.mlu.cache_size());
    }
}

#endif

}  // namespace cache
}  // namespace milvus
