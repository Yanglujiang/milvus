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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "cache/CacheMgr.h"
#include "cache/DataObj.h"
#include "config/ConfigMgr.h"

namespace milvus {
namespace cache {

#ifdef MILVUS_MLU_VERSION
class MluCacheMgr;
using MluCacheMgrPtr = std::shared_ptr<MluCacheMgr>;
using MutexPtr = std::shared_ptr<std::mutex>;

class MluCacheMgr : public CacheMgr<DataObjPtr>, public ConfigObserver {
 public:
    explicit MluCacheMgr(int64_t mlu_id);

    ~MluCacheMgr();

    static MluCacheMgrPtr
    GetInstance(int64_t mlu_id);

 public:
    void
    ConfigUpdate(const std::string& name) override;

 private:
    int64_t mlu_id_;
    static std::mutex global_mutex_;
    static std::unordered_map<int64_t, MluCacheMgrPtr> instance_;
};
#endif

}  // namespace cache
}  // namespace milvus
