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
#ifdef MILVUS_MLU_VERSION
#include "scheduler/selector/FaissMLUIVFPQPass.h"
#include "cache/MluCacheMgr.h"
#include "config/ServerConfig.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "scheduler/SchedInst.h"
#include "scheduler/Utils.h"
#include "scheduler/task/SearchTask.h"
//#include "scheduler/tasklabel/SpecResLabel.h"
#include "server/ValidationUtil.h"
#include "utils/Log.h"

namespace milvus {
namespace scheduler {

FaissMLUIVFPQPass::FaissMLUIVFPQPass() {
    ConfigMgr::GetInstance().Attach("mlu.mlu_search_threshold", this);
}

FaissMLUIVFPQPass::~FaissMLUIVFPQPass() {
    ConfigMgr::GetInstance().Detach("mlu.mlu_search_threshold", this);
}

void
FaissMLUIVFPQPass::Init() {
#ifdef MILVUS_MLU_VERSION
    mlu_enable_ = config.mlu.enable();
    threshold_ = config.mlu.mlu_search_threshold();
    search_mlus_ = ParseMLUDevices(config.mlu.search_devices());
#endif
}

bool
FaissMLUIVFPQPass::Run(const TaskPtr& task) {
    if (task->Type() != TaskType::SearchTask) {
        return false;
    }

    auto search_task = std::static_pointer_cast<SearchTask>(task);
    auto index_type = search_task->IndexType();
    if ( index_type != knowhere::IndexEnum::INDEX_FAISS_IVFPQ) {
        return false;
    }

    ResourcePtr res_ptr;
    if (!mlu_enable_) {
        LOG_SERVER_DEBUG_ << LogOut("FaissMLUIVFPQPass: mlu disable, specify cpu to search!");
        res_ptr = ResMgrInst::GetInstance()->GetResource("cpu");
    } else if (search_task->nq() < threshold_) {
        LOG_SERVER_DEBUG_ << LogOut("FaissMLUIVFPQPass: nq < mlu_search_threshold, specify cpu to search!");
        res_ptr = ResMgrInst::GetInstance()->GetResource("cpu");
    } else if (search_task->topk() > server::MLU_QUERY_MAX_TOPK) {
        LOG_SERVER_DEBUG_ << LogOut("FaissMLUIVFPQPass: topk > mlu_max_topk_threshold, specify cpu to search!");
        res_ptr = ResMgrInst::GetInstance()->GetResource("cpu");
    } else if (search_task->ExtraParam()[knowhere::IndexParams::nprobe].get<int64_t>() > server::MLU_QUERY_MAX_NPROBE) {
        LOG_SERVER_DEBUG_ << LogOut("FaissMLUIVFPQPass: nprobe > mlu_max_nprobe_threshold, specify cpu to search!");
        res_ptr = ResMgrInst::GetInstance()->GetResource("cpu");
    } else {
        LOG_SERVER_DEBUG_ << LogOut("FaissMLUIVFPQPass: nq >= mlu_search_threshold, specify mlu %d to search!",
                                    search_mlus_[idx_]);
        res_ptr = ResMgrInst::GetInstance()->GetResource(ResourceType::MLU, search_mlus_[idx_]);
        idx_ = (idx_ + 1) % search_mlus_.size();
    }
    task->resource() = res_ptr;
    return true;
}

void
FaissMLUIVFPQPass::ConfigUpdate(const std::string& name) {
    threshold_ = config.mlu.mlu_search_threshold();
}

}  // namespace scheduler
}  // namespace milvus
#endif
