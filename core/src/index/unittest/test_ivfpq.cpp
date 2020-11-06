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

#include <fiu-control.h>
#include <fiu/fiu-local.h>
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <thread>

#include "knowhere/common/Exception.h"
#include "knowhere/common/Timer.h"
#include "knowhere/index/IndexType.h"
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/IndexIVFPQ.h"
#include "knowhere/index/vector_index/IndexIVFSQ.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

#ifdef MILVUS_MLU_VERSION
#include "knowhere/index/vector_index/mlu/IndexMLUIVFPQ.h"
#endif

#include "unittest/Helper.h"
#include "unittest/utils.h"


using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

void
PrintDataset(const milvus::knowhere::DatasetPtr& base_dataset ,const milvus::knowhere::DatasetPtr& query_dataset) {
    auto rows = base_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
    auto nq = query_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
    auto dim = base_dataset->Get<int64_t>(milvus::knowhere::meta::DIM);
    auto base_tensor = base_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);
    auto query_tensor = base_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);
    
    auto total_size = rows * dim;
    FILE *base_dataset_file = fopen("./base_dataset", "wb+");
    FILE *query_dataset_file = fopen("./query_dataset", "wb+");
    for(auto i = 0; i < rows * dim; i++){
        fprintf(base_dataset_file, "%f\n", ((float *)base_tensor)[i]);
    }
    for(auto i = 0; i < nq * dim ; i++){
        fprintf(query_dataset_file, "%f\n", ((float *)query_tensor)[i]);
    }
}


class IVFPQTest : public DataGen,
                public TestWithParam<::std::tuple<milvus::knowhere::IndexType, milvus::knowhere::IndexMode>> {
 protected:
    void
    SetUp() override {
        std::tie(index_type_, index_mode_) = GetParam();
        int dim = 128;
        int nb = 10000;
        int nq = 2;
        Generate(dim, nb, nq);
        index_ = IndexFactory(index_type_, index_mode_);
        conf_ = milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, dim},
            {milvus::knowhere::meta::TOPK, 10},
            {milvus::knowhere::IndexParams::nlist, 1},
            {milvus::knowhere::IndexParams::nprobe, 1},
            {milvus::knowhere::IndexParams::m, 4},
            {milvus::knowhere::IndexParams::nbits, 8},
            {milvus::knowhere::Metric::TYPE, milvus::knowhere::Metric::L2},
            {milvus::knowhere::meta::DEVICEID, 0},
        };
    }

    void
    TearDown() override {
    }

 protected:
    milvus::knowhere::IndexType index_type_;
    milvus::knowhere::IndexMode index_mode_;
    milvus::knowhere::Config conf_;
    milvus::knowhere::IVFPtr index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(
    IVFPQParameters, IVFPQTest,
    Values(
#ifdef MILVUS_MLU_VERSION
        std::make_tuple(milvus::knowhere::IndexEnum::INDEX_FAISS_IVFPQ, milvus::knowhere::IndexMode::MODE_MLU),
#endif
        std::make_tuple(milvus::knowhere::IndexEnum::INDEX_FAISS_IVFPQ, milvus::knowhere::IndexMode::MODE_CPU)));

TEST_P(IVFPQTest, ivfpq_basic_mlu) {
    assert(!xb.empty());

    if (index_mode_ != milvus::knowhere::IndexMode::MODE_MLU) {
        return;
    }

    //null faiss index
    //ASSERT_ANY_THROW(index_->Add(base_dataset, conf_));
    //ASSERT_ANY_THROW(index_->AddWithoutIds(base_dataset, conf_));

    index_->Train(base_dataset, conf_);
    index_->AddWithoutIds(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    PrintDataset(base_dataset, query_dataset);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, k);
    PrintResult(result, nq, k);
}
