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

#ifndef MLU_H
#define MLU_H

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "cnrt.h"
#include "product_quantization_mlu.h"
#include "knowhere/index/vector_index/helpers/FaissMluResourceMgr.h"

namespace milvus {
namespace knowhere {

//namespace Mlu {

//using idx_t = int64_t;

class MluInterface {
 public:
    int nb;  // number of dataset
    int nq;
    int nlist;
    int topk;
    int d;  // dimension
    int m;  // sub-quantizer
    int nbits;
    int ksub;

    void *index_dev = nullptr;
    void *query_dev = nullptr;

    faiss::IndexIVFPQ* index_ptr;

    int level1_centroids_offset;
    int level2_centroids_offset;
    int codes_offset;
    int ids_offset;
    int query_offset;

    MluInterface(int64_t& device_id, ResWPtr& res_, faiss::IndexIVFPQ* index);


    void
    setIndex(faiss::IndexIVFPQ* index) {
        nb = index->ntotal;
        nlist = index->nlist;
        d = index->d;
        m = index->pq.M;
        nbits = index->pq.nbits;
        ksub = 1 << nbits;
        //index_ptr = index;
        GetDevAddr();
    }

    virtual ~MluInterface();

    void
    GetDevAddr();

    void
    CopyIndexToMLU();

    void
    Search(int64_t nq, const float *data, int64_t k, float *distance, int64_t *labels);

 private: 
    ResWPtr mlu_res_;
    int mlu_id_;
};
using MluInterfacePtr = std::shared_ptr<MluInterface>;
//}  // namespace Mlu
}  // namespace knowhere
}  // namespace milvus
#endif
