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

#include <faiss/AutoTune.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/clone_index.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#ifdef MILVUS_GPU_VERSION
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#endif

#include <fiu/fiu-local.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cnrt.h"

#include "faiss/BuilderSuspend.h"
#include "knowhere/common/Exception.h"
#include "knowhere/common/Log.h"
#include "knowhere/index/vector_index/IndexIVF.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#ifdef MILVUS_GPU_VERSION
#include "knowhere/index/vector_index/gpu/IndexGPUIVF.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#endif

namespace milvus {
namespace knowhere {

using stdclock = std::chrono::high_resolution_clock;

BinarySet
IVF::Serialize(const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    std::lock_guard<std::mutex> lk(mutex_);
    return SerializeImpl(index_type_);
}

void
IVF::Load(const BinarySet& binary_set) {
    std::lock_guard<std::mutex> lk(mutex_);
    LoadImpl(binary_set, index_type_);
}

void
IVF::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    GET_TENSOR_DATA_DIM(dataset_ptr)

    faiss::MetricType metric_type = GetMetricType(config[Metric::TYPE].get<std::string>());
    faiss::Index* coarse_quantizer = new faiss::IndexFlat(dim, metric_type);
    auto nlist = config[IndexParams::nlist].get<int64_t>();
    index_ = std::shared_ptr<faiss::Index>(new faiss::IndexIVFFlat(coarse_quantizer, dim, nlist, metric_type));
    index_->train(rows, reinterpret_cast<const float*>(p_data));
}

void
IVF::Add(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    std::lock_guard<std::mutex> lk(mutex_);
    GET_TENSOR_DATA_ID(dataset_ptr)
    index_->add_with_ids(rows, reinterpret_cast<const float*>(p_data), p_ids);
}

void
IVF::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    std::lock_guard<std::mutex> lk(mutex_);
    GET_TENSOR_DATA(dataset_ptr)
    index_->add(rows, reinterpret_cast<const float*>(p_data));
}

DatasetPtr
IVF::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::ConcurrentBitsetPtr& bitset) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    GET_TENSOR_DATA(dataset_ptr)

    try {
        fiu_do_on("IVF.Search.throw_std_exception", throw std::exception());
        fiu_do_on("IVF.Search.throw_faiss_exception", throw faiss::FaissException(""));
        auto k = config[meta::TOPK].get<int64_t>();
        // rows: nq, k:topk
        auto elems = rows * k;

        size_t p_id_size = sizeof(int64_t) * elems;
        size_t p_dist_size = sizeof(float) * elems;
        auto p_id = static_cast<int64_t*>(malloc(p_id_size));
        auto p_dist = static_cast<float*>(malloc(p_dist_size));

        QueryImpl(rows, reinterpret_cast<const float*>(p_data), k, p_dist, p_id, config, bitset);

        //    std::stringstream ss_res_id, ss_res_dist;
        //    for (int i = 0; i < 10; ++i) {
        //        printf("%llu", p_id[i]);
        //        printf("\n");
        //        printf("%.6f", p_dist[i]);
        //        printf("\n");
        //        ss_res_id << p_id[i] << " ";
        //        ss_res_dist << p_dist[i] << " ";
        //    }
        //    std::cout << std::endl << "after search: " << std::endl;
        //    std::cout << ss_res_id.str() << std::endl;
        //    std::cout << ss_res_dist.str() << std::endl << std::endl;

        auto ret_ds = std::make_shared<Dataset>();
        ret_ds->Set(meta::IDS, p_id);
        ret_ds->Set(meta::DISTANCE, p_dist);
        return ret_ds;
    } catch (faiss::FaissException& e) {
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

#if 0
DatasetPtr
IVF::QueryById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    auto rows = dataset_ptr->Get<int64_t>(meta::ROWS);
    auto p_data = dataset_ptr->Get<const int64_t*>(meta::IDS);

    try {
        int64_t k = config[meta::TOPK].get<int64_t>();
        auto elems = rows * k;

        size_t p_id_size = sizeof(int64_t) * elems;
        size_t p_dist_size = sizeof(float) * elems;
        auto p_id = (int64_t*)malloc(p_id_size);
        auto p_dist = (float*)malloc(p_dist_size);

        // todo: enable search by id (zhiru)
        //        auto blacklist = dataset_ptr->Get<faiss::ConcurrentBitsetPtr>("bitset");
        auto index_ivf = std::static_pointer_cast<faiss::IndexIVF>(index_);
        index_ivf->search_by_id(rows, p_data, k, p_dist, p_id, bitset_);

        //    std::stringstream ss_res_id, ss_res_dist;
        //    for (int i = 0; i < 10; ++i) {
        //        printf("%llu", res_ids[i]);
        //        printf("\n");
        //        printf("%.6f", res_dis[i]);
        //        printf("\n");
        //        ss_res_id << res_ids[i] << " ";
        //        ss_res_dist << res_dis[i] << " ";
        //    }
        //    std::cout << std::endl << "after search: " << std::endl;
        //    std::cout << ss_res_id.str() << std::endl;
        //    std::cout << ss_res_dist.str() << std::endl << std::endl;

        auto ret_ds = std::make_shared<Dataset>();
        ret_ds->Set(meta::IDS, p_id);
        ret_ds->Set(meta::DISTANCE, p_dist);
        return ret_ds;
    } catch (faiss::FaissException& e) {
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

DatasetPtr
IVF::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    auto p_data = dataset_ptr->Get<const int64_t*>(meta::IDS);
    auto elems = dataset_ptr->Get<int64_t>(meta::DIM);

    try {
        size_t p_x_size = sizeof(float) * elems;
        auto p_x = (float*)malloc(p_x_size);

        auto index_ivf = std::static_pointer_cast<faiss::IndexIVF>(index_);
        index_ivf->get_vector_by_id(1, p_data, p_x, bitset_);

        auto ret_ds = std::make_shared<Dataset>();
        ret_ds->Set(meta::TENSOR, p_x);
        return ret_ds;
    } catch (faiss::FaissException& e) {
        KNOWHERE_THROW_MSG(e.what());
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}
#endif

int64_t
IVF::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->ntotal;
}

int64_t
IVF::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->d;
}

void
IVF::Seal() {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    SealImpl();
}

void
IVF::UpdateIndexSize() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    auto ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(index_.get());
    auto nb = ivf_index->invlists->compute_ntotal();
    auto nlist = ivf_index->nlist;
    auto code_size = ivf_index->code_size;
    // ivf codes, ivf ids and quantizer
    index_size_ = nb * code_size + nb * sizeof(int64_t) + nlist * code_size;
}

VecIndexPtr
IVF::CopyCpuToGpu(const int64_t device_id, const Config& config) {
#ifdef MILVUS_GPU_VERSION
    if (auto res = FaissGpuResourceMgr::GetInstance().GetRes(device_id)) {
        ResScope rs(res, device_id, false);
        auto gpu_index = faiss::gpu::index_cpu_to_gpu(res->faiss_res.get(), device_id, index_.get());

        std::shared_ptr<faiss::Index> device_index;
        device_index.reset(gpu_index);
        return std::make_shared<GPUIVF>(device_index, device_id, res);
    } else {
        KNOWHERE_THROW_MSG("CopyCpuToGpu Error, can't get gpu_resource");
    }

#else
    KNOWHERE_THROW_MSG("Calling IVF::CopyCpuToGpu when we are using CPU version");
#endif
}

void
IVF::GenGraph(const float* data, const int64_t k, GraphType& graph, const Config& config) {
    int64_t K = k + 1;
    auto ntotal = Count();

    size_t dim = config[meta::DIM];
    auto batch_size = 1000;
    auto tail_batch_size = ntotal % batch_size;
    auto batch_search_count = ntotal / batch_size;
    auto total_search_count = tail_batch_size == 0 ? batch_search_count : batch_search_count + 1;

    std::vector<float> res_dis(K * batch_size);
    graph.resize(ntotal);
    GraphType res_vec(total_search_count);
    for (int i = 0; i < total_search_count; ++i) {
        // it is usually used in NSG::train, to check BuilderSuspend
        faiss::BuilderSuspend::check_wait();

        auto b_size = (i == (total_search_count - 1)) && tail_batch_size != 0 ? tail_batch_size : batch_size;

        auto& res = res_vec[i];
        res.resize(K * b_size);

        const float* xq = data + batch_size * dim * i;
        QueryImpl(b_size, xq, K, res_dis.data(), res.data(), config, nullptr);

        for (int j = 0; j < b_size; ++j) {
            auto& node = graph[batch_size * i + j];
            node.resize(k);
            auto start_pos = j * K + 1;
            for (int m = 0, cursor = start_pos; m < k && cursor < start_pos + k; ++m, ++cursor) {
                node[m] = res[cursor];
            }
        }
    }
}

std::shared_ptr<faiss::IVFSearchParameters>
IVF::GenParams(const Config& config) {
    auto params = std::make_shared<faiss::IVFSearchParameters>();
    params->nprobe = config[IndexParams::nprobe];
    // params->max_codes = config["max_codes"];
    return params;
}

void
IVF::QueryImpl(int64_t n, const float* data, int64_t k, float* distances, int64_t* labels, const Config& config,
               const faiss::ConcurrentBitsetPtr& bitset) {
    auto params = GenParams(config);
    auto ivf_index = dynamic_cast<faiss::IndexIVFPQ*>(index_.get());
    ivf_index->nprobe = std::min(params->nprobe, ivf_index->invlists->nlist);
    stdclock::time_point before = stdclock::now();
    if (params->nprobe > 1 && n <= 4) {
        ivf_index->parallel_mode = 1;
    } else {
        ivf_index->parallel_mode = 0;
    }

    /* -----------  train&search args  ----------- */
    uint64_t nb = ivf_index->ntotal; // nb
    int dim = ivf_index->d; // dim
    int m = ivf_index->invlists->code_size; // m
    int nlist = ivf_index->invlists->nlist;   // nlist
    int ksub = ivf_index->pq.ksub; // k: cluster numbers
    int dsub = ivf_index->pq.dsub; // dsub: dim of sub_vector

    std::cout<<" IVFPQ  nb      "<<nb<<std::endl;
    std::cout<<" IVFPQ  nlist:  "<<nlist<<std::endl;
    std::cout<<" IVFPQ  ksub:   "<<ksub<<std::endl;
    std::cout<<" IVFPQ  dim     "<<dim<<std::endl;
    std::cout<<" IVFPQ  m       "<<m<<std::endl;
    std::cout<<" IVFPQ  dusb:   "<<dsub<<std::endl;

    // float *level1_centroids = &((faiss::IndexFlatL2 *)ivf_index->quantizer)->xb[0]; // level1_cluster centers
    std::vector<float> level1_centroids = ((faiss::IndexFlatL2*)ivf_index->quantizer)->xb; // level1 cluster centers: nlist x d
    float *level2_residual_centroids = &(ivf_index->pq.centroids[0]);                        // level2_cluster centers: ksub x m x dsub
    float *level2_residual_centroids_transorder = (float *)malloc(ksub * dim * sizeof(float)); // level2_cluster_trans centers: ksub x d
    

    std::vector<int> vecs_per_partition;
    std::vector < std::vector<int64_t> > nprobe_ids; // Inverted lists for indexes, size <nlist , list_size>
    std::vector< std::vector<uint8_t> > nprobe_codes;
    std::vector<int64_t> ids;
    std::vector<uint8_t> codes;

    FILE* nlist_coarse_clusters = fopen("./nlist_coarse_clusters","wb+");
    FILE* nprobe_level2_centroids = fopen("./nprobe_level2_centroids", "wb+");
    FILE* nprobe_level2_centroids_ksub_D = fopen("./nprobe_level2_centroids_ksub_D", "wb+");
    FILE* vecs_num_per_partition = fopen("./vecs_num_per_partion","wb+");  // nlist
    FILE* ids_per_partition = fopen("./ids_per_partition","wb+");
    FILE* codes_per_partition = fopen("./codes_per_partition","wb+");
    //FILE* code_book = fopen("./code_book","wb+");
    //FILE* code_book_trans = fopen("./code_book_m_N","wb+");

    // fprintf nlist coarse clusters
    for(int i = 0; i < nlist * dim; i++){
        fprintf(nlist_coarse_clusters, "%f\n", level1_centroids[i]);
    }

    // // transorder: ksub x m x dsub --> ksub x (m x dsub)
    for (int i = 0; i < ksub; i++){
        for (int j = 0; j < m; j++){
            int boundary = dsub * ksub * j + i * dsub;
            int dst_left = i * dim + dsub * j;
            memcpy(level2_residual_centroids_transorder + dst_left, level2_residual_centroids + boundary, dsub * sizeof(float));
        }
    }

    // fprintf level2_residual_centroids: ksub x m x dsub
    for(int i = 0; i < ksub * dim; i++){
        fprintf(nprobe_level2_centroids, "%f\n", level2_residual_centroids[i]);
        fprintf(nprobe_level2_centroids_ksub_D, "%f\n", level2_residual_centroids_transorder[i]);
    }

    // uint8_t *lib = const_cast<uint8_t *>(ivf_index->invlists->get_codes(0));        // lib short codes: nb x m

    // uint8_t *lib_transorder = (uint8_t *)malloc(nb * m * sizeof(uint8_t)); 

    // transorder lib: n * m --> m * n
    // const cnrtDataType_t lib_dt = CNRT_UINT8;
    // int dimValues[] = {1, 1, (int)nb, m};
    // int dimOrder[] = {0, 1, 3, 2};
    // cnrtTransDataOrder(lib, lib_dt, lib_transorder, 4, dimValues, dimOrder);
    // for (int i = 0; i < nb * m; i++){
    //     fprintf(code_book_trans, "%u\n", lib_transorder[i]);
    // }


    // fprintf lib: nb * m
    // for (int i = 0; i < nb * m; i++){
    //     fprintf(code_book, "%u\n", lib[i]);
    //     fprintf(code_book_trans, "%u\n", lib_transorder[i]);
    // }




    // 
    int sum = 0;
    int number_base_vecs;
    int64_t *partion_ids;
    uint8_t *partion_codes;
    for (int i = 0; i < nlist; i++){
        partion_ids = const_cast<int64_t *>(ivf_index->invlists->get_ids(i));
        partion_codes = const_cast<uint8_t *>(ivf_index->invlists->get_codes(i));
        
        number_base_vecs = ivf_index->invlists->list_size(i);

        // fprintf vecs_num_per_partition
        fprintf(vecs_num_per_partition, "%d\n", number_base_vecs);

        vecs_per_partition.push_back(number_base_vecs);

        sum += number_base_vecs;
        for (int j = 0; j < number_base_vecs; j++){

            // frpintf ids per partition
            fprintf(ids_per_partition, "%ld\n", partion_ids[j]);
            ids.push_back(partion_ids[j]);
            for (int m = 0; m < 16; m++){
                fprintf(codes_per_partition, "%u\n", partion_codes[j * 16 + m]);
            }
            
            codes.push_back(partion_codes[j]);
        }
        nprobe_ids.push_back(ids);
        nprobe_codes.push_back(codes);
        codes.clear();
        ids.clear();

    }

    // for (auto elem : vecs_per_partition){
    //     std::cout<<elem<<" ";
    // }

    std::cout<<"\ntotal ids are: "<<sum<<std::endl;


    // ------  search entry -----
  
    ivf_index->search(n, data, k, distances, labels, bitset);
    stdclock::time_point after = stdclock::now();
    double search_cost = (std::chrono::duration<double, std::micro>(after - before)).count();
    LOG_KNOWHERE_DEBUG_ << "IVF search cost: " << search_cost
                        << ", quantization cost: " << faiss::indexIVF_stats.quantization_time
                        << ", data search cost: " << faiss::indexIVF_stats.search_time;
    faiss::indexIVF_stats.quantization_time = 0;
    faiss::indexIVF_stats.search_time = 0;
}

void
IVF::SealImpl() {
#ifdef MILVUS_GPU_VERSION
    faiss::Index* index = index_.get();
    auto idx = dynamic_cast<faiss::IndexIVF*>(index);
    if (idx != nullptr) {
        idx->to_readonly();
    }
#endif
}

}  // namespace knowhere
}  // namespace milvus
