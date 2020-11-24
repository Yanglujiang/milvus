#include <cassert>
#include <algorithm>

int CpuProductQuantization_Topk(float *query,
		float *code,
		float *lib_cpu,
		float *out_cpu,
		float *topk_cpu,
		uint32_t *topk_index_cpu,
		int batch,
		int m,
		int k,
		int D,
		uint64_t n){

	auto pq_distance = [&](float *x, float* code_vec, float* code_book, int k, int D, int m) -> float {
		assert(D % m == 0);
		int Dsub = D / m;
		// k center and m space
		float distance = 0;
		for (int i = 0; i < m; i++) {
			int center_index = (int)code_vec[i];
			// code_book is k x D(m * Dsub)
			float *center = code_book + center_index * D + i * Dsub;
			float *x_seg_i = x + i * Dsub;
			for (int j = 0; j < Dsub; j++) {
				distance += std::pow(x_seg_i[j] - center[j], 2);
			}
		}
		return distance;
	};

	// query matrix : query_m nq x D
	// code book : code_book   k x D
	// the code of vector in training : code_db  m * n
	// distance : output_distance nq x n
	auto pq_search = [&](float *query_m, float *code_book, float *code_db, float *out_distance,
			int nq, int D, int k, uint64_t n, int m) {
		float* code_db_j = (float*) malloc(sizeof(float) * m);
		for (int i = 0; i < nq; i++) {
			float* query_i = query_m + i * D;
			float* out_distance_i = out_distance + i * n;
			for (uint64_t j = 0; j < n; j++) {
				for (int o = 0; o < m; o++) {
					code_db_j[o] = code_db[o * n + j];
				}
				out_distance_i[j] = pq_distance(query_i, code_db_j, code_book, k, D, m);
			}
		}
		free(code_db_j);
	};
	
	pq_search(query, code, lib_cpu, out_cpu, batch, D, k, n, m);

	return 0;
}


