#include <iostream>

template <typename T>
void Randn(T *data,
		int min,
		int max ,
		int num,
		int seed){
	std::default_random_engine e(seed);
	std::uniform_real_distribution<float> u(min, max);

	for(int i = 0; i < num ; i++){
		data[i] = (T)u(e);
	}	
}

template <typename T>
void Range(T *src, T start, T end, T stride) {
  int i = 0;
  for (T v = start; v < end; v += stride) {
        src[i] = v;
	i++;
  }
}


void ComputeMse(float *out_cpu,
		float *out_mlu,
		const int64_t &output_num ){
	std::vector<float> cpu_result;
	std::vector<float> mlu_result;
	int cpu_allzero = 1;
	int mlu_allzero = 1;
	float diff1 = 0;
	float diff2 = 0 ;

	for(int64_t i =0;i < output_num ; i++){
		if(out_mlu[i] != 0.0){
			mlu_allzero = 0;
		}
		mlu_result.push_back(out_mlu[i]);
	}
	for(int64_t i =0;i < output_num ; i++){
		if(out_cpu[i] != 0.0){
			cpu_allzero = 0;
		}
		cpu_result.push_back(out_cpu[i]);
	}

	assert(cpu_result.size() == mlu_result.size());
	if (cpu_allzero && mlu_allzero) {
		std::cout << "diff1: " << "0%" << std::endl;
		std::cout << "diff2: " << "0%" << std::endl;
	} else {
		double sum = 0.0, square_sum = 0.0, tmp = 0;
		double delta_sum = 0.0, delta_square_sum = 0.0, N = cpu_result.size();
		double delta = 0.0;
		for (int i = 0; i < N; i++) {
			delta = fabs(cpu_result[i] - mlu_result[i]);
			if(delta > 0.1){
				std::cout<<"index:"<<i<<"cpu: "<<cpu_result[i] <<"mlu: "<<mlu_result[i]<<std::endl;
			}
			delta_sum += delta;
			delta_square_sum += pow(delta, 2);

			tmp = fabs(cpu_result[i]);
			sum += tmp;
			square_sum += pow(tmp, 2);
		}

		if (cpu_allzero) {
			diff1 = delta_sum;
			diff2 = delta_square_sum;
		} else {
			diff1 = (delta_sum / sum) * 100;
			diff2 = sqrt(delta_square_sum) / sqrt(square_sum) * 100;
		}
		std::cout << "diff1: " << diff1 << "%" << std::endl;
		std::cout << "diff2: " << diff2 << "%" << std::endl;
		if(diff2 > 0.0001){
		FILE *cpu_dst = fopen("./cpu_dst", "wb+");
		FILE *mlu_dst = fopen("./mlu_dst", "wb+");
		std::cout << "Error " << std::endl;
		for(int i =0;i < output_num ; i++){
			fprintf(mlu_dst,"%f\n",out_mlu[i]);
			fprintf(cpu_dst,"%f\n",out_cpu[i]);
		}
		free(cpu_dst);
		free(mlu_dst);
		abort();
		}
	}
}


