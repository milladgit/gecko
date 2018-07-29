
#include "geckoDataTypeGenerator.h"
#include "geckoUtils.h"

void gecko_type_base::setDevList(vector<int> dl) {
	*dev_count = dl.size();
	dev_list = (int*) malloc(sizeof(int) * *dev_count);
	for(int i=0;i<*dev_count;i++) {
		dev_list[i] = dl[i];
	}
}


void *gecko_type_base::alloc(GeckoGeneratorMemoryType mem_type, int datasize, int count) {
	if(mem_type == GECKO_GENERATOR_HOST) {
		return malloc(datasize * count);
	} else if(mem_type == GECKO_GENERATOR_GPU) {
		void *p;
		GECKO_CUDA_CHECK(cudaMalloc((void**) &p, datasize * count));
		return p;
	} else if(mem_type == GECKO_GENERATOR_UNIFIED) {
		void *p;
		GECKO_CUDA_CHECK(cudaMallocManaged((void**) &p, datasize * count));
		return p;
	}
	return NULL;
}

void gecko_type_base::freeMemBase(void **arr) {
	if(arr == NULL)
		return;

	if(dev_list) {
		free(dev_list);
		dev_list = NULL;
	}

	if(mem_type == GECKO_GENERATOR_HOST) {
		for (int i = 0; i < *dev_count; i++)
			free(arr[i]);
		free(arr);
	}
#ifdef CUDA_ENABLED
	else if(mem_type == GECKO_GENERATOR_GPU) {
		for (int i = 0; i < *dev_count; i++)
			cudaFree(arr[i]);
		free(arr);
	}
	else if(mem_type == GECKO_GENERATOR_UNIFIED) {
		for (int i = 0; i < *dev_count; i++)
			cudaFree(arr[i]);
		cudaFree(arr);
	}
#endif
	arr = NULL;
}
