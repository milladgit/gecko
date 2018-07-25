//
// Created by millad on 7/23/18.
//

#ifndef GECKO_GECKODATATYPEGENERATOR_H
#define GECKO_GECKODATATYPEGENERATOR_H

#include <vector>
#include <stdlib.h>

#define CUDA_ENABLED

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <openacc.h>

#include "geckoUtils.h"

using namespace std;


typedef enum {
	GECKO_GENERATOR_HOST = 0,
	GECKO_GENERATOR_GPU,
	GECKO_GENERATOR_UNIFIED,
	GECKO_GENERATOR_UNKNOWN,
	GECKO_GENERATOR_LEN
}GeckoGeneratorMemoryType;




class gecko_type_base {
protected:
	size_t total_count;
	size_t count_per_dev;
	int dev_count;
	GeckoGeneratorMemoryType mem_type;
	int sizes_in_byte[2];
	vector<int> dev_list;

	void *alloc(GeckoGeneratorMemoryType mem_type, int datasize, int count);
	void freeMemBase(void **arr);

public:
	void setDevList(vector<int> dl);
	virtual void allocateMemOnlyHost(size_t count) = 0;
	virtual void allocateMemOnlyGPU(size_t count) = 0;
	virtual void allocateMemUnifiedMem(size_t count) = 0;
	virtual void allocateMem(size_t count, vector<int> &dl) = 0;
	virtual void freeMem() = 0;
};


template<class Type>
class gecko_generator : public gecko_type_base {

	Type **arr;

public:
	gecko_generator<Type>() : arr(NULL) {
		mem_type = GECKO_GENERATOR_UNKNOWN;
		sizes_in_byte[0] = sizeof(Type);
		sizes_in_byte[1] = sizeof(Type*);
	}
	~gecko_generator<Type>() {}

	Type &operator[] (int index) {
		int new_index = index % count_per_dev;
		int dev_id = index / count_per_dev;
		if(dev_id>=dev_count) {
			dev_id = dev_count - 1;
			new_index += count_per_dev;
		}
		return arr[dev_id][new_index];
	}

	Type *operator+ (int index) {
		return &operator[](index);
	}

	void allocateMemOnlyHost(size_t count) {
		mem_type = GECKO_GENERATOR_HOST;
		count_per_dev = count;
		dev_count = 1;
		arr = (Type**) malloc(sizeof(Type*) * dev_count);
		arr[0] = (Type*) malloc(sizeof(Type) * count_per_dev);
	}

	void allocateMemOnlyGPU(size_t count) {
		if(true) {
			allocateMemUnifiedMem(count);
			return;
		}
		mem_type = GECKO_GENERATOR_GPU;
		count_per_dev = count;
		dev_count = 1;
		arr = (Type**) malloc(sizeof(Type*) * dev_count);
		cudaMalloc((void**) &arr[0], sizeof(Type) * count_per_dev);
	}

	void allocateMemUnifiedMem(size_t count) {
		mem_type = GECKO_GENERATOR_UNIFIED;
		count_per_dev = count / dev_count;
		this->total_count = count;

		GECKO_CUDA_CHECK(cudaMallocManaged((void***) &arr, sizeof(Type*) * dev_count));

		for(int i=0;i<dev_count;i++) {
			int dev_id = dev_list[i];
			int count_per_dev_refined = count_per_dev;
			if(i == dev_count-1)
				count_per_dev_refined = total_count - i*count_per_dev;
			if(dev_id != -1)
				acc_set_device_num(dev_id, acc_device_nvidia);

			void *a = NULL;
#ifdef CUDA_ENABLED
			GECKO_CUDA_CHECK(cudaMallocManaged((void**) &a, sizeof(Type) * count_per_dev_refined));
#endif
			arr[i] = (Type*) a;

#ifdef INFO
			fprintf(stderr, "===GECKO: COUNT_PER_DEV - %s: %d\n", (dev_id == -1 ? "CPU" : "GPU"), count_per_dev_refined);
#endif
		}

		for(int i=0;i<dev_count;i++) {
			int dev_id = dev_list[i];
			cudaMemAdvise(&arr[0], sizeof(Type **) * dev_count, cudaMemAdviseSetReadMostly, dev_id);
		}
	}

	void allocateMem(size_t count, vector<int> &dl) {
		setDevList(dl);
		allocateMemUnifiedMem(count);
	}

	void freeMem() {
		freeMemBase((void**)arr);
	}

};

#define G_GENERATOR(TYPE) typedef gecko_generator<TYPE> gecko_##TYPE

G_GENERATOR(char);
G_GENERATOR(int);
G_GENERATOR(long);
G_GENERATOR(float);
G_GENERATOR(double);
typedef gecko_generator<unsigned int> gecko_unsigned_int;


#endif //GECKO_GECKODATATYPEGENERATOR_H
