//
// Created by millad on 7/23/18.
//

#ifndef GECKO_GECKODATATYPEGENERATOR_H
#define GECKO_GECKODATATYPEGENERATOR_H

#include <vector>
#include <stdlib.h>

//#define CUDA_ENABLED

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <openacc.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

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
	size_t *total_count;
	size_t *count_per_dev;
	int *dev_count;
	GeckoGeneratorMemoryType mem_type;
	int sizes_in_byte[2];
	int *dev_list;

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


public:
	Type **arr;
	gecko_generator<Type>() : arr(NULL) {
		mem_type = GECKO_GENERATOR_UNKNOWN;
		sizes_in_byte[0] = sizeof(Type);
		sizes_in_byte[1] = sizeof(Type*);
		GECKO_CUDA_CHECK(cudaMallocManaged((void**) &total_count, sizeof(size_t)));
		cudaMallocManaged((void**) &count_per_dev, sizeof(size_t));
		cudaMallocManaged((void**) &dev_count, sizeof(int));
	}
	~gecko_generator<Type>() {
	}

//#pragma acc routine seq
	Type &operator[] (size_t index) {
		int new_index = index % (*count_per_dev);
		int dev_id = index / (*count_per_dev);
		if(dev_id>=*dev_count) {
			dev_id = *dev_count - 1;
//i			new_index += *count_per_dev;
			new_index = index - dev_id*(*count_per_dev);
		}
		Type *d = arr[dev_id];
		return d[new_index];
//#endif
	}

	Type *operator+ (int index) {
		return &operator[](index);
	}

	void allocateMemOnlyHost(size_t count) {
		mem_type = GECKO_GENERATOR_HOST;
		*count_per_dev = count;
		*total_count = count;
		*dev_count = 1;
		arr = (Type**) malloc(sizeof(Type*) * *dev_count);
		arr[0] = (Type*) malloc(sizeof(Type) * *count_per_dev);
	}

	void allocateMemOnlyGPU(size_t count) {
		if(true) {
			allocateMemUnifiedMem(count);
			return;
		}
//		mem_type = GECKO_GENERATOR_GPU;
//		count_per_dev = count;
//		dev_count = 1;
//		arr = (Type**) malloc(sizeof(Type*) * dev_count);
//		cudaMalloc((void**) &arr[0], sizeof(Type) * count_per_dev);
	}

	void allocateMemUnifiedMem(size_t count) {
		mem_type = GECKO_GENERATOR_UNIFIED;
		*count_per_dev = count / *dev_count;
		*total_count = count;
		int curr_dev = acc_get_device_num(acc_device_nvidia);

		acc_set_device_num(0, acc_device_nvidia);
		GECKO_CUDA_CHECK(cudaMallocManaged((void***) &arr, sizeof(Type*) * *dev_count));

		//for(int i=0;i<*dev_count;i++) {
		#pragma omp parallel num_threads(*dev_count)
		{
			int i = omp_get_thread_num();
			int dev_id = dev_list[i];
			size_t count_per_dev_refined = *count_per_dev;
			if(i == *dev_count-1)
				count_per_dev_refined = *total_count - i*(*count_per_dev);
			if(dev_id != cudaCpuDeviceId)
				acc_set_device_num(dev_id, acc_device_nvidia);
			else
				acc_set_device_num(0, acc_device_host);

			void *a = NULL;
#ifdef CUDA_ENABLED
			// printf("==============================FROM HELL\n");
			GECKO_CUDA_CHECK(cudaMallocManaged((void**) &a, sizeof(Type) * count_per_dev_refined));
			// printf("==============================FROM HELL 2 - array_count: %d - DEV_COUNT: %d\n", count_per_dev_refined, *dev_count);
#endif
#pragma acc wait
			arr[i] = (Type*) a;
			// printf("==============================FROM HELL 2 - A\n");

#ifdef INFO
			fprintf(stderr, "===GECKO: COUNT_PER_DEV - %s: %d\n", (dev_id == -1 ? "CPU" : "GPU"), count_per_dev_refined);
#endif
		}

		// printf("==============================FROM HELL 3\n");
		/*
		for(int i=0;i<*dev_count;i++) {
			int dev_id = dev_list[i];
			GECKO_CUDA_CHECK(cudaMemAdvise(&arr[dev_id], sizeof(Type **) * *dev_count, cudaMemAdviseSetReadMostly, dev_id));
		}
		 */
		// printf("==============================FROM HELL 4\n");
		acc_set_device_num(curr_dev, acc_device_nvidia);
		// printf("==============================FROM HELL 5\n");
	}

	void allocateMem(size_t count, vector<int> &dl) {
		setDevList(dl);
		allocateMemUnifiedMem(count);
	}

	void freeMem() {
		cudaFree(total_count);
		cudaFree(count_per_dev);
		cudaFree(dev_count);
		freeMemBase((void**)arr);
	}

};

#define G_GENERATOR(TYPE) typedef gecko_generator<TYPE> gecko_##TYPE

G_GENERATOR(char);
G_GENERATOR(int);
G_GENERATOR(long);
G_GENERATOR(float);
G_GENERATOR(double);
G_GENERATOR(bool);
typedef gecko_generator<unsigned int> gecko_unsigned_int;


#endif //GECKO_GECKODATATYPEGENERATOR_H
