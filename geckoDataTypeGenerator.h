//
// Created by millad on 7/23/18.
//

#ifndef GECKO_GECKODATATYPEGENERATOR_H
#define GECKO_GECKODATATYPEGENERATOR_H

#include <vector>
#include <stdlib.h>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif



using namespace std;

class gecko_type_base {
public:
	virtual void setDevList(vector<int> dl) = 0;
	virtual void allocateMemOnlyHost(size_t count) = 0;
	virtual void allocateMemOnlyGPU(size_t count) = 0;
	virtual void allocateMem(size_t count) = 0;
	virtual void allocateMem(size_t count, vector<int> &dl) = 0;
	virtual void freeMem() = 0;
};


template<class Type>
class gecko_generator : public gecko_type_base {

	Type **arr;
	size_t total_count;
	size_t count_per_dev;
	int dev_count;
	vector<int> dev_list;

public:
	gecko_generator<Type>() : arr(NULL) {}
	~gecko_generator<Type>() {freeMem();}

	Type &operator[] (int index) {
		int new_index = index % count_per_dev;
		int dev_id = index / count_per_dev;
		if(dev_id>=dev_count) {
			dev_id = dev_count - 1;
			new_index += count_per_dev;
		}
		return arr[dev_id][new_index];
	}

	void setDevList(vector<int> dl) {
		dev_list = dl;
		dev_count = dev_list.size();
	}

	void allocateMemOnlyHost(size_t count) {
		count_per_dev = count;
		dev_count = 1;
		arr = (Type**) malloc(sizeof(Type*) * dev_count);
		arr[0] = (Type*) malloc(sizeof(Type) * count_per_dev);
	}

	void allocateMemOnlyGPU(size_t count) {
		count_per_dev = count;
		dev_count = 1;
		arr = (Type**) malloc(sizeof(Type*) * dev_count);
		cudaMalloc((void**) &arr[0], sizeof(Type) * count_per_dev);
	}

	void allocateMem(size_t count) {
		count_per_dev = count / dev_count;
		this->total_count = count;

//		arr = (Type**) malloc(sizeof(Type*) * dev_count);
		cudaMallocManaged((void***) &arr, sizeof(Type*) * dev_count);

		for(int i=0;i<dev_count;i++) {
			int dev_id = dev_list[i];
			int count_per_dev_refined = count_per_dev;
			if(i == dev_count-1)
				count_per_dev_refined = total_count - i*count_per_dev;
			if(dev_id == -1) {
//				cudaSetDevice(-1);
//				arr[i] = (Type *) malloc(sizeof(Type) * count_per_dev_refined);
				void *a = NULL;
				if(cudaSuccess != cudaMallocManaged((void**) &a, sizeof(Type) * count_per_dev_refined)) {
					fprintf(stderr, "===GECKO: Unable to allocate managed memory on device (%d).\n", dev_id);
					exit(1);
				}
				arr[i] = (Type*) a;

#ifdef INFO
				fprintf(stderr, "===GECKO: COUNT_PER_DEV - CPU: %d\n", count_per_dev_refined);
#endif

			} else {
				void *a = NULL;
#ifdef CUDA_ENABLED
				cudaSetDevice(dev_id);
				if(cudaSuccess != cudaMallocManaged((void**) &a, sizeof(Type) * count_per_dev_refined)) {
					fprintf(stderr, "===GECKO: Unable to allocate managed memory on device (%d).\n", dev_id);
					exit(1);
				}
#endif
				arr[i] = (Type*) a;

#ifdef INFO
				fprintf(stderr, "===GECKO: COUNT_PER_DEV - GPU: %d\n", count_per_dev_refined);
#endif
			}

		}
		for(int i=0;i<dev_count;i++) {
			int dev_id = dev_list[i];
			cudaMemAdvise(&arr[0], sizeof(Type **) * dev_count, cudaMemAdviseSetReadMostly, dev_id);
		}

	}

	void allocateMem(size_t count, vector<int> &dl) {
		setDevList(dl);
		allocateMem(count);
	}

	void freeMem() {
		if(arr == NULL)
			return;
		for(int i=0;i<dev_count;i++) {
			int dev_id = dev_list[i];
//			if(dev_id == -1)
//				::free(arr[i]);
//			else {
#ifdef CUDA_ENABLED
				cudaFree(arr[i]);
#endif
//			}
		}
//		::free(arr);
		cudaFree(arr);
		arr = NULL;
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
