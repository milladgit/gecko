//
// Created by millad on 11/28/18.
//

#include "geckoMemory.h"
#include "geckoHierarchicalTree.h"
#include "geckoRuntime.h"

#include <unordered_map>

using namespace std;

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif


unordered_map<void*, GeckoMemory> geckoMemoryTable;


void* geckoAllocateMemory(GeckoLocationArchTypeEnum type, GeckoMemory *var) {
	void *addr = NULL;
	size_t sz_in_byte = var->dataSize * var->count;
	GeckoLocation *device = GeckoLocation::find(var->loc);
	if(device == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", var->loc.c_str());
		exit(1);
	}

	switch(type) {
		case GECKO_X32:
		case GECKO_X64:
			addr = malloc(sz_in_byte);
#ifdef INFO
			fprintf(stderr, "===GECKO: MALLOC - location: %s - size: %d - addr: %p\n", var->loc.c_str(), sz_in_byte, addr);
#endif
			break;
#ifdef CUDA_ENABLED
		case GECKO_NVIDIA:
			acc_set_device_num(device->getLocationIndex(), acc_device_nvidia);
			GECKO_CUDA_CHECK(cudaMalloc(&addr, sz_in_byte));
#ifdef INFO
			fprintf(stderr, "===GECKO: CUDAMALLOC - location: %s - size: %d - addr: %p\n", var->loc.c_str(), sz_in_byte, addr);
#endif //INFO
			break;
#endif

		case GECKO_UNIFIED_MEMORY:
#ifdef CUDA_ENABLED
			GECKO_CUDA_CHECK(cudaMallocManaged(&addr, sz_in_byte));
#ifdef INFO
			fprintf(stderr, "===GECKO: UVM ALLOCATION - location: %s - size: %d - addr: %p\n", var->loc.c_str(), sz_in_byte, addr);
#endif  // INFO
#else   // CUDA_ENABLED
			fprintf(stderr, "===GECKO: CUDA APIs are required for unified memory allocation.\n");
			exit(1);
#endif  // CUDA_ENABLED
			break;


		default:
			fprintf(stderr, "=== GECKO: Unrecognized architecture for memory allocation - Arch: %s\n",
					geckoGetLocationTypeName(type));
			exit(1);
	}

	var->address = addr;
	var->allocated = true;

	return addr;
}

inline
bool geckoAreAllChildrenCPU(GeckoLocation *node) {
	if(node == NULL)
		return true;
	size_t sz = node->getChildren().size();
	if(sz == 0) {
		return node->getLocationType().type == GECKO_X32 || node->getLocationType().type == GECKO_X64;
	}
	for(int i=0;i<sz;i++) {
		if(!geckoAreAllChildrenCPU(node->getChildren()[i]))
			return false;
	}
	return true;
}


void geckoExtractChildrenFromLocation(GeckoLocation *loc, vector<__geckoLocationIterationType> &children_names,
										int iterationCount) {

	const vector<GeckoLocation *> &v = loc->getChildren();
	const size_t sz = v.size();
	if(sz == 0) {
		__geckoLocationIterationType git;
		git.loc = loc;
		git.iterationCount = iterationCount;
		children_names.push_back(git);
		return;
	}
	for(int i=0;i<sz;i++) {
		__geckoLocationIterationType git;
		git.loc = v[i];
		if(i == sz - 1)
			git.iterationCount = static_cast<int>(iterationCount - iterationCount / sz * (sz - 1));
		else
			git.iterationCount = static_cast<int>(iterationCount / sz);

		if(v[i]->getChildren().size() == 0)
			children_names.push_back(git);
		else
			geckoExtractChildrenFromLocation(v[i], children_names, git.iterationCount);
	}

}




inline
GeckoError geckoMemoryAllocationAlgorithm(GeckoLocation *node, GeckoLocationArchTypeEnum &output_type) {
/*
 * Following the memory allocation in the paper.
 */

	if(node->getChildren().size() == 0) {
//		This node is a leaf node!
		const GeckoLocationArchTypeEnum type = node->getLocationType().type;
		if(type == GECKO_X32 || type == GECKO_X64 || type == GECKO_NVIDIA) {
			output_type = type;
		} else {
			fprintf(stderr, "===GECKO (%s:%d): Unrecognized location type for allocation. Location: %s\n",
					__FILE__, __LINE__, node->getLocationName().c_str());
			exit(1);
		}
	} else {
		const bool areAllCPUs = geckoAreAllChildrenCPU(node);
		if(areAllCPUs)
			output_type = GECKO_X64;
		else
			output_type = GECKO_UNIFIED_MEMORY;
	}

	return GECKO_SUCCESS;
}


GeckoError geckoMemoryDistribution(int loc_count, GeckoLocation **loc_list, int var_count, void **var_list,
								   int *beginIndex, int *endIndex) {

#ifdef CUDA_ENABLED
	for(int var_index=0;var_index < var_count;var_index++) {
		auto iter = geckoMemoryTable.find(var_list[var_index]);
		if(iter == geckoMemoryTable.end())
			continue;
		GeckoMemory &variable = iter->second;
		GeckoLocation *location = GeckoLocation::find(variable.loc);
		if (location->getLocationType().type != GECKO_UNIFIED_MEMORY)
			continue;

		size_t &datasize = variable.dataSize;
		void *addr = variable.address;

		for (int i = 0; i < loc_count; i++) {
			if(loc_list[i] == NULL)
				continue;
			size_t count_in_bytes = (endIndex[i] - beginIndex[i]) * datasize;
			void *ptr = (void*) ((char*)addr + i * beginIndex[i] * datasize);
			if (loc_list[i]->getLocationType().type == GECKO_NVIDIA) {
				int dstDevice = loc_list[i]->getLocationIndex();
//				int asyncID = loc_list[i]->getAsyncID();
				cudaMemAdvise(ptr, count_in_bytes, cudaMemAdviseSetPreferredLocation, dstDevice);
				cudaMemAdvise(ptr, count_in_bytes, cudaMemAdviseSetAccessedBy, dstDevice);

//				cudaMemPrefetchAsync(ptr, count_in_bytes, dstDevice, *(cudaStream_t*) acc_get_cuda_stream(asyncID));
				cudaMemPrefetchAsync(ptr, count_in_bytes, dstDevice);
			} else {
				cudaMemAdvise(ptr, count_in_bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
				cudaMemAdvise(ptr, count_in_bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

				cudaMemPrefetchAsync(ptr, count_in_bytes, cudaCpuDeviceId);
			}
		}
	}
#endif

	return GECKO_SUCCESS;
}

GeckoError geckoMemoryDeclare(void **v, size_t dataSize, size_t count, char *location, GeckoDistanceTypeEnum distance) {
	geckoInit();

	if(distance == GECKO_DISTANCE_UNKNOWN) {
		fprintf(stderr, "===GECKO: Distance at location (%s) is unknown (GECKO_DISTANCE_UNKNOWN).\n", location);
		exit(1);
	}

	GeckoMemory variable;

	variable.dataSize = dataSize;
	variable.count = count;
	variable.loc = string(location);
	GeckoLocation *const pLocation = GeckoLocation::find(variable.loc);
	if(pLocation == NULL) {
		fprintf(stderr, "===GECKO %s (%d): Unable to find the location (%s)\n", __FILE__, __LINE__, location);
		exit(1);
	}

//	variable.distance = distance;
//
//	if(distance == GECKO_DISTANCE_NOT_SET) {
	GeckoLocationArchTypeEnum type;
	geckoMemoryAllocationAlgorithm(pLocation, type);
	geckoAllocateMemory(type, &variable);
//	}

	geckoMemoryTable[variable.address] = variable;

	*v = variable.address;

	return GECKO_SUCCESS;
}







inline
void geckoFreeMemory(GeckoLocationArchTypeEnum type, void *addr, GeckoLocation *loc) {

	switch(type) {
		case GECKO_X32:
		case GECKO_X64:
			free(addr);
#ifdef INFO
			fprintf(stderr, "===GECKO: MALLOC-FREE - location %s \n", loc->getLocationName().c_str());
#endif
			break;
#ifdef CUDA_ENABLED
		case GECKO_NVIDIA:
			acc_set_device_num(loc->getLocationIndex(), acc_device_nvidia);
		case GECKO_UNIFIED_MEMORY:
		GECKO_CUDA_CHECK(cudaFree(addr));
#ifdef INFO
			fprintf(stderr, "===GECKO: CUDAFREE - location %s\n", loc->getLocationName().c_str());
#endif
			break;
#endif

		default:
			fprintf(stderr, "=== GECKO: Unrecognized architecture to free memory - Arch: %s\n",
					geckoGetLocationTypeName(type));
			exit(1);
	}
}

GeckoError geckoFree(void *ptr) {
	auto iter = geckoMemoryTable.find(ptr);
	if(iter == geckoMemoryTable.end())
		return GECKO_ERR_FAILED;

	string &loc = iter->second.loc;
	GeckoLocation *g_loc = GeckoLocation::find(loc);
	if(g_loc == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", loc);
		exit(1);
	}
//	geckoMemoryFreeAlgorithm(g_loc->getLocationType().type, ptr, g_loc);
	GeckoLocationArchTypeEnum type;
	geckoMemoryAllocationAlgorithm(g_loc, type);
	geckoFreeMemory(type, ptr, g_loc);

	iter->second.address = NULL;
	iter->second.allocated = false;

	geckoMemoryTable.erase(iter);

	return GECKO_SUCCESS;
}
