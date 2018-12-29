//
// Created by millad on 11/28/18.
//

#include "geckoMemory.h"
#include "geckoHierarchicalTree.h"
#include "geckoRuntime.h"

#include <stdlib.h>

#include <unordered_map>
#include <unistd.h>
#include <chrono>
#include <thread>

using namespace std;

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <cstring>

#endif


unordered_map<void*, GeckoMemory> geckoMemoryTable;


void* geckoAllocateMemory(GeckoLocationArchTypeEnum type, GeckoLocation *device, GeckoMemory *var) {
	void *addr = NULL;
	size_t sz_in_byte = var->dataSize * var->count;
	if(device == NULL) {
		device = GeckoLocation::find(var->loc);
		if (device == NULL) {
			fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", var->loc.c_str());
			exit(1);
		}
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
	if(v.empty()) {
		__geckoLocationIterationType git;
		git.loc = loc;
		git.iterationCount = iterationCount;
		children_names.push_back(git);
		return;
	}
	const size_t sz = v.size();
	const int delta = static_cast<int>(iterationCount / sz);
	for(int i=0;i<sz;i++) {
		__geckoLocationIterationType git;
		git.loc = v[i];

		git.iterationCount = ((i < sz) ? delta : static_cast<int>(iterationCount - delta * (sz - 1)));

		if(git.loc->getChildren().empty())
			children_names.push_back(git);
		else
			geckoExtractChildrenFromLocation(git.loc, children_names, git.iterationCount);

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

GeckoError geckoMemoryDeclare(void **v, size_t dataSize, size_t count, char *location, GeckoDistanceTypeEnum distance,
		int distance_level, GeckoDistanceAllocationTypeEnum allocationType) {

	geckoInit();

	if(distance == GECKO_DISTANCE_UNKNOWN) {
		fprintf(stderr, "===GECKO: Distance at location (%s) is unknown (GECKO_DISTANCE_UNKNOWN).\n", location);
		exit(1);
	}
	if(distance_level <= 0 && distance == GECKO_DISTANCE_FAR) {
		fprintf(stderr, "===GECKO: Distance level starts from 1.\n");
		exit(1);
	}

	GeckoMemory variable;

	variable.dataSize = dataSize;
	variable.count = count;
	variable.distance = distance;
	variable.distance_level = distance_level;
	variable.allocType = allocationType;

	if(distance == GECKO_DISTANCE_NEAR || distance == GECKO_DISTANCE_FAR) {
		variable.is_dummy = true;
		variable.address = malloc(dataSize);
	} else {
		variable.loc = string(location);
		GeckoLocation *pLocation = GeckoLocation::find(variable.loc);
		if (pLocation == NULL) {
			fprintf(stderr, "===GECKO %s (%d): Unable to find the location (%s)\n", __FILE__, __LINE__, location);
			exit(1);
		}
		variable.loc_ptr = (GeckoLocation *) pLocation;

		GeckoLocationArchTypeEnum type;
		geckoMemoryAllocationAlgorithm(pLocation, type);
		geckoAllocateMemory(type, pLocation, &variable);
	}

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

GeckoError __geckoFindLocationBasedOnPointer(void *ptr, GeckoLocationArchTypeEnum &type, GeckoLocation **out_loc) {
	*out_loc = NULL;

	const auto iter = geckoMemoryTable.find(ptr);
	if(iter == geckoMemoryTable.end())
		return GECKO_ERR_MEM_ADDRESS_NOT_FOUND;

	string &loc = iter->second.loc;
	GeckoLocation *g_loc = GeckoLocation::find(loc);
	if(g_loc == NULL)
		return GECKO_ERR_LOCATION_NOT_FOUND;

	geckoMemoryAllocationAlgorithm(g_loc, type);
}

GeckoError geckoFree(void *ptr) {
	geckoInit();

	GeckoLocation *g_loc;
	GeckoLocationArchTypeEnum type;

	const auto iter = geckoMemoryTable.find(ptr);
	if(iter == geckoMemoryTable.end()) {
		fprintf(stderr, "===GECKO: Unable to find memory '%p'.\n", ptr);
		exit(1);
	}

	string &loc = iter->second.loc;
	g_loc = GeckoLocation::find(loc);
	if(g_loc == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", loc.c_str());
		exit(1);
	}

	geckoMemoryAllocationAlgorithm(g_loc, type);

	const int distance = iter->second.distance;
	if(distance == GECKO_DISTANCE_FAR || distance == GECKO_DISTANCE_NEAR) {
		geckoFreeMemory(type, iter->second.real_address, g_loc);
		free(iter->second.address);
	} else {
		geckoFreeMemory(type, iter->second.address, g_loc);
	}

	iter->second.address = NULL;
	iter->second.allocated = false;

	geckoMemoryTable.erase(iter);

	return GECKO_SUCCESS;
}

GeckoError geckoFreeDistanceRealloc(int var_count, void **var_list) {

	for(int i=0;i<var_count;i++) {
		GeckoLocation *g_loc;
		GeckoLocationArchTypeEnum type;

		void *ptr = var_list[i];

		const auto iter = geckoMemoryTable.find(ptr);
		if (iter == geckoMemoryTable.end()) {
			fprintf(stderr, "===GECKO: Unable to find memory '%p'.\n", ptr);
			exit(1);
		}
		const int distance = iter->second.distance;
		if(distance != GECKO_DISTANCE_NEAR && distance != GECKO_DISTANCE_FAR)
			continue;
		if(iter->second.allocType != GECKO_DISTANCE_ALLOC_TYPE_REALLOC)
			continue;
		if(iter->second.real_address == NULL)
			continue;

		string &loc = iter->second.loc;
		g_loc = GeckoLocation::find(loc);
		if (g_loc == NULL) {
			fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", loc.c_str());
			exit(1);
		}

		geckoMemoryAllocationAlgorithm(g_loc, type);

		geckoFreeMemory(type, iter->second.real_address, g_loc);
		iter->second.real_address = NULL;

	}

	return GECKO_SUCCESS;
}




GeckoError geckoMemCpy(void *dest, int dest_start, size_t dest_count, void *src, int src_start, size_t src_count) {
	geckoInit();

	if(src_count != dest_count) {
		fprintf(stderr, "===GECKO: Source and destination in 'memcopy' operations should have same size (%ld != %ld). \n", dest_count, src_count);
		exit(1);
	}

	GeckoLocation *src_loc, *dest_loc;
	GeckoLocationArchTypeEnum src_type, dest_type;
	GeckoError err;

	err = __geckoFindLocationBasedOnPointer(src, src_type, &src_loc);
	if(err == GECKO_ERR_MEM_ADDRESS_NOT_FOUND) {
		fprintf(stderr, "===GECKO: Unable to find memory '%p'.\n", src);
		exit(1);
	} else if(err == GECKO_ERR_LOCATION_NOT_FOUND) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", src_loc->getLocationName().c_str());
		exit(1);
	}

	err = __geckoFindLocationBasedOnPointer(dest, dest_type, &dest_loc);
	if(err == GECKO_ERR_MEM_ADDRESS_NOT_FOUND) {
		fprintf(stderr, "===GECKO: Unable to find memory '%p'.\n", dest);
		exit(1);
	} else if(err == GECKO_ERR_LOCATION_NOT_FOUND) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", dest_loc->getLocationName().c_str());
		exit(1);
	}

	const GeckoLocationArchTypeEnum src_final_type = src_loc->getLocationType().type;
	const GeckoLocationArchTypeEnum dest_final_type = dest_loc->getLocationType().type;
	int is_host = (src_final_type == GECKO_X32 || dest_final_type == GECKO_X64);
	const auto iter = geckoMemoryTable.find(src);
	const size_t ds = iter->second.dataSize;

	if(is_host) {
		memcpy(dest, src, ds * src_count);
	}
#ifdef CUDA_ENABLED
	else {
		cudaMemcpy(dest, src, ds * src_count, cudaMemcpyDefault);
	}
#endif

	return GECKO_SUCCESS;
}

GeckoError geckoMemMove(void **addr, char *location) {
	geckoInit();

	const auto iter = geckoMemoryTable.find(*addr);
	if(iter == geckoMemoryTable.end()) {
		fprintf(stderr, "===GECKO: Unable to find memory '%p'.\n", *addr);
		exit(1);
	}

	void *temp;
	const GeckoMemory mem = iter->second;

	geckoMemoryDeclare((void**) &temp, mem.dataSize, mem.count, location, mem.distance, mem.distance_level, mem.allocType);
	geckoMemCpy(temp, 0, mem.count, *addr, 0, mem.count);
	bool is_dummy = mem.is_dummy;
	geckoFree(*addr);

	*addr = temp;
	geckoMemoryTable[*addr].is_dummy = is_dummy;

	return GECKO_SUCCESS;
}

GeckoError geckoMemRegister(void *addr, int start, size_t count, size_t dataSize, char *location) {
	geckoInit();

	GeckoMemory variable;

	variable.dataSize = dataSize;
	variable.count = count;
	variable.loc = string(location);
	GeckoLocation *pLocation = GeckoLocation::find(variable.loc);
	if(pLocation == NULL) {
		fprintf(stderr, "===GECKO %s (%d): Unable to find the location (%s)\n", __FILE__, __LINE__, location);
		exit(1);
	}
	variable.loc_ptr = (GeckoLocation*) pLocation;

	variable.address = addr;

	geckoMemoryTable[variable.address] = variable;

	return GECKO_SUCCESS;
}

GeckoError geckoMemUnregister(void *addr) {
	geckoInit();

	GeckoLocation *g_loc;
	GeckoLocationArchTypeEnum type;

	const auto iter = geckoMemoryTable.find(addr);
	if(iter == geckoMemoryTable.end()) {
		fprintf(stderr, "===GECKO: Unable to find memory '%p'.\n", addr);
		exit(1);
	}

	string &loc = iter->second.loc;
	g_loc = GeckoLocation::find(loc);
	if(g_loc == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", loc.c_str());
		exit(1);
	}

	iter->second.address = NULL;
	iter->second.allocated = false;

	geckoMemoryTable.erase(iter);

	return GECKO_SUCCESS;
}
