
#pragma once

#ifndef __GECKO_DATA_TYPES_H__
#define __GECKO_DATA_TYPES_H__

#include <string>
#include <unordered_map>

using namespace std;

typedef enum {
	GECKO_NO_ARCH = 0,
	GECKO_VIRTUAL,
	GECKO_X32,
	GECKO_X64,
	GECKO_NVIDIA,
	GECKO_UNIFIED_MEMORY,
	GECKO_PERMANENT_STORAGE,
	GECKO_UNKOWN,
	GECKO_DEVICE_LEN
} GeckoLocationArchTypeEnum;

typedef enum {
	GECKO_SUCCESS = 0,
	GECKO_ERR_FAILED,
	GECKO_ERR_TOTAL_ITERATIONS_ZERO,
	GECKO_ERR_MEM_ADDRESS_NOT_FOUND,
	GECKO_ERR_LOCATION_NOT_FOUND,
	GECKO_ERR_UNKNOWN,
	GECKO_ERR_LEN
} GeckoError;

class GeckoLocationType {
public:
	char *name;
	GeckoLocationArchTypeEnum type;
	int numCores;
	char *mem_size;
	char *mem_type;
	float bandwidth_GBps;
};

typedef enum {
	GECKO_DISTRIB_NONE = 0,
	GECKO_DISTRIB_DUPLICATE,
	GECKO_DISTRIB_DISTRIBUTE,
	GECKO_DISTRIB_TILE,
	GECKO_DISTRIB_UNKNOWN,
	GECKO_DISTRIB_LEN
} GeckoMemoryDistribPolicy;

//typedef enum {
//	GECKO_EXEC_POL_AT = 0,
//	GECKO_EXEC_POL_ATANY,
//	GECKO_EXEC_POL_ATEACH,
//	GECKO_EXEC_POL_UNKNOWN,
//	GECKO_EXEC_POL_LEN
//} GeckoExecutionPolicy;
//

typedef enum {
	GECKO_DISTANCE_NOT_SET = 0,
	GECKO_DISTANCE_NEAR,
	GECKO_DISTANCE_FAR,
	GECKO_DISTANCE_UNKNOWN,
	GECKO_DISTANCE_LEN
} GeckoDistanceTypeEnum;


typedef enum {
	GECKO_DISTANCE_ALLOC_TYPE_NOT_SET = 0,
	GECKO_DISTANCE_ALLOC_TYPE_REALLOC,
	GECKO_DISTANCE_ALLOC_TYPE_AUTO,
	GECKO_DISTANCE_ALLOC_TYPE_LEN
} GeckoDistanceAllocationTypeEnum;


class GeckoCUDAProp {
public:
	int deviceCountTotal;
	int deviceDeclared;

	/*
	 * Since at least we have one device, the default value for the device count is 1.
	 */
	explicit GeckoCUDAProp(int total=1, int declared=0) : deviceCountTotal(total), deviceDeclared(declared) {}
};



//typedef struct {
//	void **address;
//	size_t sz;
//	GeckoLocation_t loc;
//	bool duplicate;
//	bool distribute;
//	bool tile;
//
//	bool allocated;
//} GeckoVariable;

class GeckoLocation;


class GeckoMemory {
public:
	void 	*address;
	size_t 	dataSize;
	size_t 	count;
	string 	loc;
	GeckoLocation *loc_ptr;
//	GeckoMemoryDistribPolicy distributionType;

	bool 	allocated;
	GeckoDistanceTypeEnum distance;
	int 	distance_level;
	GeckoDistanceAllocationTypeEnum allocType;
	bool 	is_dummy;
	void 	*real_address;

	string 	originatedFrom;      // Name of the location that variable was originated from

	string 	filename_permanent;	// Filename for permanent memories
	int 	file_desc_id;

//	unordered_map<void*, string> memoryToLocMap;

	GeckoMemory() :
			address(NULL),
			dataSize(0),
			count(0),
//			distributionType(GECKO_DISTRIB_NONE),
			allocated(false),
			distance(GECKO_DISTANCE_NOT_SET),
			loc_ptr(NULL),
			is_dummy(false),
			distance_level(-1),
			allocType(GECKO_DISTANCE_ALLOC_TYPE_NOT_SET),
			real_address(NULL),
			filename_permanent(""),
			file_desc_id(-1)
	{};
};


class __geckoLocationIterationType {
public:
	GeckoLocation *loc;
	int iterationCount;
};


#endif
