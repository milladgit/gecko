
#pragma once

#ifndef __GECKO_DATA_TYPES_H__
#define __GECKO_DATA_TYPES_H__

#include <string>
using namespace std;

typedef enum {
	GECKO_NO_ARCH = 0,
	GECKO_VIRTUAL,
	GECKO_X32,
	GECKO_X64,
	GECKO_NVIDIA,
	GECKO_UNIFIED_MEMORY,
	GECKO_UNKOWN,
	GECKO_DEVICE_LEN
} GeckoLocationArchTypeEnum;

typedef enum {
	GECKO_SUCCESS = 0,
	GECKO_ERR_FAILED,
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
};

typedef enum {
	GECKO_DISTRIB_NONE = 0,
	GECKO_DISTRIB_DUPLICATE,
	GECKO_DISTRIB_DISTRIBUTE,
	GECKO_DISTRIB_TILE,
	GECKO_DISTRIB_UNKNOWN,
	GECKO_DISTRIB_LEN
} GeckoMemoryDistribPolicy;

typedef enum {
	GECKO_EXEC_POL_AT = 0,
	GECKO_EXEC_POL_ATANY,
	GECKO_EXEC_POL_ATEACH,
	GECKO_EXEC_POL_UNKNOWN,
	GECKO_EXEC_POL_LEN
} GeckoExecutionPolicy;


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

class GeckoMemory {
public:
	void *address;
	size_t dataSize;
	size_t count;
	string loc;
	GeckoMemoryDistribPolicy distributionType;

	bool allocated;

	string originatedFrom;      // Name of the location that variable was originated from

	GeckoMemory() :
			address(NULL),
			dataSize(0),
			count(0),
			distributionType(GECKO_DISTRIB_NONE),
			allocated(false)
	{};
};

#endif
