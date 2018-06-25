
/*
 * Naming convetion:
 * Classes and Enumerations:
 * 		- Their name is with "UpperCamelCase" convention.
 * 		- Start with fullname of the runtime: Gecko*
 *		- Variable members are "lowerCamelCase"
 *		- Methods are "lowerCamelCase"
 
 * Global static variables are with "lowerCamelCase" convention.
 
 * API functions are with "lowerCamelCase" convention.
 * 

 * Levels of information:
 *		- INFO: 	printing out the information to the stderr.
 *		- WARNING: 	printing out warnings to the stderr.
 *		- ERROR:	printing out error messages to the output. It will quit the program on reaching here.
*/


#pragma once

#ifndef __GECKO_RUNTIME_H__
#define __GECKO_RUNTIME_H__

#include <cstdlib>
#include <unordered_map>
#include <omp.h>

#include "geckoDataTypes.h"
#include "geckoHierarchicalTree.h"

#ifndef _OPENMP
#error Please enable OpenMP to use Gecko.
#endif

#ifndef _OPENACC
#error Please enable OpenACC to use Gecko.
#endif


using namespace std;


GeckoError 	geckoInit();
void 	   	geckoCleanup();

GeckoError 	geckoLoadConfigWithFile(char *filename);
GeckoError 	geckoLoadConfigWithEnv();

GeckoError 	geckoLocationtypeDeclare(char *name, GeckoLocationArchTypeEnum deviceType, const char *microArch,
                                    int numCores, const char *mem_size, const char *mem_type);
GeckoError 	geckoLocationDeclare(const char *name, const char *_type, int all, int start, int count);
GeckoError 	geckoHierarchyDeclare(char operation, const char *child_name, const char *parent, int all, int start,
								 int count);
GeckoError 	geckoMemoryDeclare(void **v, size_t dataSize, size_t count, char *location);

GeckoError geckoRegion(char *exec_pol, char *loc_at, size_t initval, size_t boundary,
					   int incremental_direction, int has_equal_sign, int *devCount,
					   int **out_beginLoopIndex, int **out_endLoopIndex,
					   GeckoLocation ***out_dev, int ranges_count, int *ranges, int var_count, void **var_list);

GeckoError 	geckoSetDevice(GeckoLocation *device);
GeckoError 	geckoSetBusy(GeckoLocation *device);
GeckoError 	geckoUnsetBusy(GeckoLocation *device);
void 	   	geckoFreeRegionTemp(int *beginLoopIndex, int *endLoopIndex, int devCount, GeckoLocation **dev, void **var_list);

GeckoError 	geckoWaitOnLocation(char *locationName);

GeckoError 	geckoFree(void *ptr);

GeckoError 	geckoBindLocationToThread(int threadID, GeckoLocation *loc);

void 		geckoDrawHierarchyTree(char *rootNode, char *filename);

#endif
