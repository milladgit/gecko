//
// Created by millad on 11/28/18.
//

#ifndef GECKO_GECKOMEMORY_H
#define GECKO_GECKOMEMORY_H

#include <vector>
#include "geckoDataTypes.h"

using namespace std;

GeckoError 	geckoMemoryAllocationAlgorithm(GeckoLocation *node, GeckoLocationArchTypeEnum &output_type);
void 		geckoExtractChildrenFromLocation(GeckoLocation *loc, vector<__geckoLocationIterationType> &children_names,
										int iterationCount);

void* 		geckoAllocateMemory(GeckoLocationArchTypeEnum type, GeckoLocation *device, GeckoMemory *var);

GeckoError 	geckoFreeDistanceRealloc(int var_count, void **var_list);

GeckoError 	geckoMemoryDeclare(void **v, size_t dataSize, size_t count, char *location, GeckoDistanceTypeEnum distance,
							  int distance_level, GeckoDistanceAllocationTypeEnum allocationType);
GeckoError 	geckoFree(void *ptr);


GeckoError 	geckoMemCpy(void *dest, int dest_start, size_t dest_count, void *src, int src_start, size_t src_count);

GeckoError 	geckoMemMove(void **addr, char *location);

GeckoError 	geckoMemRegister(void *addr, int start, size_t count, size_t dataSize, char *location);
GeckoError 	geckoMemUnregister(void *addr);




#endif //GECKO_GECKOMEMORY_H
