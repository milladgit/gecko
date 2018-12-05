//
// Created by millad on 11/28/18.
//

#ifndef GECKO_GECKOMEMORY_H
#define GECKO_GECKOMEMORY_H

#include <vector>
#include "geckoDataTypes.h"

using namespace std;

GeckoError geckoMemoryAllocationAlgorithm(GeckoLocation *node, GeckoLocationArchTypeEnum &output_type);
void geckoExtractChildrenFromLocation(GeckoLocation *loc, vector<__geckoLocationIterationType> &children_names,
										int iterationCount);


GeckoError geckoMemoryDeclare(void **v, size_t dataSize, size_t count, char *location, GeckoDistanceTypeEnum distance);

GeckoError geckoFree(void *ptr);


#endif //GECKO_GECKOMEMORY_H
