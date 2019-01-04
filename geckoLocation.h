//
// Created by millad on 12/3/18.
//

#ifndef GECKO_GECKOLOCATION_H
#define GECKO_GECKOLOCATION_H

#include "geckoDataTypes.h"

const char* geckoGetLocationTypeName(GeckoLocationArchTypeEnum deviceType);
string 		geckoGetLocationTypeNameStr(GeckoLocationArchTypeEnum deviceType);

GeckoError 	geckoLocationtypeDeclare(char *name, GeckoLocationArchTypeEnum deviceType, const char *microArch,
									   int numCores, const char *mem_size, const char *mem_type, float bandwidth_GBps);
GeckoError 	geckoLocationDeclare(const char *name, const char *_type, int all, int start, int count);

GeckoError 	geckoSetDevice(GeckoLocation *device);
GeckoError 	geckoSetBusy(GeckoLocation *device);

GeckoError 	geckoBindLocationToThread(int threadID, GeckoLocation *loc);



#endif //GECKO_GECKOLOCATION_H
