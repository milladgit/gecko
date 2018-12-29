//
// Created by millad on 11/28/18.
//

#ifndef GECKO_GECKOREGION_H
#define GECKO_GECKOREGION_H

#include "geckoDataTypes.h"

GeckoError  geckoRegion(char *exec_pol_chosen, char *loc_at, size_t initval, size_t boundary,
						int incremental_direction, int has_equal_sign, int *devCount,
						int **out_beginLoopIndex, int **out_endLoopIndex,
						GeckoLocation ***out_dev, int ranges_count, float *ranges, int var_count, void **var_list);


GeckoError 	geckoWaitOnLocation(char *locationName);

void 	   	geckoFreeRegionTemp(int *beginLoopIndex, int *endLoopIndex, int devCount, GeckoLocation **dev,
								int var_count, void **var_list, void **out_var_list);



#endif //GECKO_GECKOREGION_H
