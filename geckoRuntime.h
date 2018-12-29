
/*
 * Naming convetion:
 * Classes and Enumerations:
 * 		- Their names are in the "UpperCamelCase" convention.
 * 		- Start with fullname of the runtime: Gecko*
 *		- Variable members are "lowerCamelCase"
 *		- Methods are "lowerCamelCase"
 
 * Global static variables are in the "lowerCamelCase" convention.
 
 * API functions are in the "lowerCamelCase" convention.
 * 

 * Levels of information:
 *		- INFO: 	printing out the information to the stderr.
 *		- WARNING: 	printing out warnings to the stderr.
 *		- ERROR:	printing out error messages to the output. It will quit the program on reaching here.
*/


#pragma once

#ifndef __GECKO_RUNTIME_H__
#define __GECKO_RUNTIME_H__

#ifdef __cplusplus
#include <cstdlib>
#endif

#include <unordered_map>
#include <omp.h>
#include <openacc.h>

#include <string.h>

#include "geckoDataTypes.h"
#include "geckoDataTypeGenerator.h"
#include "geckoHierarchicalTree.h"
#include "geckoConfigLoader.h"
#include "geckoDraw.h"
#include "geckoMemory.h"
#include "geckoRegion.h"
#include "geckoHierarchy.h"
#include "geckoLocation.h"


#ifndef _OPENMP
#error Please enable OpenMP to use Gecko.
#endif

#ifndef _OPENACC
#error Please enable OpenACC to use Gecko.
#endif

using namespace std;


GeckoError 	geckoInit();
void 	   	geckoCleanup();





/*
 * This function is implementing the splitting method. The method refers to the approach where we divided a whole
 * array among multiple devices. Consider array A, A[0]...A[n-1], and k devices in our system. This function splits
 * the array A as following: A[0] ... A[n/4] is assigned to device 0, A[n/4+1] ... A[2*n/4] to device 1, and so on.
 * However, the data type should be defined among "gecko_<data-type>" types, like gecko_double for "double" variables
 * and gecko_long for "long" variables.
 *
 * Almost deprecated!
 *
 */
GeckoError 	geckoMemoryInternalTypeDeclare(gecko_type_base &Q, size_t dataSize, size_t count, char *location,
											GeckoDistanceTypeEnum distance);



#endif
