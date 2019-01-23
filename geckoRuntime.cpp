
#include "geckoRuntime.h"
#include "geckoStringUtils.h"
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>

#include <unordered_set>
#include <algorithm>
using namespace std;

#include <openacc.h>


#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include "geckoUtils.h"
#include "geckoDataTypeGenerator.h"


#define GECKO_ACQUIRE_SLEEP_DURATION_NS 100      // in nanoseconds


/*
 * Controlling how AcquireLocations function behaves.
 */
//#define GECKO_WAIT_ON_ALL_DEV_TO_BE_FREE





//class GeckoAddressInfo {
//public:
//	void    *ptr;
//	size_t   total_count;
//	size_t   startingIndex;
//
//	explicit
//	GeckoAddressInfo(void *p=NULL, size_t total_count=0, size_t startingIndex=0) :
//				ptr(p), total_count(total_count), startingIndex(startingIndex)
//				{}
//};
//
//static unordered_map<void*, GeckoAddressInfo> geckoAddressTable;



GeckoCUDAProp geckoCUDA;
unordered_map<string, GeckoLocationType> listOfAvailLocationTypes;
extern unordered_map<void*, GeckoMemory> geckoMemoryTable;
unordered_set<GeckoLocation*> freeResources;
omp_lock_t lock_freeResources;






GeckoLocation *geckoTreeHead = NULL;
int geckoStarted = 0;
int geckoCleanedup = 1;
bool   geckoPolicyRunTimeExists = false;
char  *geckoChosenPolicyRunTime = NULL;





extern GeckoError geckoMemoryDistribution(int loc_count, GeckoLocation **loc_list, int var_count, void **var_list,
								   int *beginIndex, int *endIndex);







void geckoCheckRunTimePolicy() {
	char *policy_run_time = getenv("GECKO_POLICY");
	if(policy_run_time != NULL) {
		geckoPolicyRunTimeExists = true;
		geckoChosenPolicyRunTime = strdup(policy_run_time);
#ifdef INFO
		fprintf(stderr, "===GECKO: Execution policy will be overridden by the chosen policy at runtime: %s.\n", policy_run_time);
#endif
	}
}

GeckoError geckoInit() {
	if(geckoStarted)
		return GECKO_SUCCESS;

	geckoStarted = 1;

	// for 'any' execution policy
	srand (time(NULL));

	// for nested OpenMP regions in case we target Multicore architectures
	omp_set_nested(1);

	geckoTreeHead = NULL;

	#ifdef CUDA_ENABLED
//		GECKO_CUDA_CHECK(cudaGetDeviceCount(&geckoCUDA.deviceCountTotal));
	geckoCUDA.deviceCountTotal = acc_get_num_devices(acc_device_nvidia);
	geckoCUDA.deviceDeclared = 0;
	#ifdef INFO
	fprintf(stderr, "===GECKO: CUDA Devices available(%d)\n", geckoCUDA.deviceCountTotal);
	#endif
	#endif

//		Defining Abstraction location type
	GeckoLocationType d{};
	d.type = GECKO_VIRTUAL;
	d.numCores = 0;
	d.mem_size = const_cast<char *>("");
	d.mem_type = const_cast<char *>("");
	listOfAvailLocationTypes[string("virtual")] = d;


//		Finding chosen policy at run time
	geckoCheckRunTimePolicy();


	omp_init_lock(&lock_freeResources);


	geckoCleanedup = 0;
	atexit(geckoCleanup);

	#ifdef INFO
	fprintf(stderr, "===GECKO: Started...\n");
	#endif

	return GECKO_SUCCESS;
}

void geckoCleanup() {
	if(geckoCleanedup)
		return;

	// #ifdef DEBUG_SHOW_CLEANUP
	// chamListAllLocationtypes();
	// chamListAllLocations();
	// chamListAllVariables();
	// chamListAllMemoryAllocations();
	// #endif

	omp_destroy_lock(&lock_freeResources);

//	Finding chosen policy at run time
	if(geckoChosenPolicyRunTime)
		free(geckoChosenPolicyRunTime);


	geckoStarted = 0;
	geckoCleanedup = 1;

	#ifdef INFO
	fprintf(stderr, "===GECKO: Stopped...\n");
	#endif
}





GeckoError geckoMemoryInternalTypeDeclare(gecko_type_base &Q, size_t dataSize, size_t count, char *location,
											GeckoDistanceTypeEnum distance) {
	GeckoLocationArchTypeEnum type;
	GeckoLocation *const pLocation = GeckoLocation::find(location);
	if(pLocation == NULL) {
		fprintf(stderr, "===GECKO %s (%d): Unable to find the location (%s)\n", __FILE__, __LINE__, location);
		exit(1);
	}
	geckoMemoryAllocationAlgorithm(pLocation, type);
	vector<__geckoLocationIterationType> childrenList;
	vector<int> childrenListFinal;

#ifdef INFO
	fprintf(stderr, "===GECKO: Allocating internal data type: Location(%s) - LocationType(%s) - Count(%d) \n", location, geckoGetLocationTypeName(type), count);
#endif

	switch(type) {
		case GECKO_X32:
		case GECKO_X64:
			Q.allocateMemOnlyHost(count);
			break;
		case GECKO_NVIDIA:
			Q.allocateMemOnlyGPU(count);
			break;
		case GECKO_UNIFIED_MEMORY:
			geckoExtractChildrenFromLocation(pLocation, childrenList, 0);
			for(int i=0;i<childrenList.size();i++) {
				GeckoLocationArchTypeEnum type = childrenList[i].loc->getLocationType().type;
				if(type == GECKO_X32 || type == GECKO_X64) {
					childrenListFinal.push_back(cudaCpuDeviceId);
				} else {
					childrenListFinal.push_back(childrenList[i].loc->getLocationIndex());
				}
			}
#ifdef INFO
			fprintf(stderr, "===GECKO: \tlocation list: ");
			for(int i=0;i<childrenList.size();i++)
				fprintf(stderr, "%s, ", childrenList[i].loc->getLocationName().c_str());
			fprintf(stderr, "\n");
#endif
			Q.allocateMem(count, childrenListFinal);
			break;
		default:
			fprintf(stderr, "=== GECKO: Unrecognized architecture for memory allocation - Arch: %s\n",
			        geckoGetLocationTypeName(type));
			exit(1);
	}

	return GECKO_SUCCESS;
}
