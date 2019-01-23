//
// Created by millad on 12/3/18.
//

#include <unordered_map>
#include "geckoLocation.h"
#include "geckoHierarchicalTree.h"
#include "geckoDataTypeGenerator.h"


#ifdef _OPENACC
#include <openacc.h>
#include <unordered_set>
#include <omp.h>

#endif


#define GECKO_LOCATION_INDEX_OFFSET 10


extern GeckoError 	geckoInit();

extern unordered_map<string, GeckoLocationType> listOfAvailLocationTypes;
extern GeckoCUDAProp geckoCUDA;;
extern unordered_set<GeckoLocation*> freeResources;
extern omp_lock_t lock_freeResources;

unordered_map<int, string> geckoThreadDeviceMap;



const char* geckoGetLocationTypeName(GeckoLocationArchTypeEnum deviceType) {
	switch(deviceType) {
		case GECKO_NO_ARCH:
			return "NoArch";
		case GECKO_VIRTUAL:
			return "Virtual";
		case GECKO_UNIFIED_MEMORY:
			return "Unified Memory";
		case GECKO_X32:
			return "x32";
		case GECKO_X64:
			return "X64";
		case GECKO_NVIDIA:
			return "NVIDIA";
		case GECKO_PERMANENT_STORAGE:
			return "Permanent Storage";
		default:
			return "UNKNOWN";
	}
}

string geckoGetLocationTypeNameStr(GeckoLocationArchTypeEnum deviceType) {
	return string(geckoGetLocationTypeName(deviceType));
}

GeckoError geckoLocationtypeDeclare(char *name, GeckoLocationArchTypeEnum deviceType, const char *microArch,
									int numCores, const char *mem_size, const char *mem_type, float bandwidth_GBps) {
	geckoInit();

	GeckoLocationType d{};
	d.name = name;
	d.type = deviceType;
	d.numCores = numCores;
	d.mem_size = (char*)mem_size;
	d.mem_type = (char*)mem_type;
	d.bandwidth_GBps = bandwidth_GBps;
	listOfAvailLocationTypes[string(name)] = d;
#ifdef INFO
	fprintf(stderr, "===GECKO: Defining location type \"%s\" as %s \n", name, geckoGetLocationTypeName(deviceType));
#endif
	return GECKO_SUCCESS;
}



inline
void __geckoSetLocationType(const char *_type, GeckoLocationType &d) {
	if(_type == NULL || _type[0] == 0) {
		d.type = GECKO_NO_ARCH;
	} else {
		string type = string(_type);
		if (listOfAvailLocationTypes.find(type) == listOfAvailLocationTypes.end()) {
			fprintf(stderr, "=== GECKO: Unable to find the device type (%s)\n", _type);
			exit(1);
		}
		d = listOfAvailLocationTypes[type];
#ifdef INFO
		fprintf(stderr, "===GECKO: Loading type: %s\n", type.c_str(), geckoGetLocationTypeName(d.type));
#endif
	}

}

// implementing the all keyword.
inline
void __geckoLocDeclAll(const char *_type, int all, int &begin, int &end) {
	if(all) {
		GeckoLocationType locObj{};

		__geckoSetLocationType(_type, locObj);
		if(locObj.type == GECKO_X64) {
			// putting NUMA-related API calls in here
			begin = 0;
			end = 2;
		}
#ifdef CUDA_ENABLED
		else if(locObj.type == GECKO_NVIDIA) {
			begin = 0;
			end = geckoCUDA.deviceCountTotal;
		}
#else
		else if(locObj.type == GECKO_NVIDIA) {
			fprintf(stderr, "===GECKO: No CUDA is available on this system.\n");
			exit(1);
		}
#endif

	}
}

GeckoError geckoLocationDeclare(const char *_name, const char *_type, int all, int start, int count) {
	geckoInit();

	static int geckoX64DeviceIndex = 0;
	static int geckoCUDADeviceIndex = 0;

	int begin, end;
	if(start == -1) {
		begin = 0;
		end = 1;
	} else {
		begin = start;
		end = start + count;
	}

	__geckoLocDeclAll(_type, all, begin, end);

	for(int devID=begin;devID<end;devID++) {

		// extract the name of the device
		char name[128];
		if(all)
			sprintf(&name[0], "%s[%d]", _name, devID);
		else
			sprintf(&name[0], "%s", _name);

		if(GeckoLocation::find(&name[0]) != NULL) {
			fprintf(stderr, "===GECKO: Unable to declare duplicate location: %s\n", &name[0]);
			exit(1);
		}

		GeckoLocationType locObj{};

		__geckoSetLocationType(_type, locObj);

		int index = -1;
		if(locObj.type == GECKO_X32 || locObj.type == GECKO_X64) {
			index = geckoX64DeviceIndex++;
		}
#ifdef CUDA_ENABLED
		else if(locObj.type == GECKO_NVIDIA) {
			if (geckoCUDA.deviceDeclared >= geckoCUDA.deviceCountTotal) {
				fprintf(stderr, "===GECKO: Unable to declare more than available CUDA devices. "
								"Total Available: %d - Trying to declare: %d - Location: %s\n",
						geckoCUDA.deviceCountTotal, geckoCUDA.deviceDeclared + 1, &name[0]);
				exit(1);
			}
			index = geckoCUDA.deviceDeclared++;

			// if user asks for NVidia GPU devices, then we perform initialization of them.
			acc_init(acc_device_nvidia);
		}
#endif


		static int locationIndex = 0;

		// adding the newly created location to the map
		// that is persisted by GeckoLocation.
		GeckoLocation *loc = new GeckoLocation(string(&name[0]), NULL, locObj, index,
											   GECKO_LOCATION_INDEX_OFFSET+locationIndex);
		freeResources.insert(loc);
		locationIndex++;

#ifdef INFO
		fprintf(stderr, "===GECKO: Declaring location '%s' - type: %s - index: %d - total declared: %d\n", &name[0],
		        geckoGetLocationTypeName(locObj.type), index, locationIndex);
#endif

	}

	return GECKO_SUCCESS;
}


GeckoError geckoSetDevice(GeckoLocation *device) {

	GeckoLocationArchTypeEnum loc_type = device->getLocationType().type;

#ifdef INFO
	fprintf(stderr, "===GECKO: Setting device to %s - loctype: %s - location index: %d\n", device->getLocationName().c_str(), geckoGetLocationTypeName(loc_type), device->getLocationIndex());
#endif

	if(loc_type == GECKO_NVIDIA)
		acc_set_device_num(device->getLocationIndex(), acc_device_nvidia);
	else if(loc_type == GECKO_X64 || loc_type == GECKO_X32)
		acc_set_device_num(device->getLocationIndex(), acc_device_host);

	return GECKO_SUCCESS;
}


GeckoError geckoSetBusy(GeckoLocation *device) {
	omp_set_lock(&lock_freeResources);
	const unordered_set<GeckoLocation *>::iterator &iter = freeResources.find(device);
	if(iter == freeResources.end()) {
		omp_unset_lock(&lock_freeResources);
		return GECKO_ERR_FAILED;
	}
	freeResources.erase(iter);
	omp_unset_lock(&lock_freeResources);
	return GECKO_SUCCESS;
}


GeckoError geckoBindLocationToThread(int threadID, GeckoLocation *loc) {


#ifdef INFO
	fprintf(stderr, "===GECKO: Binding location (%s) to thread %d\n", (loc==NULL ? "NULL" : loc->getLocationName().c_str()), threadID);
#endif

	if(loc == NULL)
		return GECKO_ERR_FAILED;

	geckoThreadDeviceMap[threadID] = loc->getLocationName();

	return GECKO_SUCCESS;
}

