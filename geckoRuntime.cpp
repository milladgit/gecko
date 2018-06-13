
#include "geckoRuntime.h"



class GeckoCUDAProp {
public:
	int deviceCountTotal;
	int deviceDeclared;

	GeckoCUDAProp() : deviceCountTotal(1), deviceDeclared(0) {}
	GeckoCUDAProp(int total, int declared) : deviceCountTotal(total), deviceDeclared(declared) {}
};

class GeckoAddressInfo {
public:
	void    *p;
	size_t   count;
	size_t   startingIndex;

	GeckoAddressInfo() : p(NULL), count(0), startingIndex(0) {}
	GeckoAddressInfo(void *p, size_t count, size_t startingIndex) : p(p), count(count), startingIndex(startingIndex) {}
};


static GeckoCUDAProp geckoCUDA;
static unordered_map<string, GeckoLocationType> listOfAvailLocationTypes;
static unordered_map<void*, GeckoMemory> geckoMemoryTable;
//static unordered_map<void*, GeckoAddressInfo> geckoAddressTable;


static inline 
char* geckoGetLocationTypeName(GeckoLocationArchTypeEnum deviceType) {
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
		case GECKO_CUDA:
			return "CUDA";
	}
	return "UNKNOWN";
}





static GeckoLocation *geckoTreeHead = NULL;
static int geckoStarted = 0;

GeckoError geckoInit() {
	if(!geckoStarted) {
		geckoStarted = 1;

		geckoTreeHead = NULL;

		#ifdef CUDA_ENABLED
		GECKO_CUDA_CHECK(cudaGetDeviceCount(&geckoCUDA.deviceCountTotal));
		if(geckoCUDA.deviceCountTotal != -1)
			geckoCUDA.deviceDeclared = 0;
		#ifdef INFO
		fprintf(stderr, "===GECKO: CUDA Devices available(%d)\n", geckoCUDA.deviceCountTotal);
		#endif
		#endif

//		Defining Abstraction location type
		GeckoLocationType d;
		d.type = GECKO_VIRTUAL;
		d.numCores = 0;
		d.mem_size = "";
		d.mem_type = "";
		listOfAvailLocationTypes[string("virtual")] = d;


		atexit(geckoCleanup);

		#ifdef INFO
		fprintf(stderr, "===GECKO: Started...\n");
		#endif
	}
	return GECKO_ERR_SUCCESS;
}

void geckoCleanup() {
	// #ifdef DEBUG_SHOW_CLEANUP
	// chamListAllLocationtypes();
	// chamListAllLocations();
	// chamListAllVariables();
	// chamListAllMemoryAllocations();
	// #endif

	#ifdef INFO
	fprintf(stderr, "===GECKO: Stopped...\n");
	#endif
}


GeckoError geckoLocationtypeDeclare(char *name, GeckoLocationArchTypeEnum deviceType, const char *microArch,
                                    int numCores, const char *mem_size, const char *mem_type) {
	geckoInit();

	GeckoLocationType d;
	d.name = name;
	d.type = deviceType;
	d.numCores = numCores;
	d.mem_size = (char*)mem_size;
	d.mem_type = (char*)mem_type;
	listOfAvailLocationTypes[string(name)] = d;
	#ifdef INFO
	fprintf(stderr, "===GECKO: Defining location type \"%s\" as (%s) \n", name,
	        geckoGetLocationTypeName(deviceType));
	#endif
	return GECKO_ERR_SUCCESS;
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

	if(all) {
		GeckoLocationType locObj;

		__geckoSetLocationType(_type, locObj);
		#ifdef CUDA_ENABLED
		if(locObj.type == GECKO_CUDA) {
			begin = 0;
			end = geckoCUDA.deviceCountTotal;
		}
		#else
		if(locObj.type == GECKO_CUDA) {
			fprintf(stderr, "===GECKO: No CUDA is available on this system.\n");
			exit(1);			
		}
		#endif

		if(locObj.type == GECKO_X64) {
			// putting NUMA-related API calls in here
			begin = 0;
			end = 2;
		}
	}

	for(int devID=begin;devID<end;devID++) {

		char name[128];
		if(all)
			sprintf(&name[0], "%s[%d]", _name, devID);
		else
			sprintf(&name[0], "%s", _name);

		if(GeckoLocation::find(&name[0]) != NULL) {
			fprintf(stderr, "===GECKO: Unable to declare duplicate location: %s\n", &name[0]);
			exit(1);			
		}

		GeckoLocationType locObj;

		__geckoSetLocationType(_type, locObj);

		#ifdef CUDA_ENABLED
		if(locObj.type == GECKO_CUDA) {
			if (geckoCUDA.deviceDeclared >= geckoCUDA.deviceCountTotal) {
				fprintf(stderr, "===GECKO: Unable to declare more than available CUDA devices. "
						        "Total Available: %d - Trying to declare: %d - Location: %s\n",
				        geckoCUDA.deviceCountTotal, geckoCUDA.deviceDeclared + 1, &name[0]);
				exit(1);
			}
			geckoCUDA.deviceDeclared++;
		}
		#endif

		int index;
		if(locObj.type == GECKO_X32 || locObj.type == GECKO_X64) {
			index = geckoX64DeviceIndex++;
		}
		#ifdef CUDA_ENABLED
		if(locObj.type == GECKO_CUDA) {
			index = geckoCUDADeviceIndex++;
		}
		#endif


		new GeckoLocation(string(&name[0]), NULL, locObj, index);
		#ifdef INFO
		fprintf(stderr, "===GECKO: Declaring location %s\n", &name[0]);
		#endif		

	}

	return GECKO_ERR_SUCCESS;
}


GeckoError geckoHierarchyDeclare(char operation, const char *child_name, const char *parent, int all, int start, int count) {
	geckoInit();

	if(operation != '+' && operation != '-') {
		fprintf(stderr, "===GECKO: Unrecognizeable operation ('%c') in hierarchy declaration for location '%s'.\n", operation, child_name);
		exit(1);
	}


	int begin, end;
	if(start == -1) {
		begin = 0;
		end = 1;
	} else {
		begin = start;
		end = start + count;
	}

	if(all) {
		char name[128];
		sprintf(&name[0], "%s[%d]", child_name, 0);
		GeckoLocation *child = GeckoLocation::find(string(name));
		if(child == NULL) {
			fprintf(stderr, "===GECKO: Unable to find (%s)!\n", child_name);
			exit(1);
		}
		GeckoLocationType locObj = child->getLocationType();
		// __geckoSetLocationType(_type, locObj);
		#ifdef CUDA_ENABLED
		if(locObj.type == GECKO_CUDA) {
			begin = 0;
			end = geckoCUDA.deviceCountTotal;
		}
		#else
		if(locObj.type == GECKO_CUDA) {
			fprintf(stderr, "===GECKO: No CUDA is available on this system.\n");
			exit(1);			
		}
		#endif

		if(locObj.type == GECKO_X64) {
			// putting NUMA-related API calls in here
			begin = 0;
			end = 2;
		}
	}


	GeckoLocation *parentNode = NULL;
	parentNode = GeckoLocation::find(string(parent));
	if(parentNode == NULL) {
		fprintf(stderr, "===GECKO: Unable to find parent (%s)!\n", parent);
		exit(1);
	}
	#ifdef INFO
	char operation_name[16];
	#endif
	if(operation == '+') {
		#ifdef INFO
		strcpy(&operation_name[0], "Declaring");
		#endif
	}
	else if(operation == '-') {
		#ifdef INFO
		strcpy(&operation_name[0], "Removing");
		#endif
	}


	for(int devID=begin;devID<end;devID++) {

		char name[128];
		if(all)
			sprintf(&name[0], "%s[%d]", child_name, devID);
		else
			sprintf(&name[0], "%s", child_name);

		GeckoLocation *childNode =  GeckoLocation::find(string(&name[0]));
		if(childNode == NULL) {
			fprintf(stderr, "===GECKO (%s:%d): Unable to find child (%s)!\n", __FILE__, __LINE__, &name[0]);
			exit(1);
		}

		if(operation == '+') {
			childNode->setParent(parentNode);
			parentNode->appendChild(childNode);
		} else {
			childNode->setParent(NULL);
			parentNode->removeChild(childNode);
		}

		#ifdef INFO
		fprintf(stderr, "===GECKO: %s '%s' as child of '%s'.\n", &operation_name[0], &name[0], parent);
		#endif

	}
	return GECKO_ERR_SUCCESS;
}

inline
void __geckoDrawPerNode(FILE *f, GeckoLocation *p) {
    if(p == NULL)
        return;
	char line[4096];
	string parentName = p->getLocationName();
    const vector<GeckoLocation *> &children = p->getChildren();
    const int size = children.size();
    for(int i=0;i<size;i++) {
        const string &childName = children[i]->getLocationName();
        sprintf(&line[0], "\"%s\" -> \"%s\";\n", parentName.c_str(), childName.c_str());
        fwrite(&line[0], sizeof(char), strlen(&line[0]), f);

        __geckoDrawPerNode(f, children[i]);
    }
}

void geckoDrawHierarchyTree(char *rootNode, char *filename) {
	FILE *f;
	char line[128];
	if(filename == NULL || strlen(filename) == 0) {
		#ifdef WARNING
		fprintf(stderr, "===GECKO: Incorrect output filename for 'draw' clause.\n");
		#endif
		return;
	}
	f = fopen(filename, "w");
	sprintf(&line[0], "digraph Gecko {\n");
	fwrite(&line[0], sizeof(char), strlen(&line[0]), f);
	GeckoLocation *root = GeckoLocation::find(string(rootNode));
	fprintf(stderr, "===GECKO: Root node: %s\n", root->getLocationName().c_str());
	__geckoDrawPerNode(f, root);
	sprintf(&line[0], "}\n");
	fwrite(&line[0], sizeof(char), strlen(&line[0]), f);
	fclose(f);
}

void* geckoAllocateMemory(GeckoLocationArchTypeEnum type, GeckoMemory *var) {
	void *addr = NULL;

	switch(type) {
		case GECKO_X32:
		case GECKO_X64:
			addr = malloc(var->dataSize * var->count);
#ifdef INFO
		fprintf(stderr, "===GECKO: MALLOC - location %s - size %d\n", var->loc.c_str(),
			        var->dataSize * var->count);
#endif
			break;
#ifdef CUDA_ENABLED
		case GECKO_CUDA:
			CHAM_CUDA_CHECK(cudaMalloc(&addr, var->dataSize * var->count));
#ifdef INFO
			fprintf(stderr, "===GECKO: CUDAMALLOC - location %s - size %d\n", var->loc.c_str(),
			        var->dataSize * var->count);
#endif //INFO
			break;
#endif

		case GECKO_UNIFIED_MEMORY:
#ifdef CUDA_ENABLED
			CHAM_CUDA_CHECK(cudaMallocManaged(&addr, var->dataSize * var->count));
#ifdef INFO
			fprintf(stderr, "===GECKO: UVM ALLOCATION - location %s - size %d\n", var->loc.c_str(),
			        var->dataSize * var->count);
#endif  // INFO
#else   // CUDA_ENABLED
			fprintf(stderr, "===GECKO: CUDA APIs are required for unified memory allocation.\n");
			exit(1);
#endif  // CUDA_ENABLED
			break;


		default:
			fprintf(stderr, "=== GECKO: Unrecognized architecture for memory allocation - Arch: %s\n",
					geckoGetLocationTypeName(type));
			exit(1);
	}


//	geckoAddressTable[addr] = GeckoAddressInfo(addr, var->count, 0);

#ifdef INFO
	fprintf(stderr, "===GECKO: ALLOCATION variable %p - location %s - addr %p\n", addr, var->loc.c_str(), addr);
#endif

	return addr;
}

inline
bool geckoAreAllChildrenCPU(GeckoLocation *node) {
	if(node == NULL)
		return true;
	int sz = node->getChildren().size();
	if(sz == 0) {
		if(node->getLocationType().type == GECKO_X32 || node->getLocationType().type == GECKO_X64)
			return true;
		else
			return false;
	}
	for(int i=0;i<sz;i++) {
		if(!geckoAreAllChildrenCPU(node->getChildren()[i]))
			return false;
	}
	return true;
}

inline
GeckoError geckoMemoryAllocationAlgorithm(GeckoMemory &var) {
	geckoInit();

/*
 * Following the memory allocation in the paper.
 */

	void *addr = NULL;

	GeckoLocation *node = GeckoLocation::find(var.loc);
	if(node->getChildren().size() == 0) {
//		This node is a leaf node!
		GeckoLocationArchTypeEnum type = node->getLocationType().type;
		if(type == GECKO_X32 || type == GECKO_X64 || type == GECKO_CUDA)
			addr = geckoAllocateMemory(type, &var);
		else {
			fprintf(stderr, "===GECKO (%s:%d): Unrecognized location type for allocation. Location: %s\n",
					__FILE__, __LINE__, var.loc.c_str());
		}
	} else {
		bool areAllCPUs = geckoAreAllChildrenCPU(node);
		if(areAllCPUs) {
			addr = geckoAllocateMemory(GECKO_X64, &var);
		} else {
			addr = geckoAllocateMemory(GECKO_UNIFIED_MEMORY, &var);
		}
	}

	var.address = addr;

	return GECKO_ERR_SUCCESS;
}

GeckoError geckoMemoryDeclare(void **v, size_t dataSize, size_t count, char *location) {
	geckoInit();

	GeckoMemory variable;

	variable.dataSize = dataSize;
	variable.count = count;
	GeckoLocation *const pLocation = GeckoLocation::find(string(location));
	if(pLocation == NULL) {
		fprintf(stderr, "===GECKO %s (%d): Unable to find the location (%s)\n", __FILE__, __LINE__, location);
		exit(1);
	}
	variable.loc = string(location);

	geckoMemoryAllocationAlgorithm(variable);

	geckoMemoryTable[variable.address] = variable;

	*v = variable.address;

	return GECKO_ERR_SUCCESS;
}

class __geckoLocationIterationType {
public:
	GeckoLocation *loc;
	int iterationCount;
};

void geckoExtractChildrenFromLocation(GeckoLocation *loc, vector<__geckoLocationIterationType> &children_names, int iterationCount) {
	const vector<GeckoLocation *> &v = loc->getChildren();
	const size_t sz = v.size();
	if(sz == 0) {
		__geckoLocationIterationType git;
		git.loc = loc;
		git.iterationCount = iterationCount;
		children_names.push_back(git);
		return;
	}
	for(int i=0;i<sz;i++) {
		__geckoLocationIterationType git;
		git.loc = v[i];
		if(i == sz - 1)
			git.iterationCount = iterationCount - iterationCount/sz*(sz-1);
		else
			git.iterationCount = iterationCount / sz;

		if(v[i]->getChildren().size() == 0)
			children_names.push_back(git);
		else
			geckoExtractChildrenFromLocation(v[i], children_names, git.iterationCount);
	}
}

GeckoError geckoRegion(char *exec_pol, char *loc_at, size_t initval, size_t boundary, int incremental_direction,
					   int *devCount, int *beginLoopIndex, int *endLoopIndex, GeckoLocation **dev) {
	geckoInit();

#ifdef INFO
	fprintf(stderr, "===GECKO: Execution policy (%s) at location (%s)\n", exec_pol, loc_at);
#endif

	*devCount = 0;

	GeckoLocation *location = GeckoLocation::find(string(loc_at));
	if(location == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", loc_at);
		exit(1);
	}

	int totalIterations = boundary - initval;

	printf("TotalIterations: %d\n", totalIterations);

	if(totalIterations == 0)
		return GECKO_ERR_SUCCESS;

	vector<__geckoLocationIterationType> children_names;
	geckoExtractChildrenFromLocation(location, children_names, (totalIterations >= 0 ? totalIterations : -1*totalIterations));
	*devCount = children_names.size();

	beginLoopIndex = (int*) malloc(sizeof(int) * (*devCount));
	endLoopIndex = (int*) malloc(sizeof(int) * (*devCount));
	dev = (GeckoLocation**) malloc(sizeof(GeckoLocation*) * (*devCount));

	if(strcmp(exec_pol, "static") == 0) {
		int start = initval;
		for(int i=0;i<*devCount;i++) {
			printf("\t\tChild %d: %s - share: %d - ", i, children_names[i].loc->getLocationName().c_str(), children_names[i].iterationCount);
			int end = (incremental_direction ? start + children_names[i].iterationCount : start - children_names[i].iterationCount);
			beginLoopIndex[i] = start;
			endLoopIndex[i] = end;
			dev[i] = children_names[i].loc;
			printf("[%d, %d]\n", start, end);
			if(incremental_direction)
				start += children_names[i].iterationCount;
			else
				start -= children_names[i].iterationCount;
		}
	} else if(strcmp(exec_pol, "static") == 0) {

	} else if(strcmp(exec_pol, "flatten") == 0) {

	}

	return GECKO_ERR_SUCCESS;
}

GeckoError geckoSetDevice(GeckoLocation *device) {
#ifdef INFO
	fprintf(stderr, "===GECKO: Setting device to %s\n", device->getLocationName().c_str());
#endif

	return GECKO_ERR_SUCCESS;
}

void geckoFreeRegionTemp(int *beginLoopIndex, int *endLoopIndex, int devCount, GeckoLocation *dev) {
	free(beginLoopIndex);
	free(endLoopIndex);

	free(dev);
}

//GeckoError geckoRegion(chameleonExecutionPolicy_t policy, char *loc_list, int count, void ***varlist) {
//	geckoInit();
//
//	for(int i=0;i<count;i++) {
//		void **v = varlist[i];
//		printf("===%d : %p\n", i, v);
//	}
//
//	string locList(loc_list);
//
//	if(policy == CHAMELEON_EXEC_POL_AT) {
//		lastParallelNode = ChameleonNode::find(locList);
//		if(lastParallelNode == NULL) {
//			fprintf(stderr, "===CHAMELEON: Device ('%s') does not exist!\n", loc_list);
//			exit(1);
//		}
////		switch(lastParallelNode->getLocationType().type) {
////#ifdef CUDA_ENABLED
////			case CHAMELEON_CUDA:
//////				acc_set_device_type(acc_device_nvidia);
////				acc_set_device_num(lastParallelNode->getLocationIndex(), acc_device_nvidia);
////#ifdef DEBUG
////				fprintf(stderr, "===CHAMELEON: Setting device to NVIDIA (%d)\n", lastParallelNode->getLocationIndex());
////#endif
////				break;
////#endif
////			case CHAMELEON_X32:
////			case CHAMELEON_X64:
////				acc_set_device_num(lastParallelNode->getLocationIndex(), acc_device_host);
////#ifdef DEBUG
////				fprintf(stderr, "===CHAMELEON: Setting device to HOST (%d)\n", lastParallelNode->getLocationIndex());
////#endif
////				break;
////		}
//		chamSetDevice(lastParallelNode->getLocationType().type, lastParallelNode->getLocationIndex());
//	} else if(policy == CHAMELEON_EXEC_POL_ATANY) {
//		fprintf(stderr, "===CHAMELEON: atany execution policy is not defined yet.\n");
//		exit(1);
//	} else if(policy == CHAMELEON_EXEC_POL_ATEACH) {
//		fprintf(stderr, "===CHAMELEON: ateach execution policy is not defined yet.\n");
//		exit(1);
//	} else if(policy == CHAMELEON_EXEC_POL_UNKNOWN) {
//		fprintf(stderr, "===CHAMELEON: unknown execution policy is not defined yet.\n");
//		exit(1);
//	}
//
//	return CHAMELEON_ERR_SUCCESS;
//}
