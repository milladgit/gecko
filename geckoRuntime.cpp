
#include "geckoRuntime.h"
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



#define GECKO_ACQUIRE_SLEEP_DURATION_NS 100      // in nanoseconds
#define GECKO_LOCATION_INDEX_OFFSET 10


/*
 * Controlling how AcquireLocations function behaves.
 */
//#define GECKO_WAIT_ON_ALL_DEV_TO_BE_FREE



class GeckoCUDAProp {
public:
	int deviceCountTotal;
	int deviceDeclared;

	GeckoCUDAProp() : deviceCountTotal(1), deviceDeclared(0) {}
	GeckoCUDAProp(int total, int declared) : deviceCountTotal(total), deviceDeclared(declared) {}
};



//class GeckoAddressInfo {
//public:
//	void    *p;
//	size_t   count;
//	size_t   startingIndex;
//
//	GeckoAddressInfo() : p(NULL), count(0), startingIndex(0) {}
//	GeckoAddressInfo(void *p, size_t count, size_t startingIndex) : p(p), count(count), startingIndex(startingIndex) {}
//};




static GeckoCUDAProp geckoCUDA;
static unordered_map<string, GeckoLocationType> listOfAvailLocationTypes;
static unordered_map<void*, GeckoMemory> geckoMemoryTable;
//static unordered_map<void*, GeckoAddressInfo> geckoAddressTable;
static unordered_set<GeckoLocation*> freeResources;
static omp_lock_t lock_freeResources;
static unordered_map<int, string> geckoThreadDeviceMap;



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

		// for 'any' execution policy
		srand (time(NULL));

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
		GeckoLocationType d;
		d.type = GECKO_VIRTUAL;
		d.numCores = 0;
		d.mem_size = "";
		d.mem_type = "";
		listOfAvailLocationTypes[string("virtual")] = d;


		omp_init_lock(&lock_freeResources);


		atexit(geckoCleanup);

		#ifdef INFO
		fprintf(stderr, "===GECKO: Started...\n");
		#endif
	}
	return GECKO_SUCCESS;
}

void geckoCleanup() {
	// #ifdef DEBUG_SHOW_CLEANUP
	// chamListAllLocationtypes();
	// chamListAllLocations();
	// chamListAllVariables();
	// chamListAllMemoryAllocations();
	// #endif

	omp_destroy_lock(&lock_freeResources);


	#ifdef INFO
	fprintf(stderr, "===GECKO: Stopped...\n");
	#endif
}

inline
string &trim(string &str) {
	str.erase(0, str.find_first_not_of(" \t\n"));       //prefixing spaces
	str.erase(str.find_last_not_of(" \t\n")+1);         //surfixing spaces
	return str;
}
inline
string &toUpper(string &str) {
	std::transform(str.begin(), str.end(), str.begin(), ::toupper);
	return str;
}


inline
void __geckoGetFields(char *line, vector<string> &v, char *delim) {
	char* tmp = strdup(line);
	const char* tok;
	for (tok = strtok(line, delim);
		 tok && *tok;
		 tok = strtok(NULL, delim)) {
		string string_tok = string(tok);
		trim(string_tok);
		if(string_tok.size() == 0 || string_tok.compare(string("")) == 0)
			continue;
		v.push_back(string_tok);
	}
	free(tmp);
}

inline
void __geckoConfigFileLoadFile(char *filename, vector< vector<string> > &lines) {
	char line[1024];
	char *delim = ";\n";
	FILE *f = fopen(filename, "r");
	if(f == NULL) {
		fprintf(stderr, "===GECKO: Unable to load config file (%s).\n", filename);
		exit(1);
	}
	while (fgets(line, 1024, f)) {
		vector<string> fields;
		__geckoGetFields(line, fields, delim);
		if(fields.size() == 0)
			continue;
		lines.push_back(fields);
	}
	fclose(f);
}

inline
void __geckoLoadConfFileDeclLocType(vector<string> &fields) {
	string name, kind, num_cores, mem_size;
	for(int j=1;j<fields.size();j++) {
		vector<string> values;
		__geckoGetFields((char*)fields[j].c_str(), values, ",\n");
		if(values[0].compare("name") == 0)
			name = values[1];
		else if(values[0].compare("kind") == 0)
			kind = values[1];
		else if(values[0].compare("num_cores") == 0)
			num_cores = values[1];
		else if(values[0].compare("mem") == 0)
			mem_size = values[1];
	}

	trim(kind);
	toUpper(kind);
	trim(num_cores);

	if(name.compare("") == 0 || kind.compare("") == 0) {
		fprintf(stderr, "===GECKO: Error in declaring location type within the config file: name(%s) - "
				  "kind(%s) - num_cores(%s) - mem(%s)\n", name.c_str(), kind.c_str(), num_cores.c_str(), mem_size.c_str());
		exit(1);
	}

	GeckoLocationArchTypeEnum deviceType;
	if(kind.compare("X32") == 0)
		deviceType = GECKO_X32;
	else if(kind.compare("X64") == 0)
		deviceType = GECKO_X64;
	else if(kind.compare("CUDA") == 0)
		deviceType = GECKO_CUDA;
	else if(kind.compare("UNIFIED_MEMORY") == 0)
		deviceType = GECKO_UNIFIED_MEMORY;
	else
		deviceType = GECKO_UNKOWN;

	int num_cores_int = 0;
	if(num_cores.compare("") != 0)
		num_cores_int = stoi(num_cores, NULL, 10);

	geckoLocationtypeDeclare((char*)name.c_str(), deviceType, "", num_cores_int, mem_size.c_str(), "");

}

inline
void __geckoLoadConfFileLocDeclare(vector<string> &fields) {

	string type;
	vector<string> names;
	int all=0, start=0, count=1;
	for(int j=1;j<fields.size();j++) {
		vector<string> values;
		__geckoGetFields((char*)fields[j].c_str(), values, ",\n");
		if(values[0].compare("name") == 0) {
			for(int k=1;k<values.size();k++)
				names.push_back(values[k]);
		} else if(values[0].compare("type") == 0)
			type = values[1];
		else if(values[0].compare("all") == 0)
			all = 1;
		else if(values[0].compare("start") == 0)
			start = stoi(values[1], NULL, 10);
		else if(values[0].compare("count") == 0)
			count = stoi(values[1], NULL, 10);
	}

	if(names.size() == 0 || type.compare("") == 0) {
		for(int k=0;k<names.size();k++)
			fprintf(stderr, "===GECKO: Error in declaring location(s) within the config file: name(%s) - "
							"type(%s)\n", names[k].c_str(), type.c_str());
		exit(1);
	}

	for(int k=0;k<names.size();k++)
		geckoLocationDeclare(trim(names[k]).c_str(), trim(type).c_str(), all, start, count);

}

inline
void __geckoLoadConfFileHierDeclare(vector<string> &fields) {

	string op, parent;
	vector<string> children;
	int all=0, start=0, count=1;
	for(int j=1;j<fields.size();j++) {
		vector<string> values;
		__geckoGetFields((char*)fields[j].c_str(), values, ",\n");
		if(values[0].compare("children") == 0) {
			op = values[1];
			for(int k=2;k<values.size();k++)
				children.push_back(values[k]);
		} else if(values[0].compare("parent") == 0)
			parent = values[1];
		else if(values[0].compare("all") == 0)
			all = 1;
		else if(values[0].compare("start") == 0)
			start = stoi(values[1], NULL, 10);
		else if(values[0].compare("count") == 0)
			count = stoi(values[1], NULL, 10);
	}

	if(children.size() == 0 || parent.compare("") == 0 || (op.compare("+") == 0 && op.compare("-") == 0)) {
		for(int k=0;k<children.size();k++)
			fprintf(stderr, "===GECKO: Error in declaring hierarchy from  config file: children(%s) - "
							"op(%s) - parent(%s)\n", children[k].c_str(), op.c_str(), parent.c_str());
		exit(1);
	}

	for(int k=0;k<children.size();k++)
		geckoHierarchyDeclare(op[0], (char*) trim(children[k]).c_str(), trim(parent).c_str(), all, start, count);


}

GeckoError geckoLoadConfigWithFile(char *filename) {
	vector< vector<string> > lines;
	__geckoConfigFileLoadFile(filename, lines);

	for(int i=0;i<lines.size();i++) {
		vector<string> &fields = lines[i];

		if(fields[0].compare("loctype") == 0) {
			__geckoLoadConfFileDeclLocType(fields);
		} else if(fields[0].compare("location") == 0) {
			__geckoLoadConfFileLocDeclare(fields);
		} else if(fields[0].compare("hierarchy") == 0) {
			__geckoLoadConfFileHierDeclare(fields);
		}
	}

	return GECKO_SUCCESS;
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
//			geckoCUDA.deviceDeclared++;
		}
		#endif

		int index;
		if(locObj.type == GECKO_X32 || locObj.type == GECKO_X64) {
			index = geckoX64DeviceIndex++;
		}
		#ifdef CUDA_ENABLED
		else if(locObj.type == GECKO_CUDA) {
//			index = geckoCUDADeviceIndex++;
			index = geckoCUDA.deviceDeclared++;
		}
		#endif


		static int locationIndex = 0;
		// adding the newly created location to the map
		// that is persisted by GeckoLocation.
		GeckoLocation *loc = new GeckoLocation(string(&name[0]), NULL, locObj, index, GECKO_LOCATION_INDEX_OFFSET+locationIndex);
		freeResources.insert(loc);
		locationIndex++;

		#ifdef INFO
		fprintf(stderr, "===GECKO: Declaring location '%s' - index: %d - total declared: %d\n", &name[0], index, locationIndex);
		#endif		

	}

	return GECKO_SUCCESS;
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
	return GECKO_SUCCESS;
}

inline
void __geckoDrawPerNode(FILE *f, GeckoLocation *p) {
    if(p == NULL)
        return;
	char line[4096];
	string parentName = p->getLocationName();
	sprintf(&line[0], "\"%s\"  [shape=box];\n", parentName.c_str());
	fwrite(&line[0], sizeof(char), strlen(&line[0]), f);
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
	__geckoDrawPerNode(f, root);
	sprintf(&line[0], "}\n");
	fwrite(&line[0], sizeof(char), strlen(&line[0]), f);
	fclose(f);
}

void* geckoAllocateMemory(GeckoLocationArchTypeEnum type, GeckoMemory *var) {
	void *addr = NULL;
	size_t sz_in_byte = var->dataSize * var->count;
	GeckoLocation *device = GeckoLocation::find(var->loc);
	if(device == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", var->loc);
		exit(1);
	}

	switch(type) {
		case GECKO_X32:
		case GECKO_X64:
			addr = malloc(sz_in_byte);
#ifdef INFO
		fprintf(stderr, "===GECKO: MALLOC - location: %s - size: %d - addr: %p\n", var->loc.c_str(), sz_in_byte, addr);
#endif
			break;
#ifdef CUDA_ENABLED
		case GECKO_CUDA:
			acc_set_device_num(device->getLocationIndex(), acc_device_nvidia);
			GECKO_CUDA_CHECK(cudaMalloc(&addr, sz_in_byte));
#ifdef INFO
			fprintf(stderr, "===GECKO: CUDAMALLOC - location: %s - size: %d - addr: %p\n", var->loc.c_str(), sz_in_byte, addr);
#endif //INFO
			break;
#endif

		case GECKO_UNIFIED_MEMORY:
#ifdef CUDA_ENABLED
			GECKO_CUDA_CHECK(cudaMallocManaged(&addr, sz_in_byte));
#ifdef INFO
			fprintf(stderr, "===GECKO: UVM ALLOCATION - location: %s - size: %d - addr: %p\n", var->loc.c_str(), sz_in_byte, addr);
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
		const GeckoLocationArchTypeEnum type = node->getLocationType().type;
		if(type == GECKO_X32 || type == GECKO_X64 || type == GECKO_CUDA) {
			addr = geckoAllocateMemory(type, &var);
		} else {
			fprintf(stderr, "===GECKO (%s:%d): Unrecognized location type for allocation. Location: %s\n",
					__FILE__, __LINE__, var.loc.c_str());
			exit(1);
		}
	} else {
		const bool areAllCPUs = geckoAreAllChildrenCPU(node);
		if(areAllCPUs)
			addr = geckoAllocateMemory(GECKO_X64, &var);
		else
			addr = geckoAllocateMemory(GECKO_UNIFIED_MEMORY, &var);
	}

	var.address = addr;

	return GECKO_SUCCESS;
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

	return GECKO_SUCCESS;
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

void geckoAcquireLocations(vector<__geckoLocationIterationType> &locList) {
#ifdef GECKO_WAIT_ON_ALL_DEV_TO_BE_FREE
	const int count = locList.size();
	while(1) {
		omp_set_lock(&lock_freeResources);
		int i;
		for(i=0;i<count;i++) {
			if(freeResources.find(locList[i].loc) == freeResources.end()) {     // found a busy resource
				break;
			}
		}
		if(i < count) {
			omp_unset_lock(&lock_freeResources);
			usleep(GECKO_ACQUIRE_SLEEP_DURATION_NS);
			continue;
		}
		for(int i=0;i<count;i++) {
			GeckoLocation *device = locList[i].loc;
			const unordered_set<GeckoLocation *>::iterator &iter = freeResources.find(device);
			freeResources.erase(iter);
		}
		omp_unset_lock(&lock_freeResources);
		break;
	}
#else
	const int count = locList.size();
	omp_set_lock(&lock_freeResources);
	for(int i=0;i<count;i++) {
		GeckoLocation *device = locList[i].loc;
		const unordered_set<GeckoLocation *>::iterator &iter = freeResources.find(device);
		if(iter != freeResources.end())
			freeResources.erase(iter);
	}
	omp_unset_lock(&lock_freeResources);
#endif
}

void geckoAcquireLocationForAny(vector<__geckoLocationIterationType> &locList) {

#ifdef GECKO_WAIT_ON_ALL_DEV_TO_BE_FREE
	const int locListSize = locList.size();
	int *indexes = (int *) malloc(sizeof(int) * locListSize);
	while(1) {
		omp_set_lock(&lock_freeResources);
		int count = 0;
		int i;
		for(i=0;i<locListSize;i++) {
			if(freeResources.find(locList[i].loc) != freeResources.end()) {     // found a free resource
				indexes[count++] = i;
			}
		}
		if(count == 0) {
			omp_unset_lock(&lock_freeResources);
			usleep(GECKO_ACQUIRE_SLEEP_DURATION_NS);
			continue;
		}
		i = rand() % count;
		int index = indexes[i];
		GeckoLocation *device = locList[index].loc;
		const unordered_set<GeckoLocation *>::iterator &iter = freeResources.find(device);
		freeResources.erase(iter);
		__geckoLocationIterationType gliter = locList[index];
		locList.clear();
		locList.push_back(gliter);
		omp_unset_lock(&lock_freeResources);
		break;
	}
	free(indexes);

#else
	int i = rand() % ((int)locList.size());
	GeckoLocation *device = locList[i].loc;
	__geckoLocationIterationType gliter = locList[i];
	locList.clear();
	locList.push_back(gliter);
#endif

}


GeckoError geckoRegion(char *exec_pol, char *loc_at, size_t initval, size_t boundary,
                       int incremental_direction, int *devCount, int **out_beginLoopIndex, int **out_endLoopIndex,
                       GeckoLocation ***out_dev, int ranges_count, int *ranges) {
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

#ifdef INFO
	fprintf(stderr, "===GECKO: TotalIterations: %d\n", totalIterations);
#endif

	if(totalIterations == 0)
		return GECKO_SUCCESS;

	vector<__geckoLocationIterationType> children_names;
	geckoExtractChildrenFromLocation(location, children_names, (totalIterations >= 0 ? totalIterations : -1*totalIterations));
	*devCount = children_names.size();


	int loop_index_count = *devCount;
	if(strcmp(exec_pol, "range") == 0 || strcmp(exec_pol, "percentage") == 0)
		loop_index_count = ranges_count;

	int *beginLoopIndex = (int*) malloc(sizeof(int) * loop_index_count);
	int *endLoopIndex = (int*) malloc(sizeof(int) * loop_index_count);
	GeckoLocation **dev = (GeckoLocation**) malloc(sizeof(GeckoLocation*) * loop_index_count);



	if(strcmp(exec_pol, "static") == 0) {
		geckoAcquireLocations(children_names);

		int start = initval;
		for(int i=0;i<*devCount;i++) {
			int end = (incremental_direction ? start + children_names[i].iterationCount : start - children_names[i].iterationCount);
#ifdef INFO
			printf("\t\tChild %d: %s - share: %d - ", i, children_names[i].loc->getLocationName().c_str(),
				   children_names[i].iterationCount);
			printf("[%d, %d] at %p\n", start, end, children_names[i].loc);
#endif
			beginLoopIndex[i] = start;
			endLoopIndex[i] = end;
			dev[i] = children_names[i].loc;
			start = end;
		}
	} else if(strcmp(exec_pol, "flatten") == 0) {
		geckoAcquireLocations(children_names);

		int start, end, delta = totalIterations / (*devCount);
		start = initval;
		for(int i=0;i<*devCount;i++) {
			end = (incremental_direction ? start + delta : start - delta);
			if(i == *devCount-1)
				end = boundary;
#ifdef INFO
			printf("\t\tChild %d: %s - share: %d - ", i, children_names[i].loc->getLocationName().c_str(),
				   (end - start) * (incremental_direction ? 1 : -1)  );
			printf("[%d, %d] at %p\n", start, end, children_names[i].loc);
#endif
			beginLoopIndex[i] = start;
			endLoopIndex[i] = end;
			dev[i] = children_names[i].loc;
			start = end;
		}
	} else if(strcmp(exec_pol, "any") == 0) {
		geckoAcquireLocationForAny(children_names);
		*devCount = 1;
		beginLoopIndex[0] = initval;
		endLoopIndex[0] = boundary;
		dev[0] = children_names[0].loc;

#ifdef INFO
		fprintf(stderr, "===GECKO: Choosing location %s for 'any' execution policy.\n", dev[0]->getLocationName().c_str());
#endif

	} else if(strcmp(exec_pol, "range") == 0) {
		geckoAcquireLocations(children_names);

		int start, end, delta;
		start = initval;
		for(int j=0;j<ranges_count;j++) {
			int i = j % *devCount;
			delta = ranges[j];
			end = (incremental_direction ? start + delta : start - delta);
//			if(j == ranges_count-1)
//				end = boundary;
#ifdef INFO
			printf("\t\tChild %d: %s - share: %d - ", i, children_names[i].loc->getLocationName().c_str(),
			       (end - start) * (incremental_direction ? 1 : -1)  );
			printf("[%d, %d] at %p\n", start, end, children_names[i].loc);
#endif
			beginLoopIndex[j] = start;
			endLoopIndex[j] = end;
			dev[j] = children_names[i].loc;
			start = end;
		}
	} else if(strcmp(exec_pol, "percentage") == 0) {
		geckoAcquireLocations(children_names);

		int start, end, delta;
		start = initval;
		for(int j=0;j<ranges_count;j++) {
			int i = j % *devCount;
			delta = (int) floor(ranges[j] / 100.0 * totalIterations);
			end = (incremental_direction ? start + delta : start - delta);
//			if(j == ranges_count-1)
//				end = boundary;
#ifdef INFO
			printf("\t\tChild %d: %s - share: %d - ", i, children_names[i].loc->getLocationName().c_str(),
			       (end - start) * (incremental_direction ? 1 : -1)  );
			printf("[%d, %d] at %p\n", start, end, children_names[i].loc);
#endif
			beginLoopIndex[j] = start;
			endLoopIndex[j] = end;
			dev[j] = children_names[i].loc;
			start = end;
		}
	}



	*out_dev = dev;
	*out_beginLoopIndex = beginLoopIndex;
	*out_endLoopIndex = endLoopIndex;

	return GECKO_SUCCESS;
}

GeckoError geckoSetDevice(GeckoLocation *device) {
#ifdef INFO
	fprintf(stderr, "===GECKO: Setting device to %s\n", device->getLocationName().c_str());
#endif


	GeckoLocationArchTypeEnum loc_type = device->getLocationType().type;
	if(loc_type == GECKO_CUDA)
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

GeckoError geckoUnsetBusy(GeckoLocation *device) {
	omp_set_lock(&lock_freeResources);
	freeResources.insert(device);
	omp_unset_lock(&lock_freeResources);
	return GECKO_SUCCESS;
}

void geckoFreeRegionTemp(int *beginLoopIndex, int *endLoopIndex, int devCount, GeckoLocation **dev) {
	free(beginLoopIndex);
	free(endLoopIndex);

	free(dev);
}

GeckoError geckoWaitOnLocation(char *loc_at) {
	if(strlen(loc_at) == 0)
		return GECKO_SUCCESS;

	GeckoLocation *location = GeckoLocation::find(string(loc_at));
	if(location == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", loc_at);
		exit(1);
	}

	vector<__geckoLocationIterationType> children_names;
	geckoExtractChildrenFromLocation(location, children_names, 0);
	int devCount = children_names.size();

#pragma omp parallel num_threads(devCount)
//	for(int devIndex=0;devIndex<devCount;devIndex++)
	{
		int tid = omp_get_thread_num();
		GeckoLocation *loc = GeckoLocation::find(geckoThreadDeviceMap[tid]);
		if(loc != NULL) {
			geckoSetDevice(loc);
			int async_id = loc->getAsyncID();
#pragma acc wait(async_id)
			geckoUnsetBusy(loc);
		}
	}

	return GECKO_SUCCESS;
}

inline
void geckoFreeMemory(GeckoLocationArchTypeEnum type, void *addr, GeckoLocation *loc) {

	switch(type) {
		case GECKO_X32:
		case GECKO_X64:
			free(addr);
#ifdef INFO
		fprintf(stderr, "===GECKO: FREE - location %s \n", loc->getLocationName().c_str());
#endif
			break;
#ifdef CUDA_ENABLED
		case GECKO_CUDA:
			acc_set_device_num(loc->getLocationIndex(), acc_device_nvidia);
		case GECKO_UNIFIED_MEMORY:
			GECKO_CUDA_CHECK(cudaFree(addr));
#ifdef INFO
			fprintf(stderr, "===GECKO: CUDAFREE - location %s\n", loc->getLocationName().c_str());
#endif
			break;
#endif

		default:
			fprintf(stderr, "=== GECKO: Unrecognized architecture to free memory - Arch: %s\n",
					geckoGetLocationTypeName(type));
			exit(1);
	}
}

inline
GeckoError geckoMemoryFreeAlgorithm(GeckoLocationArchTypeEnum type, void *addr, GeckoLocation *node) {
	geckoInit();

/*
 * Following the memory allocation in the paper.
 */

	if(node->getChildren().size() == 0) {
//		This node is a leaf node!
		GeckoLocationArchTypeEnum type = node->getLocationType().type;
		if(type == GECKO_X32 || type == GECKO_X64 || type == GECKO_CUDA)
			geckoFreeMemory(type, addr, node);
		else {
			fprintf(stderr, "===GECKO (%s:%d): Unrecognized location type for freeing memory. Location: %s\n",
					__FILE__, __LINE__, node->getLocationName().c_str());
		}
	} else {
		bool areAllCPUs = geckoAreAllChildrenCPU(node);
		if(areAllCPUs) {
			geckoFreeMemory(GECKO_X64, addr, node);
		} else {
			geckoFreeMemory(GECKO_UNIFIED_MEMORY, addr, node);
		}
	}

	return GECKO_SUCCESS;
}

GeckoError geckoFree(void *ptr) {
	auto iter = geckoMemoryTable.find(ptr);
	if(iter == geckoMemoryTable.end())
		return GECKO_ERR_FAILED;

	string &loc = iter->second.loc;
	GeckoLocation *g_loc = GeckoLocation::find(loc);
	if(g_loc == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", loc);
		exit(1);
	}
	geckoMemoryFreeAlgorithm(g_loc->getLocationType().type, ptr, g_loc);
	geckoMemoryTable.erase(iter);

	return GECKO_SUCCESS;
}

GeckoError geckoBindLocationToThread(int threadID, GeckoLocation *loc) {
	geckoThreadDeviceMap[threadID] = loc->getLocationName();

	return GECKO_SUCCESS;
}
