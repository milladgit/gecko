//
// Created by millad on 11/28/18.
//

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <omp.h>
#include "geckoRegion.h"
#include "geckoHierarchicalTree.h"
#include "geckoStringUtils.h"
#include "geckoMemory.h"

#ifdef _OPENACC
#include <openacc.h>
#endif

// for malloc
#include <stdlib.h>
#include <cstdlib>


using namespace std;

extern GeckoCUDAProp geckoCUDA;
extern unordered_map<string, GeckoLocationType> listOfAvailLocationTypes;
extern unordered_map<void*, GeckoMemory> geckoMemoryTable;
//static unordered_map<void*, GeckoAddressInfo> geckoAddressTable;
extern unordered_set<GeckoLocation*> freeResources;
extern omp_lock_t lock_freeResources;
extern unordered_map<int, string> geckoThreadDeviceMap;


extern GeckoLocation *geckoTreeHead;
extern int geckoStarted;
extern bool   geckoPolicyRunTimeExists;
extern char  *geckoChosenPolicyRunTime;



extern void geckoInit();
extern void geckoExtractChildrenFromLocation(GeckoLocation *loc, vector<__geckoLocationIterationType> &children_names,
			int iterationCount);
extern GeckoError geckoMemoryDistribution(int loc_count, GeckoLocation **loc_list, int var_count, void **var_list,
					int *beginIndex, int *endIndex);


void geckoAcquireLocations(vector<__geckoLocationIterationType> &locList) {
#ifdef GECKO_WAIT_ON_ALL_DEV_TO_BE_FREE
	const int total_count = locList.size();
	while(1) {
		omp_set_lock(&lock_freeResources);
		int i;
		for(i=0;i<total_count;i++) {
			if(freeResources.find(locList[i].loc) == freeResources.end()) {     // found a busy resource
				break;
			}
		}
		if(i < total_count) {
			omp_unset_lock(&lock_freeResources);
			usleep(GECKO_ACQUIRE_SLEEP_DURATION_NS);
			continue;
		}
		for(int i=0;i<total_count;i++) {
			GeckoLocation *device = locList[i].loc;
			const unordered_set<GeckoLocation *>::iterator &iter = freeResources.find(device);
			freeResources.erase(iter);
		}
		omp_unset_lock(&lock_freeResources);
		break;
	}
#else
	const int count = (int) locList.size();
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

void geckoAcquireLocationForAny(vector<__geckoLocationIterationType> &locList, int &acquiredLocation) {

#ifdef GECKO_WAIT_ON_ALL_DEV_TO_BE_FREE
	const int locListSize = locList.size();
	int *indexes = (int *) malloc(sizeof(int) * locListSize);
	while(1) {
		omp_set_lock(&lock_freeResources);
		int total_count = 0;
		int i;
		for(i=0;i<locListSize;i++) {
			if(freeResources.find(locList[i].loc) != freeResources.end()) {     // found a free resource
				indexes[total_count++] = i;
			}
		}
		if(total_count == 0) {
			omp_unset_lock(&lock_freeResources);
			usleep(GECKO_ACQUIRE_SLEEP_DURATION_NS);
			continue;
		}
		i = rand() % total_count;
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
	acquiredLocation = i;
#endif

}

inline
void geckoBindThreadsToAccDevices(int *devCount) {
	if(GeckoLocation::getAllLeavesOnce(devCount)) {
		vector<GeckoLocation *> &locList = GeckoLocation::getChildListForThreads();
		// If hierarchy has changed since the last call to above function,
		// reassign OpenACC devices to OpenMP threads.
#pragma omp parallel num_threads(*devCount)
		{
			int id = omp_get_thread_num();
			GeckoLocation *loc = locList[id];

			if(loc->getLocationType().type == GECKO_X32 || loc->getLocationType().type == GECKO_X64) {
				acc_set_device_num(loc->getLocationIndex(), acc_device_host);
				loc->setThreadID(id);
#ifdef INFO
				fprintf(stderr, "===GECKO: Assigning location %s to thread id %d\n", loc->getLocationName().c_str(), id);
#endif

			} else if(loc->getLocationType().type == GECKO_NVIDIA) {
				acc_set_device_num(loc->getLocationIndex(), acc_device_nvidia);
				loc->setThreadID(id);
#ifdef INFO
				fprintf(stderr, "===GECKO: Assigning location %s to thread id %d\n", loc->getLocationName().c_str(), id);
#endif

			}

		}
	}
}

bool __geckoParseRangePercentagePolicy(char *exec_pol, string &exec_pol_return, int &ranges_count, float **ranges) {
	char *percentage_policy = const_cast<char *>("percentage");
	char *range_policy = const_cast<char *>("range");
	vector<string> fields;
	char *__exec_pol = strdup(exec_pol);
	__geckoGetFields(__exec_pol, fields, const_cast<char *>(":\n"));
	free(__exec_pol);

	if(fields.size() == 1) {
		/*
		 * It is been already parsed on the Python script or compiler side.
		 */
		exec_pol_return = fields[0];
		return false;
	}
	if(fields[0] == percentage_policy)
		exec_pol_return = string(percentage_policy);
	else if(fields[0] == range_policy)
		exec_pol_return = string(range_policy);

	string list_of_numbers = fields[1];
	if(list_of_numbers[0] != '[' || list_of_numbers[list_of_numbers.size()-1] != ']') {
		fprintf(stderr, "===GECKO: Error in number list provided for the execution policy (%s)\n", exec_pol);
		exit(1);
	}
	list_of_numbers = list_of_numbers.substr(1, list_of_numbers.size()-2);
	char *tmp = strdup(list_of_numbers.c_str());
	__geckoGetFields(tmp, fields, ",");
	free(tmp);

	ranges_count = static_cast<int>(fields.size());
	float *r = (float*) malloc(sizeof(float) * ranges_count);
	for(int i=0;i<ranges_count;i++)
		r[i] = strtof(fields[i].c_str(), nullptr);

	*ranges = r;

	return true;
}

inline
void __geckoExecPolStatic(const size_t initval, const size_t boundary, const int incremental_direction,
						  vector<__geckoLocationIterationType> &children_names, int *beginLoopIndex,
						  int *endLoopIndex, GeckoLocation ** dev) {

	geckoAcquireLocations(children_names);

	int start = (int) initval, end;
	const int childCount = static_cast<const int>(children_names.size());

	for(int i=0;i<childCount;i++) {

		if(i == childCount - 1)
			end = (int) boundary;
		else
			end = (incremental_direction ? start + children_names[i].iterationCount : start -
					children_names[i].iterationCount);

	#ifdef INFO
		fprintf(stderr, "===GECKO:\tChild %d: %s - share: %d - ", i, children_names[i].loc->getLocationName().c_str(),
			   children_names[i].iterationCount);
		fprintf(stderr, "[%d, %d]\n", start, end);
	#endif

		GeckoLocation *loc = children_names[i].loc;
		int loc_thread_id = loc->getThreadID();
		dev[loc_thread_id] = loc;
		beginLoopIndex[loc_thread_id] = start;
		endLoopIndex[loc_thread_id] = end;

		start = end;
	}
}

inline
void
__geckoExecPolFlatten(const size_t initval, const size_t boundary, const int incremental_direction, const int *devCount,
					  int totalIterations, vector<__geckoLocationIterationType> &children_names, int *beginLoopIndex,
					  int *endLoopIndex, GeckoLocation **dev) {
	geckoAcquireLocations(children_names);

	int start, end;
	start = initval;
	const int &childCount = static_cast<const int &>(children_names.size());
	int delta = totalIterations / childCount;
	for(int i=0;i<childCount;i++) {

		if(i == childCount-1)
			end = boundary;
		else
			end = (incremental_direction ? start + delta : start - delta);
	#ifdef INFO
		fprintf(stderr, "\t\tChild %d: %s - share: %d - ", i, children_names[i].loc->getLocationName().c_str(),
			   (end - start) * (incremental_direction ? 1 : -1)  );
		fprintf(stderr, "[%d, %d] at %p\n", start, end, children_names[i].loc);
	#endif

		GeckoLocation *loc = children_names[i].loc;
		int loc_thread_id = loc->getThreadID();
		beginLoopIndex[loc_thread_id] = start;
		endLoopIndex[loc_thread_id] = end;
		dev[loc_thread_id] = loc;
		start = end;
	}
}

void
__geckoExecPolAny(const size_t initval, const size_t boundary, vector<__geckoLocationIterationType> &children_names,
				  int *beginLoopIndex, int *endLoopIndex, GeckoLocation **dev) {
	int acquiredLocation;
	geckoAcquireLocationForAny(children_names, acquiredLocation);
//		*devCount = 1;
	GeckoLocation *loc = children_names[acquiredLocation].loc;
	const int loc_thread_id = loc->getThreadID();
	dev[loc_thread_id] = loc;
	beginLoopIndex[loc_thread_id] = initval;
	endLoopIndex[loc_thread_id] = boundary;

#ifdef INFO
	fprintf(stderr, "===GECKO: Choosing location %s for 'any' execution policy.\n", dev[loc_thread_id]->getLocationName().c_str());
#endif

}

void
__geckoExecPolRange(const size_t initval, const int incremental_direction, const int ranges_count, const float *ranges,
					vector<__geckoLocationIterationType> &children_names, int *beginLoopIndex, int *endLoopIndex,
					GeckoLocation **dev) {
	geckoAcquireLocations(children_names);

	/*
	 * Mixing ranges with each other before submitting to devices.
	 */
	const int &old_range_count = ranges_count;
	const int &new_range_count = static_cast<const int &>(children_names.size());

	int *counter = (int*) malloc(sizeof(int) * new_range_count);
	for(int i=0;i<new_range_count;i++)
			counter[i] = 0;
	for(int i=0;i<old_range_count;i++)
			counter[i%new_range_count]++;

	float *new_ranges = (float*) malloc(sizeof(float) * new_range_count);
	for(int i=0;i<new_range_count;i++)
			new_ranges[i] = 0;

	int index = 0;
	for(int i=0;i<new_range_count;i++) {
		for(int j=0;j<counter[i];j++) {
			new_ranges[i] += ranges[index];
			index++;
		}
	}

	free(counter);

#ifdef INFO
	if(old_range_count > 0) {
		fprintf(stderr, "===GECKO: Old range : [%.2f", ranges[0]);
		for (int i = 1; i < old_range_count; i++)
			fprintf(stderr, ", %.2f", ranges[i]);
		fprintf(stderr, "]\n");
	} else {
		fprintf(stderr, "===GECKO: Old range : []");
	}
	if(new_range_count > 0) {
		fprintf(stderr, "===GECKO: New range : [%.2f", new_ranges[0]);
		for(int i=1;i<new_range_count;i++)
			fprintf(stderr, ", %.2f", new_ranges[i]);
		fprintf(stderr, "]\n");
	} else {
		fprintf(stderr, "===GECKO: New range : []");
	}
#endif

	int start, end, delta;
	start = initval;

	for(int dev_id=0;dev_id < new_range_count; dev_id++) {
			if(new_ranges[dev_id] == 0)
				continue;

			delta = (int) new_ranges[dev_id];
			end = (incremental_direction ? start + delta : start - delta);
#ifdef INFO
			fprintf(stderr, "\t\tChild %d: %s - share: %d - ", dev_id, children_names[dev_id].loc->getLocationName().c_str(),
			        (end - start) * (incremental_direction ? 1 : -1)  );
			fprintf(stderr, "[%d, %d] at %p\n", start, end, children_names[dev_id].loc);
#endif
			GeckoLocation *loc = children_names[dev_id].loc;
			const int tid = loc->getThreadID();
			dev[tid] = loc;
			beginLoopIndex[tid] = start;
			endLoopIndex[tid] = end;

			start = end;
		}

	free(new_ranges);
}

void __geckoExecPolPercent(const size_t initval, const size_t boundary, const int incremental_direction,
								  const int ranges_count, const float *ranges, const int totalIterations,
								  vector<__geckoLocationIterationType> &children_names, int *beginLoopIndex,
								  int *endLoopIndex, GeckoLocation **dev) {
	geckoAcquireLocations(children_names);

	/*
	 * Mixing ranges with each other before submitting to devices.
	 */
	const int &old_range_count = ranges_count;
	const int &new_range_count = static_cast<const int &>(children_names.size());

	int *counter = (int*) malloc(sizeof(int) * new_range_count);
	for(int i=0;i<new_range_count;i++)
			counter[i] = 0;
	for(int i=0;i<old_range_count;i++)
			counter[i%new_range_count]++;

	float *new_ranges = (float*) malloc(sizeof(float) * new_range_count);
	for(int i=0;i<new_range_count;i++)
			new_ranges[i] = 0;

	int index = 0;
	for(int i=0;i<new_range_count;i++) {
			for(int j=0;j<counter[i];j++) {
				new_ranges[i] += ranges[index];
				index++;
			}
		}

	free(counter);

#ifdef INFO
	if(old_range_count > 0) {
		fprintf(stderr, "===GECKO: Old range : [%.2f", ranges[0]);
		for (int i = 1; i < old_range_count; i++)
			fprintf(stderr, ", %.2f", ranges[i]);
		fprintf(stderr, "]\n");
	} else {
		fprintf(stderr, "===GECKO: Old range : []");
	}
	if(new_range_count > 0) {
		fprintf(stderr, "===GECKO: New range : [%.2f", new_ranges[0]);
		for(int i=1;i<new_range_count;i++)
			fprintf(stderr, ", %.2f", new_ranges[i]);
		fprintf(stderr, "]\n");
	} else {
		fprintf(stderr, "===GECKO: New range : []");
	}
#endif

	int start, end, delta;
	start = (int) initval;

	for(int dev_id=0;dev_id < new_range_count; dev_id++) {
			if(new_ranges[dev_id] == 0)
				continue;

			delta = (int) floor(new_ranges[dev_id] / 100.0 * totalIterations);
			end = (incremental_direction ? start + delta : start - delta);
			if(dev_id == new_range_count - 1)
				end = (int) boundary;

#ifdef INFO
			fprintf(stderr, "\t\tChild %d: %s - share: %d - ", dev_id, children_names[dev_id].loc->getLocationName().c_str(),
			        (end - start) * (incremental_direction ? 1 : -1)  );
			fprintf(stderr, "[%d, %d] at %p\n", start, end, children_names[dev_id].loc);
#endif
			GeckoLocation *loc = children_names[dev_id].loc;
			const int tid = loc->getThreadID();
			dev[tid] = loc;
			beginLoopIndex[tid] = start;
			endLoopIndex[tid] = end;

			start = end;
		}

	free(new_ranges);
}

void __geckoGetPathToRoot(string &loc, vector<GeckoLocation*> *path) {
	path->clear();
	GeckoLocation *location = GeckoLocation::find(loc);
	while(location != NULL) {
		path->push_back(location);
		location = location->getParent();
	}

#ifdef INFO
	fprintf(stderr, "===GECKO: Path from the location '%s' to the root: ", loc.c_str());
	const vector<GeckoLocation*> &__path = *path;
	for(int i=0;i<path->size();i++)
		fprintf(stderr, "%s %s", __path[i]->getLocationName().c_str(), (i == path->size()-1 ? "" : ", "));
	fprintf(stderr, "\n");
#endif
}

GeckoLocation* __geckoRegionFindByVarList(int var_list_count, void **var_list) {
	if(var_list_count == 0) {
		fprintf(stderr, "===GECKO: The size of the variable list is zero!\n");
		exit(1);
	}

	/*
	 *
	 * Avoiding those variables that their "distance" trait is set to "near" or "far".
	 * Such variables should not be taken into consideration in the location selection
	 * process since they are waiting for the process to be finalized!
	 *
	 */

	int begin_index = 0;

	for(;begin_index < var_list_count; begin_index++) {
		auto iter = geckoMemoryTable.find(var_list[begin_index]);
		if (iter == geckoMemoryTable.end()) {
			fprintf(stderr, "===GECKO: Unable to find a variable in the list (index: %d).\n", 0);
			exit(1);
		}
		const int &dist = iter->second.distance;
		if(dist != GECKO_DISTANCE_NEAR && dist != GECKO_DISTANCE_FAR)
			break;
	}
	if(begin_index == var_list_count) {
		fprintf(stderr, "===GECKO: Unable to find a location based on the list of variables.\n");
		exit(1);
	}
	string &base_loc = geckoMemoryTable[var_list[begin_index]].loc;

	GeckoLocation *base_location = GeckoLocation::find(base_loc);
	if(base_location == NULL) {
		fprintf(stderr, "===GECKO: Unable to find the location: %s\n", base_loc.c_str());
		exit(1);
	}

	vector<GeckoLocation*> *base_path, *temp_path;
	base_path = new vector<GeckoLocation*>();
	temp_path = new vector<GeckoLocation*>();
	__geckoGetPathToRoot(base_loc, base_path);
	for(int i=begin_index+1;i<var_list_count;i++) {
		auto iter = geckoMemoryTable.find(var_list[i]);
		if(iter == geckoMemoryTable.end()) {
			fprintf(stderr, "===GECKO: Unable to find a variable in the list (index: %d).\n", i);
			exit(1);
		}
		const int &dist = iter->second.distance;
		if(dist == GECKO_DISTANCE_NEAR || dist == GECKO_DISTANCE_FAR)
			continue;
		string &temp_loc = geckoMemoryTable[var_list[i]].loc;

		GeckoLocation *temp_location = GeckoLocation::find(temp_loc);
		if(temp_location == NULL) {
			fprintf(stderr, "===GECKO: Unable to find the location: %s\n", temp_loc.c_str());
			exit(1);
		}

		if(find(base_path->begin(), base_path->end(), temp_location) != base_path->end())
			continue;

		__geckoGetPathToRoot(temp_loc, temp_path);
		if(find(temp_path->begin(), temp_path->end(), base_location) == temp_path->end()) {
			fprintf(stderr, "===GECKO: Unable to find a common grandchildren among following locations: \n\t\t\t");
			for(int j=0;j<var_list_count;j++)
				fprintf(stderr, "%s %s", geckoMemoryTable[var_list[j]].loc.c_str(), (j == var_list_count-1 ? "" : ", "));
			fprintf(stderr, "\n");
			exit(1);
		}

		vector<GeckoLocation*> *t = base_path;
		base_path = temp_path;
		temp_path = t;
		base_location = temp_location;
	}


#ifdef INFO
	fprintf(stderr, "===GECKO: Chosen location based on the variable list: %s\n", base_location->getLocationName().c_str());

	fprintf(stderr, "===GECKO: Path from the chosen location '%s' to the root: {", base_location->getLocationName().c_str());
	vector<GeckoLocation*> __path = *base_path;
	for(int i=0;i<__path.size();i++)
		fprintf(stderr, "%s %s", __path[i]->getLocationName().c_str(), (i == __path.size()-1 ? "" : ", "));
	fprintf(stderr, "}\n");
#endif

	delete base_path;
	delete temp_path;

	return base_location;
}


bool __geckoCheckForGrandChildren(GeckoLocation *supposedToBeGrandParent, GeckoLocation *supposedToBeChild) {
	while(supposedToBeChild != NULL) {
		if(supposedToBeChild == supposedToBeGrandParent)
			return true;
		supposedToBeChild = supposedToBeChild->getParent();
	}
	return false;
}

void __geckoUpdateVarListWithRealAddr(int var_count, void **var_list, GeckoLocation *location) {
	for(int i=0;i<var_count;i++) {
		const auto iter = geckoMemoryTable.find(var_list[i]);
		if(iter == geckoMemoryTable.end())
			continue;

		GeckoMemory &variable = iter->second;
		const int &distance = variable.distance;

		if(distance == GECKO_DISTANCE_NEAR || distance == GECKO_DISTANCE_FAR) {

#ifdef INFO
			fprintf(stderr, "===GECKO: Checking variable at index %d as a '%s' variable\n", i, (distance == GECKO_DISTANCE_NEAR ? "Near" : "Far"));
#endif

			if(variable.real_address != NULL)
				continue;

			int traverse_distance = variable.distance_level;
			if(distance == GECKO_DISTANCE_NEAR)
				traverse_distance = 0;

			for(int j=0;j<traverse_distance && location->getParent() != NULL;j++)
				location = location->getParent();

			if(variable.allocType == GECKO_DISTANCE_ALLOC_TYPE_REALLOC) {

				variable.loc = location->getLocationName();
				variable.loc_ptr = location;

				void *temp = variable.address;

				GeckoLocationArchTypeEnum type;
				geckoMemoryAllocationAlgorithm(location, type);
				geckoAllocateMemory(type, location, &variable);

				variable.real_address = variable.address;
				variable.address = temp;

				var_list[i] = variable.real_address;    // updating the array with real addresses

#ifdef INFO
				fprintf(stderr, "===GECKO: Assigning variable at index %d to location: %s\n", i, location->getLocationName().c_str());
#endif

			} else if(variable.allocType == GECKO_DISTANCE_ALLOC_TYPE_AUTO) {
				if(variable.loc == location->getLocationName())
					continue;

				// Do not move a variable if it is already in the parent of current node!
				if(__geckoCheckForGrandChildren(variable.loc_ptr, location))
					continue;


				// TODO: this feature is not functional!
				void *temp;
				// this line does not work because of the alloc type
				geckoMemoryDeclare(&temp, variable.dataSize, variable.count, (char*) location->getLocationName().c_str(),
						variable.distance, variable.distance_level, variable.allocType, (char*)variable.filename_permanent.c_str());
				geckoMemCpy(temp, 0, variable.count, var_list[i], 0, variable.count);
				bool is_dummy = variable.is_dummy;
				geckoFree(var_list[i]);

				var_list[i] = temp;
				geckoMemoryTable[var_list[i]].is_dummy = is_dummy;
			}
		}
	}
}






GeckoError geckoRegion(char *exec_pol_chosen, char *loc_at, size_t initval, size_t boundary,
					   int incremental_direction, int has_equal_sign, int *devCount,
					   int **out_beginLoopIndex, int **out_endLoopIndex,
					   GeckoLocation ***out_dev, int ranges_count, float *ranges, int var_count, void **var_list,
					   float arithmetic_intensity) {
	geckoInit();

	string exec_pol;

	bool shouldRangesBeFreed;
	if(geckoPolicyRunTimeExists) {
		shouldRangesBeFreed = __geckoParseRangePercentagePolicy(geckoChosenPolicyRunTime, exec_pol, ranges_count, &ranges);
	} else {
		shouldRangesBeFreed = __geckoParseRangePercentagePolicy(exec_pol_chosen, exec_pol, ranges_count, &ranges);
	}

#ifdef INFO
	fprintf(stderr, "===GECKO: Execution policy (%s) at location (%s)\n", exec_pol.c_str(), loc_at);
#endif

	*devCount = 0;

	geckoBindThreadsToAccDevices(devCount);

#ifdef INFO
	fprintf(stderr, "===GECKO: Total number of threads generated: %d\n", *devCount);
#endif

	GeckoLocation *location = NULL;
	if(strcmp(loc_at, "") == 0)
		location = __geckoRegionFindByVarList(var_count, var_list);
	else
		location = GeckoLocation::find(string(loc_at));

	if(location == NULL) {
		fprintf(stderr, "===GECKO: Unable to find location '%s'.\n", loc_at);
		exit(1);
	}

#ifdef INFO
	fprintf(stderr, "===GECKO: Extract real addresses - Start\n");
#endif
	__geckoUpdateVarListWithRealAddr(var_count, var_list, location);
#ifdef INFO
	fprintf(stderr, "===GECKO: Extract real addresses - End\n");
#endif


	// finding total iteration of the loop
	int totalIterations = static_cast<int>(boundary - initval + has_equal_sign * (incremental_direction ? 1 : -1));
#ifdef INFO
	fprintf(stderr, "===GECKO: TotalIterations: %d\n", totalIterations);
#endif
	if(totalIterations == 0)
		return GECKO_ERR_TOTAL_ITERATIONS_ZERO;


	vector<__geckoLocationIterationType> children_names;
	geckoExtractChildrenFromLocation(location, children_names, (totalIterations >= 0 ? totalIterations : -1*totalIterations));
//	*devCount = children_names.size();


	int loop_index_count = *devCount;
//	if(strcmp(exec_pol, "range") == 0 || strcmp(exec_pol, "percentage") == 0)
//		loop_index_count = ranges_count;

	int *beginLoopIndex = (int*) malloc(sizeof(int) * loop_index_count);
	int *endLoopIndex = (int*) malloc(sizeof(int) * loop_index_count);
	GeckoLocation **dev = (GeckoLocation**) malloc(sizeof(GeckoLocation*) * loop_index_count);
	for(int i=0;i<loop_index_count;i++) {
		dev[i] = NULL;
		beginLoopIndex[i] = 0;
		endLoopIndex[i] = 0;
	}


#ifdef INFO
	fprintf(stderr, "===GECKO: Number of locations for distribution: %d\n", children_names.size());
#endif


	if(exec_pol == "static") {
		__geckoExecPolStatic(initval, boundary, incremental_direction, children_names, beginLoopIndex, endLoopIndex,
				dev);
	} else if(exec_pol == "flatten") {
		__geckoExecPolFlatten(initval, boundary, incremental_direction, devCount, totalIterations, children_names,
							  beginLoopIndex, endLoopIndex,
							  dev);
	} else if(exec_pol == "any") {
		__geckoExecPolAny(initval, boundary, children_names, beginLoopIndex, endLoopIndex, dev);
	} else if(exec_pol == "range") {
		__geckoExecPolRange(initval, incremental_direction, ranges_count, ranges, children_names, beginLoopIndex,
							endLoopIndex, dev);
	} else if(exec_pol == "percentage") {
		__geckoExecPolPercent(initval, boundary, incremental_direction, ranges_count, ranges, totalIterations,
							  children_names, beginLoopIndex,
							  endLoopIndex, dev);
	} else {
		fprintf(stderr, "===GECKO: Unknown chosen execution policy: '%s'.", exec_pol.c_str());
		exit(1);
	}


	/*
	 * Allocating distance-based variables
	 */
//	for(int i=0;i<var_count;i++) {
//		auto iter = geckoMemoryTable.find(var_list[i]);
//		if(iter == geckoMemoryTable.end())
//			continue;
//		GeckoMemory &variable = iter->second;
//		if(!variable.allocated)
//			continue;
//		GeckoLocation *pLocation;
//		pLocation = GeckoLocation::find(variable.loc);
//		if(variable.distance == GECKO_DISTANCE_NEAR) {
//			pLocation = location;
//			variable.loc = string(loc_at);
//		} else if(variable.distance == GECKO_DISTANCE_FAR) {
//			pLocation = GeckoLocation::findRoot();
//			variable.loc = pLocation->getLocationName();
//		}
//		GeckoLocationArchTypeEnum type;
//		geckoMemoryAllocationAlgorithm(pLocation, type);
//		geckoAllocateMemory(type, &variable);
//	}


//#ifdef INFO
//	fprintf(stderr, "===GECKO: Advising memory allocation at location %s.\n", loc_at == NULL ? "" : loc_at);
//#endif
//	for(int i=0;i<var_count;i++)
//		geckoMemoryDistribution(*devCount, dev, var_count, var_list, beginLoopIndex, endLoopIndex);

	if(shouldRangesBeFreed) {
		free(ranges);
		ranges = NULL;
	}


	*out_dev = dev;
	*out_beginLoopIndex = beginLoopIndex;
	*out_endLoopIndex = endLoopIndex;

	return GECKO_SUCCESS;
}







GeckoError geckoUnsetBusy(GeckoLocation *device) {
	omp_set_lock(&lock_freeResources);
	freeResources.insert(device);
	omp_unset_lock(&lock_freeResources);
	return GECKO_SUCCESS;
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
	if(devCount == 0)
		return GECKO_SUCCESS;

	GeckoLocation **locs = (GeckoLocation**) malloc(sizeof(GeckoLocation*) * devCount);
	for(int i=0;i<devCount;i++)
		locs[i] = children_names[i].loc;

#ifdef INFO
	fprintf(stderr, "===GECKO: Begin to wait on %s - Number of children to wait on: %d\n", loc_at, devCount);
#endif
#pragma omp parallel num_threads(devCount)
//	for(int devIndex=0;devIndex<devCount;devIndex++)
	{
		int tid = omp_get_thread_num();
		//GeckoLocation *loc = GeckoLocation::find(geckoThreadDeviceMap[tid]);
		GeckoLocation *loc = locs[tid];
#ifdef INFO
		if(loc == NULL) fprintf(stderr, "===GECKO: \tUnable to find the location associated to thread ID: %d\n", tid);
#endif
		if(loc != NULL) {
//			geckoSetDevice(loc);
			int async_id = loc->getAsyncID();
#ifdef INFO
			fprintf(stderr, "===GECKO: \tWaiting on location %s with asyncID %d on thread %d.\n", loc->getLocationName().c_str(), async_id, tid); fflush(stderr);
#endif
#pragma acc wait(async_id)
#ifdef INFO
			fprintf(stderr, "===GECKO: \tWaiting on location %s with asyncID %d on thread %d - Done.\n", loc->getLocationName().c_str(), async_id, tid); fflush(stderr);
#endif
			geckoUnsetBusy(loc);
		}
	}
#ifdef INFO
	fprintf(stderr, "===GECKO: End of wait on %s - Number of children to wait on: %d\n", loc_at, devCount);
#endif

	free(locs);

	return GECKO_SUCCESS;
}


void geckoFreeRegionTemp(int *beginLoopIndex, int *endLoopIndex, int devCount, GeckoLocation **dev,
		int var_count, void **var_list, void **out_var_list) {

	geckoFreeDistanceRealloc(var_count, out_var_list);

	if(beginLoopIndex)
		free(beginLoopIndex);
	if(endLoopIndex)
		free(endLoopIndex);
	if(dev)
		free(dev);
	if(var_list)
		free(var_list);
	if(out_var_list)
		free(out_var_list);
}

