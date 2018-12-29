
#pragma once

#ifndef __GECKO_HIERARCHICAL_TREE_H__
#define __GECKO_HIERARCHICAL_TREE_H__

#include <vector>
#include <unordered_map>
#include <string>

#include "geckoDataTypes.h"

using namespace std;


class GeckoLocation {

	static unordered_map<string, GeckoLocation*> geckoListOfAllNodes;

	string			locationName;
	GeckoLocationType 	locationObj;
	int				locationIndex;

	int             async_id;

	vector<GeckoLocation*> children;
	GeckoLocation *parent;

	int thread_id;

	static vector<GeckoLocation*> childrenInCategories[GECKO_DEVICE_LEN];
	static bool treeHasBeenModified;
	static vector<GeckoLocation*> finalChildListForThreads;

public:
	GeckoLocation(string locationName, GeckoLocation *parent, GeckoLocationType locationObj, int locIndex,
				  int async_id);
	~GeckoLocation();

	static GeckoLocation *find(string name);

	void appendChild(GeckoLocation *location);
	void removeChild(GeckoLocation *location);

	string getLocationName();
	GeckoLocationType getLocationType();
	GeckoLocation *getParent();
	void setParent(GeckoLocation *p);
	vector<GeckoLocation*> getChildren();
	int getLocationIndex();
	int getAsyncID();
	void setAsyncID(int id);

	int getThreadID();
	void setThreadID(int id);

	static GeckoLocation *findRoot();

	// This function does the following:
	// Has hierarchy changed since the last call to this function?
	//
	// So, if something has changed, do something about it:
	// If yes, then some code needs to reassign the devices to the
	// threads.
	// Otherwise, leave the thread assignment intact.
	static bool getAllLeavesOnce(int *numDevices);

	static vector<GeckoLocation*> &getChildListForThreads();

	static unordered_map<string, GeckoLocation*> getAllLocations();

	static void dumpTable();

};



#endif
