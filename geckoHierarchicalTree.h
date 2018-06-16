
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

public:
	GeckoLocation(string locationName, GeckoLocation *parent, GeckoLocationType locationObj, int locIndex, int async_id);
	~GeckoLocation();

	static GeckoLocation *find(string name);

	void appendChild(GeckoLocation *node);
	void removeChild(GeckoLocation *node);

	string getLocationName();
	GeckoLocationType getLocationType();
	GeckoLocation *getParent();
	void setParent(GeckoLocation *p);
	vector<GeckoLocation*> getChildren();
	int getLocationIndex();
	int getAsyncID();
	void setAsyncID(int id);

	static GeckoLocation *findRoot();

};



#endif
