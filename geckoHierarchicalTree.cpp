
#include "geckoHierarchicalTree.h"
#include <algorithm>

unordered_map<string, GeckoLocation*> GeckoLocation::geckoListOfAllNodes;

GeckoLocation::GeckoLocation(string locationName, GeckoLocation *parent, GeckoLocationType locationObj,
							 int locIndex, int async_id) {
	this->locationName = locationName;
	this->locationObj = locationObj;
	this->parent = parent;
	this->locationIndex = locIndex;
	this->async_id = async_id;

	geckoListOfAllNodes[locationName] = this;
}

GeckoLocation::~GeckoLocation() {
	auto iter = geckoListOfAllNodes.find(this->locationName);
	if(iter != geckoListOfAllNodes.end()) {
		geckoListOfAllNodes.erase(iter);
	}
}

void GeckoLocation::appendChild(GeckoLocation *location) {
	if(std::find(children.begin(), children.end(), location) != children.end())
		return;

	treeHasBeenModified = true;
	children.push_back(location);
	GeckoLocationArchTypeEnum type = location->getLocationType().type;
	childrenInCategories[type].push_back(location);
}

void GeckoLocation::removeChild(GeckoLocation *location) {
	auto iter = std::find(children.begin(), children.end(), location);
	if(iter == children.end())
		return;

	treeHasBeenModified = true;

	children.erase(iter);
	
	GeckoLocationArchTypeEnum type = location->getLocationType().type;
	vector<GeckoLocation *> &childCategory = childrenInCategories[type];
	auto iter2 = std::find(childCategory.begin(), childCategory.end(), type);
	if(iter2 != childCategory.end())
		childCategory.erase(iter2);
}


// static
GeckoLocation *GeckoLocation::find(string name) {
	auto iter = geckoListOfAllNodes.find(name);
	if(iter != geckoListOfAllNodes.end())
		return iter->second;

	return NULL;
}

string GeckoLocation::getLocationName() {
	return this->locationName;
}
GeckoLocationType GeckoLocation::getLocationType() {
	return this->locationObj;
}

GeckoLocation *GeckoLocation::getParent() {
	return parent;
}

void GeckoLocation::setParent(GeckoLocation *p) {
	parent = p;
}

vector<GeckoLocation*> GeckoLocation::getChildren() {
	return children;
}

int GeckoLocation::getLocationIndex() {
	return locationIndex;
}

GeckoLocation *GeckoLocation::findRoot() {
	if(geckoListOfAllNodes.size() == 0)
		return NULL;
	unordered_map<string, GeckoLocation*>::iterator iter = geckoListOfAllNodes.begin();
	GeckoLocation *node = iter->second;
	GeckoLocation *prev;
	while(node) {
		prev = node;
		node = node->getParent();
	}
	return prev;
}

int GeckoLocation::getAsyncID() {
	return async_id;
}

void GeckoLocation::setAsyncID(int id) {
	async_id = id;
}

int GeckoLocation::getThreadID() {
	return thread_id;
}

void GeckoLocation::setThreadID(int id) {
	thread_id = id;
}

bool GeckoLocation::getAllLeavesOnce(int *numDevices) {
	if(!treeHasBeenModified) {
		*numDevices = finalChildListForThreads.size();
		return false;
	}

	treeHasBeenModified = false;
	finalChildListForThreads.clear();

	const int listOfTypesCount = 3;
	GeckoLocationArchTypeEnum listOfTypes[listOfTypesCount] = {GECKO_X32, GECKO_X64, GECKO_NVIDIA};

	*numDevices = 0;
	for(int devTypeIndex=0;devTypeIndex<listOfTypesCount; devTypeIndex++) {
		*numDevices += childrenInCategories[listOfTypes[devTypeIndex]].size();
	}

	if(*numDevices == 0)
		return true;

	for(int devTypeIndex=0;devTypeIndex<listOfTypesCount; devTypeIndex++) {
		const vector<GeckoLocation *> &childCategory = childrenInCategories[devTypeIndex];
		int sz = childCategory.size();
		for(int i=0;i<sz;i++)
			finalChildListForThreads.push_back(childCategory[i]);
	}

	return true;
}

static vector<GeckoLocation*> &GeckoLocation::getChildListForThreads() {
	return finalChildListForThreads;
}
