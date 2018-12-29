
#include "geckoHierarchicalTree.h"
#include <algorithm>

unordered_map<string, GeckoLocation*> GeckoLocation::geckoListOfAllNodes;
vector<GeckoLocation*> GeckoLocation::childrenInCategories[GECKO_DEVICE_LEN];
bool GeckoLocation::treeHasBeenModified;
vector<GeckoLocation*> GeckoLocation::finalChildListForThreads;

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
//#ifdef INFO
//	fprintf(stderr, "===GECKO: Destructor on '%s' is called.\n", this->locationName.c_str());
//#endif
//	auto iter = geckoListOfAllNodes.find(this->locationName);
//	if(iter != geckoListOfAllNodes.end())
//		geckoListOfAllNodes.erase(iter);
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
	
	int type = (int) location->getLocationType().type;
	vector<GeckoLocation *> &childCategory = childrenInCategories[type];
	auto iter2 = std::find(childCategory.begin(), childCategory.end(), location);
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

#ifdef INFO
	fprintf(stderr, "===GECKO: Found %d leaves\n", *numDevices);
#endif

	if(*numDevices == 0)
		return true;

#ifdef INFO
	int child_index = 0;
#endif
	for(int devTypeIndex=0;devTypeIndex<listOfTypesCount; devTypeIndex++) {
		vector<GeckoLocation *> &childCategory = childrenInCategories[listOfTypes[devTypeIndex]];
		int sz = childCategory.size();
		for(int i=0;i<sz;i++) {
			finalChildListForThreads.push_back(childCategory[i]);
#ifdef INFO
			fprintf(stderr, "===GECKO: \tLeaf %d is %s\n", child_index++, childCategory[i]->getLocationName().c_str());
#endif
		}
	}

	return true;
}

vector<GeckoLocation*> &GeckoLocation::getChildListForThreads() {
	return finalChildListForThreads;
}

unordered_map<string, GeckoLocation*> GeckoLocation::getAllLocations() {
	return geckoListOfAllNodes;
}

void GeckoLocation::dumpTable() {
	auto iter = geckoListOfAllNodes.begin();
	fprintf(stderr, "===GECKO: Dump location table:\n");
	for(;iter != geckoListOfAllNodes.end(); iter++) {
		fprintf(stderr, "===\t\t%s at %p\n", iter->first.c_str(), iter->second);
	}
	fprintf(stderr, "===GECKO: Dump location table ... Done\n");
}
