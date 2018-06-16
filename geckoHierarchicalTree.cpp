
#include "geckoHierarchicalTree.h"
#include <algorithm>

unordered_map<string, GeckoLocation*> GeckoLocation::geckoListOfAllNodes;

GeckoLocation::GeckoLocation(string locationName, GeckoLocation *parent, GeckoLocationType locationObj, int locIndex, int async_id) {
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

void GeckoLocation::appendChild(GeckoLocation *node) {
	if(std::find(children.begin(), children.end(), node) != children.end())
		return;
	children.push_back(node);
}

void GeckoLocation::removeChild(GeckoLocation *node) {
	auto iter = std::find(children.begin(), children.end(), node);
	if(iter == children.end())
		return;

	children.erase(iter);
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
