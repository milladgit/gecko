//
// Created by millad on 11/28/18.
//

#include "geckoDraw.h"
#include "geckoHierarchicalTree.h"

#include <stdlib.h>
#include <cstring>
#include <cstdio>
#include <unordered_set>

__attribute__((always_inline))
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

__attribute__((always_inline))
void __geckoDrawWithAllNodes(FILE *f, unordered_map<string, GeckoLocation*> &locMap) {
	char line[4096];
	unordered_set<string> alreadyBoxDrawn;
	for(auto iter=locMap.begin(); iter != locMap.end(); iter++) {
		string currentLocName = iter->first;
		if(iter->second == NULL) {
//			fprintf(stderr, "==============ERROR IN SECOND: %s\n", currentLocName.c_str());
			continue;
		}

		if(iter->second->getParent() == NULL) {
//			fprintf(stderr, "==============ERROR IN PARENT: %s\n", currentLocName.c_str());
			continue;
		}
		string parentName = iter->second->getParent()->getLocationName();


		if(alreadyBoxDrawn.find(currentLocName) == alreadyBoxDrawn.end()) {
			sprintf(&line[0], "\"%s\"  [shape=box];\n", currentLocName.c_str());
			fwrite(&line[0], sizeof(char), strlen(&line[0]), f);
		}

		if(alreadyBoxDrawn.find(parentName) == alreadyBoxDrawn.end()) {
			sprintf(&line[0], "\"%s\"  [shape=box];\n", parentName.c_str());
			fwrite(&line[0], sizeof(char), strlen(&line[0]), f);
		}


		sprintf(&line[0], "\"%s\" -> \"%s\";\n", parentName.c_str(), currentLocName.c_str());
		fwrite(&line[0], sizeof(char), strlen(&line[0]), f);

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
//	GeckoLocation *root = GeckoLocation::find(string(rootNode));
//	__geckoDrawPerNode(f, root);
	unordered_map<string, GeckoLocation*> locMap = GeckoLocation::getAllLocations();
	__geckoDrawWithAllNodes(f, locMap);
	sprintf(&line[0], "}\n");
	fwrite(&line[0], sizeof(char), strlen(&line[0]), f);
	fclose(f);
}

