//
// Created by millad on 11/29/18.
//

#include "geckoHierarchy.h"
#include "geckoHierarchicalTree.h"
#include <string.h>

extern GeckoError geckoInit();
extern GeckoCUDAProp geckoCUDA;


inline
void __geckoHierDeclAll(const char *child_name, int all, int &begin, int &end) {
	if(all) {
		char name[128];
		sprintf(&name[0], "%s[%d]", child_name, 0);
		GeckoLocation *child = GeckoLocation::find(string(name));
		if(child == NULL) {
			fprintf(stderr, "===GECKO: Unable to find (%s)!\n", child_name);
			exit(1);
		}

		GeckoLocationType locObj = child->getLocationType();

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
		if(locObj.type == GECKO_NVIDIA) {
			fprintf(stderr, "===GECKO: No CUDA is available on this system.\n");
			exit(1);
		}
#endif

	}

}

GeckoLocation * __geckoDetermineParent(char operation, const char *parent_name) {
	GeckoLocation *parentNode = NULL;
	if(operation == '+') {
		parentNode = GeckoLocation::find(string(parent_name));
		if (parentNode == NULL) {
			fprintf(stderr, "===GECKO: Unable to find parent (%s)!\n", parent_name);
			exit(1);
		}
	}
	return parentNode;
}

GeckoError geckoHierarchyDeclare(char operation, const char *child_name, const char *parent_name, int all, int start,
								int count) {

	geckoInit();

	if(operation != '+' && operation != '-') {
		fprintf(stderr, "===GECKO: Unrecognizeable operation ('%c') in hierarchy declaration for location '%s'.\n",
				operation, child_name);
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


	__geckoHierDeclAll(child_name, all, begin, end);

	GeckoLocation *parentNode = __geckoDetermineParent(operation, parent_name);

#ifdef INFO
	char operation_name[16];
	strcpy(&operation_name[0], operation == '+' ? "Declaring" : "Removing");
#endif


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

		childNode->setParent(parentNode);
		if(operation == '+') {
			parentNode->appendChild(childNode);
		} else {
			parentNode->removeChild(childNode);
		}

#ifdef INFO
		fprintf(stderr, "===GECKO: %s '%s' as child of '%s'.\n", &operation_name[0], &name[0], parent_name);
#endif

	}

	return GECKO_SUCCESS;
}

