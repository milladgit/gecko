//
// Created by millad on 11/28/18.
//

#include "geckoConfigLoader.h"
#include "geckoStringUtils.h"

#include <vector>
#include <string>

using namespace std;



extern GeckoError 	geckoLocationtypeDeclare(char *name, GeckoLocationArchTypeEnum deviceType, const char *microArch,
					int numCores, const char *mem_size, const char *mem_type, float bandwidth_GBps);
extern GeckoError 	geckoLocationDeclare(const char *name, const char *_type, int all, int start, int count);
extern GeckoError 	geckoHierarchyDeclare(char operation, const char *child_name, const char *parent, int all, int start,
					int count);



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
	string name, kind, num_cores, mem_size, bw_str, filename;
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
		else if(values[0].compare("bw") == 0 || values[0].compare("bandwidth") == 0 )
			bw_str = values[1];
	}

	trim(kind);
	toUpper(kind);
	trim(num_cores);

	float bw = strtof(bw_str.c_str(), NULL);

	if(bw == 0.0f)
		bw = -1;

	if(name.compare("") == 0 || kind.compare("") == 0 || bw == 0.0f) {
		fprintf(stderr, "===GECKO: Error in declaring location type within the config file: name(%s) - "
						"kind(%s) - num_cores(%s) - mem(%s)\n", name.c_str(), kind.c_str(), num_cores.c_str(), mem_size.c_str());
		exit(1);
	}

	GeckoLocationArchTypeEnum deviceType;
	if(kind.compare("X32") == 0)
		deviceType = GECKO_X32;
	else if(kind.compare("X64") == 0)
		deviceType = GECKO_X64;
	else if(kind.compare("NVIDIA") == 0)
		deviceType = GECKO_NVIDIA;
	else if(kind.compare("UNIFIED_MEMORY") == 0)
		deviceType = GECKO_UNIFIED_MEMORY;
	else if(kind.compare("PERMANENT_STORAGE") == 0)
		deviceType = GECKO_PERMANENT_STORAGE;
	else
		deviceType = GECKO_UNKOWN;

	int num_cores_int = 0;
	if(num_cores.compare("") != 0)
		num_cores_int = stoi(num_cores, NULL, 10);

	geckoLocationtypeDeclare((char*)name.c_str(), deviceType, "", num_cores_int, mem_size.c_str(), "", bw);

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
		else if(values[0].compare("total_count") == 0)
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
		else if(values[0].compare("total_count") == 0)
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

		if(fields[0][0] == '#')
			continue;

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

GeckoError  geckoLoadConfigWithEnv() {
	const char *env_name = "GECKO_CONFIG_FILE";
	char *filename = getenv(env_name);
	if(filename == NULL) {
#if defined(WARNING) || defined(INFO)
		fprintf(stderr, "===GECKO: Unable to find the environment variable (%s). \n", env_name);
#endif
		return GECKO_ERR_FAILED;
	}
	geckoLoadConfigWithFile(filename);
	return GECKO_SUCCESS;
}

