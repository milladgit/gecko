//
// Created by millad on 11/28/18.
//

#ifndef GECKO_GECKOSTRINGUTILS_H
#define GECKO_GECKOSTRINGUTILS_H


#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>

using namespace std;


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

void __geckoGetFields(char *line, vector<string> &v, char *delim);


#endif //GECKO_GECKOSTRINGUTILS_H
