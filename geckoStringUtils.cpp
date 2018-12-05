//
// Created by millad on 11/28/18.
//

#include "geckoStringUtils.h"

void __geckoGetFields(char *line, vector<string> &v, char *delim) {
	v.clear();
	char* tmp = strdup(line);
	const char* tok;
	for (tok = strtok(line, delim);
		 tok && *tok;
		 tok = strtok(NULL, delim)) {
		string string_tok = string(tok);
		trim(string_tok);
		if(string_tok.size() == 0 || string_tok.compare(string("")) == 0)
			continue;
		v.push_back(string_tok);
	}
	free(tmp);
}

