
import os, sys, re

def parseForLoop(line):
	l = line.strip()
	pattern_for = r'for\s*'
	pattern_data_type = r'((?P<datatype>\w+))*\s*'
	pattern_var_name = r'(?P<varname>\w+)\s*'
	pattern_initval = r'\s*(?P<initval>\w+)\s*'
	pattern_var_cond = r'\s*(?P<varcond>\w+)\s*'
	pattern_cond = r'(?P<cond>[>=<]+)\s*'
	pattern_boundary = r'(?P<boundary>\w+)\s*'
	pattern_increment = r'\s*(?P<increment1>[0-9a-zA-Z+=\-*/]*)\s*(?P<increment2>[0-9a-zA-Z+=\-*/]*)\s*(?P<increment3>[0-9a-zA-Z+=\-*/]*)\s*'
	pattern = r'%s\((%s%s=%s)*;%s%s%s;%s\)\s*(?P<paranthesis>[{])*' % (pattern_for, pattern_data_type, pattern_var_name, pattern_initval, pattern_var_cond, pattern_cond, pattern_boundary, pattern_increment)
	match = re.search(pattern, l)
	if match == None:
		return None

	datatype = match.group('datatype')
	varname = match.group('varname')
	initval = match.group('initval')
	varcond = match.group('varcond')
	cond = match.group('cond')
	boundary = match.group('boundary')
	inc1 = match.group("increment1")
	inc2 = match.group("increment2")
	inc3 = match.group("increment3")
	paranthesis = match.group("paranthesis")

	inc = ""
	if inc1 is not None:
		inc = inc1
		if inc2 is not None:
			inc += inc2
			if inc3 is not None:
				inc += inc3

	return (datatype, varname, initval, varcond, cond, boundary, inc, paranthesis)


def parsePragmaACC(line):
	l = line.strip()
	pattern_pragma = r'#pragma\s+acc\s+(?P<pargma_content>.*)'
	match = re.search(pattern_pragma, l)
	if match is None:
		return None
	return match.group('pargma_content')

