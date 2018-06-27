
import os, sys, re

def parseForLoop(line):
	l = line.strip()
	pattern_for = r'for\s*'
	# pattern_data_type   = r'((?P<datatype>[\w\s_]+))*\s*'
	# pattern_data_type   = r'((?P<datatype>\w+))*'
	# pattern_data_type_2 = r'((?P<datatype2>\w+))*'
	pattern_data_type_var_name = r'((?P<datatype_varname>[a-zA-Z0-9_ ]+)\s*)*'
	#pattern_var_name = r'(?P<varname>\w+)\s*'
	pattern_initval = r'\s*(?P<initval>[0-9a-zA-Z_+=\-*/]+)\s*'
	pattern_var_cond = r'\s*(?P<varcond>[_.\w]+)\s*'
	pattern_cond = r'(?P<cond>[>=<]+)\s*'
	pattern_boundary = r'(?P<boundary>[\w+_.\-*/]+)\s*'
	pattern_increment = r'\s*(?P<increment1>[0-9a-zA-Z_+=\-*/]*)\s*(?P<increment2>[0-9a-zA-Z+=\-*/]*)\s*(?P<increment3>[0-9a-zA-Z+=\-*/]*)\s*'
	pattern = r'%s\((%s=%s)*;%s%s%s;%s\)\s*(?P<statements>[{])*' % (pattern_for, pattern_data_type_var_name, pattern_initval, pattern_var_cond, pattern_cond, pattern_boundary, pattern_increment)
	match = re.search(pattern, l)
	if match == None:
		return None

	# datatype  = match.group('datatype')
	# varname = match.group('varname')

	datatype_varname  = match.group('datatype_varname')
	datatype_varname = datatype_varname.strip().split()
	datatype = ""
	varname = ""
	if len(datatype_varname) > 0:
		varname = datatype_varname[-1]
		if len(datatype_varname) > 1:
			datatype = ' '.join(datatype_varname[:-1])

	initval = match.group('initval')
	varcond = match.group('varcond')
	cond = match.group('cond')
	boundary = match.group('boundary')
	inc1 = match.group("increment1")
	inc2 = match.group("increment2")
	inc3 = match.group("increment3")
	statements = match.group("statements")

	inc = ""
	if inc1 is not None:
		inc = inc1
		if inc2 is not None:
			inc += inc2
			if inc3 is not None:
				inc += inc3

	return (datatype, varname, initval, varcond, cond, boundary, inc, statements)


def parsePragmaACC(line):
	l = line.strip()
	pattern_pragma = r'#pragma\s+acc\s+(?P<pargma_content>.*)'
	match = re.search(pattern_pragma, l)
	if match is None:
		return None
	return match.group('pargma_content')


def parseRangePolicy(policy_type, policy):
	pattern = r'%s:\[(?P<ranges>[\w\s,]+)\]' % (policy_type)
	match = re.search(pattern, policy)
	if match is None:
		return None
	return match.group('ranges')
