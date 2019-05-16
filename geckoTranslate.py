
import os,sys,re,glob,atexit
import geckoREUtilities as gREU
import geckoFileUtils as gFU
import copy


pragma_keyword 				= "gecko"
pragma_prefix  				= "GECKO"
pragma_prefix_funcname 		= "gecko"


preserveOriginalCodeAsComment = False
autoGenerateACCPragmas = True


def isFloat(stringInput):
	try:
		f = float(stringInput)
		return True, f
	except ValueError:
		return False, 0




class SourceFile(object):
	"""docstring for SourceFile"""
	def __init__(self, filename):
		super(SourceFile, self).__init__()
		self.filename = filename
		self.line_for_end = ""

		self.parsing_region_state = 0
		self.pragmaForRegion = ""

		self.var_list = ""
		self.variable_list_internal = ""

		self.exec_pol = ""
		self.exec_pol_type = ""
		self.exec_pol_option = ""
		self.at = '\"\"'

		self.gang = False
		self.gang_count = -1
		self.vector = False
		self.vector_count = -1
		self.kernels = False
		self.reduction_list = list()

		self.intensity = ""


	def processLocationType(self, keywords, lineNumber):
		name = "-"
		family = "X64"
		micro_arch = '"-"'
		num_cores = "0"
		mem_size = '"-"'
		mem_type = '"-"'
		bandwidth = ""
		i = 3
		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "name":
				name = k[1][:-1]
			elif k[0] == "kind":
				kind_text = k[1][:-1]
				kind = kind_text.split(",")
				family = kind[0].upper()
				family = family[1:-1]
				if len(kind) == 2:
					micro_arch = kind[1].upper()
					# micro_arch = micro_arch[1:-1]

				if family not in ("X64", "X32", "NVIDIA", "UNIFIED_MEMORY", "PERMANENT_STORAGE"):
					print "Line %d - Error in kind of locationtype - Unknown family (%s)" % (lineNumber, family)
					exit(1)

			elif k[0] == "num_cores":
				num_cores = k[1][:-1]
			elif k[0] == "mem" or k[0] == "size":
				mem_text = k[1][:-1]
				mem = mem_text.split(',')
				mem_len = len(mem)
				if mem_len == 1:
					mem_size = mem[0].strip()
				elif mem_len == 2:
					mem_size = mem[0].strip()
					mem_type = mem[1].strip()
				else:
					print "Line %d - Error in kind of locationtype - Unknown mem clause format (%s)" % (lineNumber, mem_text)
					exit(1)
			elif k[0] == "bandwidth":		# "bandwidth" and "bw" are possible and mean the same
				bandwidth = k[1][:-1]
			elif k[0] == "bw":
				bandwidth = k[1][:-1]

			i += 1


		bandwidth_is_float, bandwidth_val_float = isFloat(bandwidth)
		if bandwidth_is_float:
			bandwidth = bandwidth_val_float
		else:
			if bandwidth == '':
				bandwidth = -1


		line = '%sLocationtypeDeclare(%s, %s_%s, %s, %s, %s, %s, %s);\n' % (pragma_prefix_funcname, name, pragma_prefix, family, micro_arch, num_cores, mem_size, mem_type, bandwidth)
		return line


	def processLocation(self, keywords, lineNumber):
		name = '"-"'
		_type = ""
		_all = 0
		i = 3
		name_list = []

		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "name":
				name = k[1][:-1].strip()
				# if there is a comma in the name, so we have a list and not a range name
				if "," in name:
					name_list = name.split(",")
					name_list = [(_n.strip(), -1, -1) for _n in name_list]
				else:
					name_range = name.split("[")
					name = name_range[0]
					start = end = -1
					if len(name_range) > 1:
						name = name[1:]
						_range = name_range[1]
						_range = _range[:-2]
						start_end = _range.split(":")
						start = start_end[0]
						end   = start_end[1]
					name_list.append((name, start, end))

			elif k[0] == "type":
				_type = k[1][:-1]
			elif k[0] == "all":
				_all = 1
			i += 1

		line = ""
		for n in name_list:
			name = n[0]
			start = n[1]
			end = n[2]
			if start != -1:
				name = '"%s"' % (name)
			line += '%sLocationDeclare(%s, %s, %d, %s, %s);\n' % (pragma_prefix_funcname, name, _type, _all, start, end)
		return line


	def processHierarchy(self, keywords, lineNumber):
		parent = '""'
		cmd = '+'
		children = '""'
		_all = 0
		i = 3
		name_list = []

		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "children":
				cmd_children = k[1][:-1]
				cmd_children = cmd_children.split(":")					
				cmd = cmd_children[0]
				if len(cmd) == 1 and cmd not in ['+', '-']:
					print "Line %d - Unknown keyword in children clause (%s)." % (lineNumber, cmd)
					exit(1)
				if len(cmd) > 1 and ("'" in cmd or '"' in cmd):
					print "Line %d - Unknown keyword in children clause (%s)." % (lineNumber, cmd)
					exit(1)
				if len(cmd) == 1:
					cmd = "'%s'" % (cmd)

				if len(cmd_children) > 2:
					children = ':'.join(cmd_children[1:])
				else:
					children = cmd_children[1]


				if "," in children:					
					name_list = children.split(",")
					name_list = [(_n.strip(), -1, -1) for _n in name_list]
				else:
					name_range = children.split("[")
					name = name_range[0]
					start = end = -1
					if len(name_range) > 1:
						name = name[1:]
						_range = name_range[1]
						_range = _range[:-2]
						start_end = _range.split(":")
						start = start_end[0]
						end   = start_end[1]

					name_list.append((name, start, end))

			elif k[0] == "parent":
				parent = k[1][:-1]
				if len(parent.split(",")) > 1:
					print "Line %d - Error in parent name - Only one parent should be specified (%s)" % (lineNumber, parent)
					exit(1)

			elif k[0] == "all":
				_all = 1

			i += 1

		line = ""
		for n in name_list:
			name = n[0]
			start = n[1]
			end = n[2]
			if start != -1:
				name = '"%s"' % (name)
			line += '%sHierarchyDeclare(%s, %s, %s, %d, %s, %s);\n' % (pragma_prefix_funcname, cmd, name, parent, _all, start, end)
		return line


	def processMemory(self, keywords, lineNumber):
		name = "-"
		_type = ""
		loc = '""'
		tile = ""
		varToFree = ""
		varToFreeObj = ""
		distance = ""
		distance_level = -1					# one-based index (starts from 1)
		distance_alloc_type = ""			# auto or realloc?
		copy_op = False
		move_var = ""
		register_var = ""
		unregister_var = ""
		loc_kw = ""
		from_var = ""
		to_var = ""
		is_auto = False 
		is_realloc = False
		filename_permanent = None
		i = 3
		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "allocate":
				name = k[1][:-1]
			elif k[0] == "type":
				_type = k[1][:-1]
			elif k[0] == "location":
				loc = k[1][:-1]
			elif k[0] == "distance":
				distance = k[1][:-1]
			elif k[0] == "free":
				varToFree = k[1][:-1]
			elif k[0] == "freeobj":
				varToFreeObj = k[1][:-1]
			elif k[0] == "register":
				register_var = k[1][:-1]
			elif k[0] == "unregister":
				unregister_var = k[1][:-1]
			elif k[0] == "loc":
				loc_kw = k[1][:-1]
			elif k[0] == "copy":
				copy_op = True
			elif k[0] == "move":
				move_var = k[1][:-1]
			elif k[0] == "from":
				from_var = k[1][:-1]
			elif k[0] == "to":
				to_var = k[1][:-1]
			elif k[0] == "auto":
				is_auto = True
			elif k[0] == "realloc":
				is_realloc = True
			elif k[0] == "file":
				filename_permanent = k[1][:-1]

			i += 1


		# Taking care of freeing a variable
		if varToFree != "":
			varList = varToFree.split(",")
			line = ""
			for v in varList:
				if len(v)>0:
					line += "%sFree(%s);\n" % (pragma_prefix_funcname, v)
			return line
		elif varToFreeObj != "":
			varList = varToFreeObj.split(",")
			line = ""
			for v in varList:
				if len(v)>0:
					line += "%s.freeMem();\n" % (v)
			return line


		# Taking care of copying a variable
		if copy_op:
			from_var_result = gREU.parseVariableString(from_var)
			if from_var_result is None:
				print "Line: %d - Unable to extract variable name and its corresponding range for '%s' keyword." % (lineNumber, 'from')
				exit(1)
			to_var_result = gREU.parseVariableString(to_var)
			if to_var_result is None:
				print "Line: %d - Unable to extract variable name and its corresponding range for '%s' keyword." % (lineNumber, 'to')
				exit(1)
			line = '%sMemCpy(%s, %s, %s, %s, %s, %s);\n' % (pragma_prefix_funcname, to_var_result[0], to_var_result[1], to_var_result[2], from_var_result[0], from_var_result[1], from_var_result[2])
			return line


		# Taking care of moving variable around
		if move_var != "":
			if to_var == "":
				print "Line: %d - Unable to extract destination location." % (lineNumber)
				exit(1)
			line = '%sMemMove((void**) &%s, %s);\n' % (pragma_prefix_funcname, move_var, to_var)
			return line


		# Taking care of registering variables
		if register_var != "":
			register_var_result = gREU.parseVariableString(register_var)
			if register_var_result is None:
				print "Line: %d - Unable to extract variable name and its corresponding range for '%s' keyword." % (lineNumber, 'register')
				exit(1)
			if loc_kw == "":
				print "Line: %d - Unable to extract destination location." % (lineNumber)
				exit(1)
			if _type == "":
				print "Line: %d - Unable to extract the type." % (lineNumber)
				exit(1)

			line = '%sMemRegister(%s, %s, %s, sizeof(%s), %s);\n' % (pragma_prefix_funcname, register_var_result[0], register_var_result[1], register_var_result[2], _type, loc_kw)
			return line

		# Taking care of unregistering variables
		if unregister_var != "":
			line = '%sMemUnregister(%s);\n' % (pragma_prefix_funcname, unregister_var)
			return line


		# Taking care of distances
		if is_auto and is_realloc:
			print "Line: %d - 'auto' and 'realloc' should not be set simultaneously." % (lineNumber)
			exit(1)
		# default behavior is "realloc"
		if not is_auto and not is_realloc:
			is_realloc = True

		if is_auto:
			distance_alloc_type = "GECKO_DISTANCE_ALLOC_TYPE_AUTO"
		elif is_realloc:
			distance_alloc_type = "GECKO_DISTANCE_ALLOC_TYPE_REALLOC"

		if distance == "":
			distance = "GECKO_DISTANCE_NOT_SET"
		elif "near" in distance:
			distance = "GECKO_DISTANCE_NEAR"
		elif "far" in distance:
			dist_list = distance.split(":")
			distance = "GECKO_DISTANCE_FAR"
			distance_level = 1
			if len(dist_list) > 1:
				distance_level = dist_list[1]
		else:
			distance = "GECKO_DISTANCE_UNKNOWN"


		if name is None:
			print "Line %d - Cannot recognize the variable %s" % (lineNumber, name)
			exit(1)
		name_list = gREU.parseVariableString(name)
		if name_list is None:
			print "Line %d - Cannot recognize the variable %s" % (lineNumber, name)
			exit(1)
		name = name_list[0]
		# name_list[1] should always be 0!
		count = name_list[2]

		if filename_permanent is None:
			filename_permanent = 'NULL'


		if "gecko_" in _type:
			line = '%sMemoryInternalTypeDeclare(%s, sizeof(%s), %s, %s, %s, %s, %s);\n' % (pragma_prefix_funcname, name, _type[6:], count, loc, distance, distance_level, distance_alloc_type)
		else:
			line = '%sMemoryDeclare((void**)&%s, sizeof(%s), %s, %s, %s, %s, %s, %s);\n' % (pragma_prefix_funcname, name, _type, count, loc, distance, distance_level, distance_alloc_type, filename_permanent)

		return line


	def processRegion(self, keywords, lineNumber):
		at = ""
		exec_pol = ""
		exec_pol_int = ""
		var_list = ""
		variable_list_internal = ""
		end = False
		wait = False
		gang = False
		gang_count = -1
		vector = False
		vector_count = -1
		kernels = False
		reduction_list = list()
		intensity = ""
		collapse = ""
		independent_loop = ""

		i = 3
		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "at":
				at = k[1][:-1]
			elif k[0] == "exec_pol":
				exec_pol = k[1][:-1]
			elif k[0] == "exec_pol_int":
				exec_pol_int = k[1][:-1]
			elif k[0] == "variable_list":
				var_list = k[1][:-1]
			elif k[0] == "variable_list_internal":
				variable_list_internal = k[1][:-1]
			elif k[0] == "end":
				end = True
			elif k[0] == "pause":
				wait = True
			elif k[0] == "gang":
				gang = True
				if len(k) > 1:
					gang_count = int(k[1][:-1])
			elif k[0] == "vector":
				vector = True
				if len(k) > 1:
					vector_count = int(k[1][:-1])
			elif k[0] == "kernels":
				kernels = True
			elif k[0] == "reduction":
				reduction_list.append(k[1][:-1])
			elif k[0] == "intensity":
				intensity = k[1][:-1]
			elif k[0] == "collapse":
				collapse = "collapse(%s)" % (k[1][:-1])
			elif k[0] == "independent":
				independent_loop = "independent"

			i += 1



		if end:
			self.parsing_region_state = -1
			return self.parseRegionKernel(keywords, lineNumber)
		elif wait:
			if at == "":
				at = '""'
			line = "%sWaitOnLocation(%s);\n" % (pragma_prefix_funcname, at)
			return line

		if len(exec_pol) > 0 and len(exec_pol_int) > 0:
			print "Line: %d - Cannot have 'exec_pol' and 'exec_pol_int' at the same time." % (lineNumber)
			exit(1)


		if (gang or vector) and kernels:
			print "Line: %d - the 'kernels' contruct in OpenACC could not be utilized while 'gang' and 'vector' are chosen." % (lineNumber)
			exit(1)



		self.gang = gang
		self.gang_count = gang_count
		self.vector = vector
		self.vector_count = vector_count
		self.kernels = kernels
		self.reduction_list = reduction_list


		self.var_list = var_list
		self.variable_list_internal = variable_list_internal

		intensity_is_float, intensity_val_float = isFloat(intensity)
		if intensity_is_float:
			intensity = intensity_val_float
		else:
			if intensity == '':
				intensity = -1
		self.intensity = intensity


		self.exec_pol = exec_pol
		if at == '':
			at = '""'
		self.at = at


		if "range" in exec_pol:
			pol = gREU.parseRangePolicy("range", exec_pol)
			if pol is None:
				pol = exec_pol.split(":")[1][:-1]
				if "," not in pol:
					print "Line: %d - 'Range' execution policy should include the ranges." % (lineNumber)
					exit(1)
				self.exec_pol_type = "runtime"
			else:
				self.exec_pol_type = "array"

			self.exec_pol = '"range"'
			self.exec_pol_option = pol

		elif "percentage" in exec_pol:
			pol = gREU.parseRangePolicy("percentage", exec_pol)
			if pol is None:
				pol = exec_pol.split(":")[1][:-1]
				print pol
				if "," not in pol:
					print "Line: %d - 'Percentage' execution policy should include the percentages." % (lineNumber)
					exit(1)
				self.exec_pol_type = "runtime"
			else:
				self.exec_pol_type = "array"

			self.exec_pol = '"percentage"'
			self.exec_pol_option = pol



		if autoGenerateACCPragmas:
			self.parsing_region_state = 2
		else:
			self.parsing_region_state = 1

		return ""

	def generateRangeLine(self):
		range_line_begin = ""
		range_line_end = ""

		if self.exec_pol == '"range"':
			if self.exec_pol_type == "array":
				range_arr = self.exec_pol_option.split(',')
				ranges_count = "%d" % (len(range_arr))
				range_line_begin = "int ranges_count = %s;\n" % (ranges_count)
				range_line_begin += "float *ranges = (float*) malloc(sizeof(float) * ranges_count);\n" 
				for i, a in enumerate(range_arr):
					range_line_begin += "ranges[%d] = %s;\n" % (i, a)
				range_line_end = "free(ranges);\n"

			elif self.exec_pol_type == "runtime":
				arr = self.exec_pol_option.split(',')
				ranges_count = arr[0]
				range_name = arr[1]
				range_line_begin = "int ranges_count = %s;\n" % (ranges_count)
				range_line_begin += "float *ranges = &%s[0];\n" % (range_name)

		elif self.exec_pol == '"percentage"':
			if self.exec_pol_type == "array":
				range_arr = self.exec_pol_option.split(',')
				ranges_count = "%d" % (len(range_arr))
				range_line_begin = "int ranges_count = %s;\n" % (ranges_count)
				range_line_begin += "float *ranges = (float*) malloc(sizeof(float) * ranges_count);\n" 
				for i, a in enumerate(range_arr):
					range_line_begin += "ranges[%d] = %s;\n" % (i, a)
				range_line_end = "free(ranges);\n"

			elif self.exec_pol_type == "runtime":
				arr = self.exec_pol_option.split(',')
				ranges_count = arr[0]
				range_name = arr[1]
				range_line_begin = "int ranges_count = %s;\n" % (ranges_count)
				range_line_begin += "float *ranges = &%s[0];\n" % (range_name)
		else:
			range_line_begin = "int ranges_count = 0;\n"
			range_line_begin += "float *ranges = NULL;\n"

		return range_line_begin, range_line_end


	def generateVarLine(self):
		temp_varList = self.var_list.split(",")
		varList2 = []
		for v in temp_varList:
			v = v.strip()
			if v == "":
				continue
			varList2.append(v)

		temp_varList = varList2

		var_line_before = ""
		var_line_after = ""
		var_line_end = ""

		var_line_before += "int var_count = %d;\n" % (len(temp_varList))
		if len(temp_varList) == 0:
			var_line_before += "void **var_list = NULL;\n"
			var_line_before += "void **old_var_list = NULL;\n"
		else:
			var_line_before += "void **var_list = (void **) malloc(sizeof(void*) * var_count);\n"
			var_line_before += "void **old_var_list = (void **) malloc(sizeof(void*) * var_count);\n"
			# var_line_before += "int __v_id = 0;\n"
			# for v in temp_varList:
			# 	var_line_before += "var_list[__v_id] = %s;\n" % (v)
			# 	var_line_before += "__v_id++;\n"
			for v_index, v in enumerate(temp_varList):
				var_line_before += "old_var_list[%d] = var_list[%d] = %s;\n" % (v_index, v_index, v)
				# var_line_after += "%s = var_list[%d];\n" % (v, v_index)
				var_line_after += "memcpy((void**) &%s, &var_list[%d], sizeof(void*));\n" % (v, v_index)
				# var_line_end += "%s = old_var_list[%d];\n" % (v, v_index)
				var_line_end += "memcpy((void**) &%s, &old_var_list[%d], sizeof(void*));\n" % (v, v_index)

		return var_line_before, var_line_after, var_line_end



	def generateVarListInternalClause(self):
		varList = self.variable_list_internal.split(",")
		varList2 = []
		for v in varList:
			v = v.strip()
			if v == "":
				continue
			varList2.append(v)

		varList = varList2

		var_clause = ""
		for v in varList:
			var_clause += "%s[0:1]," % (v)
		if len(varList) > 0:
			var_clause = var_clause[0:-1]

		return var_clause



	def generatePragmaAccClause(self):
		if self.kernels:
			return "#pragma acc kernels"


		pragma = "#pragma acc parallel loop "
		if self.gang:
			pragma += "gang "
			if self.gang_count > 0:
				pragma += "num_gangs(%d) " % (self.gang_count)

		if self.vector:
			pragma += "vector "
			if self.vector_count > 0:
				pragma += "vector_length(%d) " % (self.vector_count)

		reduction_stmt = ""
		for reduc in self.reduction_list:
			reduction_stmt += " reduction(%s) " % (reduc)
		pragma += reduction_stmt

		return pragma, reduction_stmt



	def parseRegionKernel(self, keywords, lineNumber):
		if self.parsing_region_state == -1:
			# we are dealing with the OpenACC pragma line after our directive

			var_list_line_before, var_list_line_after, var_list_line_end = self.generateVarLine()

			ret = ""
			# ret  += "#pragma acc wait(devIndex)\ngeckoUnsetBusy(dev[devIndex]);\n"
			ret += "#pragma acc wait(asyncID)\n"
			ret += "} // end of if(dev[devIndex]!=NULL)\n"
			ret += "} // end of OpenMP pragma \n"
			ret += "} // end of checking: err != GECKO_ERR_TOTAL_ITERATIONS_ZERO \n"
			ret += var_list_line_end
			ret += "geckoFreeRegionTemp(beginLoopIndex, endLoopIndex, devCount, dev, var_count, var_list, old_var_list);\n"
			ret += "}\n"

			self.parsing_region_state = 0

			return ret

		elif self.parsing_region_state == 1:
			# we are dealing with the OpenACC pragma line after our directive

			line = ' '.join(keywords)
			restOfPragma = gREU.parsePragmaACC(line)
			if restOfPragma is None or restOfPragma == "":
				print "Line: %d - After '#pragma %s', an OpenACC pragma should exist." % (lineNumber, pragma_keyword)
				exit(1)

			self.parsing_region_state = 2

			self.pragmaForRegion = "#pragma acc %s" % (restOfPragma)

			return ""

		elif self.parsing_region_state == 2:
			# we are dealing with for loop after OpenACC pragma and after our directive

			line = ' '.join(keywords)
			for_loop = gREU.parseForLoop(line)
			if for_loop is None:
				print "Line: %d - Unrecognizable for-loop format. [%s]" % (lineNumber, str(for_loop))
				exit(1)

			(datatype, varname, initval, varcond, cond, boundary, inc, paranthesis) = for_loop
			# print "for_loop:",for_loop

			incremental_direction = None
			for c in ["++", "+=", "*="]:
				if c in inc:
					incremental_direction = 1
			for c in ["--", "-=", "/="]:
				if c in inc:
					incremental_direction = 0


			if incremental_direction is None:
				print "Line: %d - Unknown iteration statment. Unrecognizable for-loop format." % (lineNumber)
				exit(1)


			has_equal_sign = 0
			if "=" in cond:
				has_equal_sign = 1

			if datatype is None:
				datatype = ""


			range_line_begin = ""
			range_line_end = ""
			range_line_begin, range_line_end = self.generateRangeLine()


			var_list_line_before, var_list_line_after, var_list_line_end = self.generateVarLine()
			var_list_internal_clause = self.generateVarListInternalClause()


			reduction_stmt_omp = ""

			# The following line corresponds to the new approach: no "#pragma acc" by the user! 
			# The "#pragma gecko region" will generate it!
			if autoGenerateACCPragmas:
				self.pragmaForRegion, reduction_stmt_omp = self.generatePragmaAccClause()


			line = '{\n'
			line += "int *beginLoopIndex=NULL, *endLoopIndex=NULL, jobCount, devCount, devIndex;\n"
			line += "GeckoLocation **dev = NULL;\n"
			line += range_line_begin		# this line contains 'ranges_count' and 'ranges'
			line += var_list_line_before
			line += 'GeckoError err = %sRegion(%s, %s, %s, %s, %d, %d, &devCount, &beginLoopIndex, &endLoopIndex, &dev, ranges_count, ranges, var_count, var_list, %s);\n' \
					 % (pragma_prefix_funcname, self.exec_pol, self.at, initval, boundary, incremental_direction, has_equal_sign, self.intensity)
			line += var_list_line_after
			line += range_line_end
			# line += "geckoRegionDistribute(&devCount, beingID, endID);\n"
#			if self.exec_pol in ['"range"', '"percentage"']:
#				line += "jobCount = ranges_count;\n"
#			else:
#				line += "jobCount = devCount;\n"

			line += "jobCount = devCount;\n"
			line += "if(err != GECKO_ERR_TOTAL_ITERATIONS_ZERO) {\n"
			# line += "for(devIndex=0;devIndex < jobCount;devIndex++) \n"
			line += "#pragma omp parallel num_threads(jobCount) %s\n" % (reduction_stmt_omp)
			line += "{\n"
			line += "int devIndex = omp_get_thread_num();\n"
			# line += "%sSetDevice(dev[devIndex]);\n" % (pragma_prefix_funcname)
			line += "if(dev[devIndex] != NULL) {\n"
			#line += "%sBindLocationToThread(devIndex, dev[devIndex]);\n"  % (pragma_prefix_funcname)
			line += "int beginLI = beginLoopIndex[devIndex], endLI = endLoopIndex[devIndex];\n"
			line += "int asyncID = dev[devIndex]->getAsyncID();\n"
			line += "int locType = (int) dev[devIndex]->getLocationType().type;\n"


			# generating OpenMP
			# line += "if(locType == GECKO_X32 || locType == GECKO_X64) {\n"

			line += "%s deviceptr(%s) async(asyncID) " % (self.pragmaForRegion, self.var_list)
			if var_list_internal_clause == "":
				line += "\n"
			else:
				line += " copyin(%s)\n" % (var_list_internal_clause)


			line += "for(%s %s = %s;%s %s %s;%s)" % (datatype, varname, "beginLI", varcond, cond, "endLI", inc)

			if paranthesis is None:
				line += "\n"
			else:
				line += " {\n"

			self.parsing_region_state = 0

			return line

		return None







	def extractVarList(self):
		temp_varList = self.var_list.split(",")
		varList2 = []
		for v in temp_varList:
			v = v.strip()
			if v == "":
				continue
			varList2.append(v)

		return varList2



	def generatePragmaAccOmpClauseNewApproach(self, presentClause, collapse, independent_loop):

		reduction_stmt = ""
		for reduc in self.reduction_list:
			reduction_stmt += " reduction(%s) " % (reduc)


		privateVars = self.extractVarList()
		privateVarsString = ""
		if len(privateVars) != 0:
			for pv in privateVars:
				privateVarsString += "%s," % (pv)

		if len(privateVarsString) > 0:
			privateVarsString = " firstprivate(%s) " % (privateVarsString[:-1])


		pragmaACC = ""

		if self.kernels:
			pragmaACC = "#pragma acc kernels"
		else:
			pragmaACC = "#pragma acc parallel loop "
			if self.gang:
				pragmaACC += "gang "
				if self.gang_count > 0:
					pragmaACC += "num_gangs(%d) " % (self.gang_count)

			if self.vector:
				pragmaACC += "vector "
				if self.vector_count > 0:
					pragmaACC += "vector_length(%d) " % (self.vector_count)


		pragmaACC += "%s %s async(asyncID) %s %s" % (reduction_stmt, presentClause, collapse, independent_loop)

		pragmaOMP = "#pragma omp parallel for " + reduction_stmt

		
		return pragmaACC, pragmaOMP, reduction_stmt, privateVarsString



	def generateDevicePtrClause(self):
		temp_varList = self.extractVarList()
		if len(temp_varList) == 0:
			return ""

		presentStr = ""
		for p in temp_varList:
			presentStr += "%s," % p
		presentStr = presentStr[:-1]
		return "deviceptr(%s)" % (presentStr)



	def processRegionNewApproach(self, keywords, lineNumber, lines_list):
		at = ""
		exec_pol = ""
		exec_pol_int = ""
		var_list = ""
		variable_list_internal = ""
		end = False
		wait = False
		gang = False
		gang_count = -1
		vector = False
		vector_count = -1
		kernels = False
		reduction_list = list()
		intensity = ""
		collapse = ""
		independent_loop = ""

		# parsing the region line:
		i = 3
		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "at":
				at = k[1][:-1]
			elif k[0] == "exec_pol":
				exec_pol = k[1][:-1]
			elif k[0] == "exec_pol_int":
				exec_pol_int = k[1][:-1]
			elif k[0] == "variable_list":
				var_list = k[1][:-1]
			elif k[0] == "variable_list_internal":
				variable_list_internal = k[1][:-1]
			elif k[0] == "end":
				end = True
			elif k[0] == "pause":
				wait = True
			elif k[0] == "gang":
				gang = True
				if len(k) > 1:
					gang_count = int(k[1][:-1])
			elif k[0] == "vector":
				vector = True
				if len(k) > 1:
					vector_count = int(k[1][:-1])
			elif k[0] == "kernels":
				kernels = True
			elif k[0] == "reduction":
				reduction_list.append(k[1][:-1])
			elif k[0] == "intensity":
				intensity = k[1][:-1]
			elif k[0] == "collapse":
				collapse = "collapse(%s)" % (k[1][:-1])
			elif k[0] == "independent":
				independent_loop = "independent"

			i += 1


		if wait:
			if at == "":
				at = '""'
			line = "%sWaitOnLocation(%s);\n" % (pragma_prefix_funcname, at)
			return line, lineNumber


		if len(exec_pol) > 0 and len(exec_pol_int) > 0:
			print "Line: %d - Cannot have 'exec_pol' and 'exec_pol_int' at the same time." % (lineNumber)
			exit(1)


		if (gang or vector) and kernels:
			print "Line: %d - the 'kernels' contruct in OpenACC could not be utilized while 'gang' and 'vector' are chosen." % (lineNumber)
			exit(1)



		self.gang = gang
		self.gang_count = gang_count
		self.vector = vector
		self.vector_count = vector_count
		self.kernels = kernels
		self.reduction_list = reduction_list


		self.var_list = var_list
		self.variable_list_internal = variable_list_internal

		intensity_is_float, intensity_val_float = isFloat(intensity)
		if intensity_is_float:
			intensity = intensity_val_float
		else:
			if intensity == '':
				intensity = -1
		self.intensity = intensity


		self.exec_pol = exec_pol
		if at == '':
			at = '""'
		self.at = at


		if "range" in exec_pol:
			pol = gREU.parseRangePolicy("range", exec_pol)
			if pol is None:
				pol = exec_pol.split(":")[1][:-1]
				if "," not in pol:
					print "Line: %d - 'Range' execution policy should include the ranges." % (lineNumber)
					exit(1)
				self.exec_pol_type = "runtime"
			else:
				self.exec_pol_type = "array"

			self.exec_pol = '"range"'
			self.exec_pol_option = pol

		elif "percentage" in exec_pol:
			pol = gREU.parseRangePolicy("percentage", exec_pol)
			if pol is None:
				pol = exec_pol.split(":")[1][:-1]
				print pol
				if "," not in pol:
					print "Line: %d - 'Percentage' execution policy should include the percentages." % (lineNumber)
					exit(1)
				self.exec_pol_type = "runtime"
			else:
				self.exec_pol_type = "array"

			self.exec_pol = '"percentage"'
			self.exec_pol_option = pol



		all_lines_in_kernel = list()
		i = lineNumber
		while i < len(lines_list):
			line = lines_list[i]
			line = line.strip()
			__keywords = line.split()
			if len(__keywords) >= 4 and __keywords[0] == "#pragma" and __keywords[1] == "gecko" and __keywords[2] == "region" and __keywords[3] == "end":
				break
			all_lines_in_kernel.append(line + "\n")
			i += 1


		for_loop = None
		for line_index, line in enumerate(all_lines_in_kernel):
			for_loop = gREU.parseForLoop(line)
			if for_loop is not None:
				break

		if for_loop is None:
			print "Line: %d - Unrecognizable for-loop format." % (lineNumber)
			exit(1)
		(datatype, varname, initval, varcond, cond, boundary, inc, paranthesis) = for_loop

		incremental_direction = None
		for c in ["++", "+=", "*="]:
			if c in inc:
				incremental_direction = 1
		for c in ["--", "-=", "/="]:
			if c in inc:
				incremental_direction = 0


		if incremental_direction is None:
			print "Line: %d - Unknown iteration statment. Unrecognizable for-loop format." % (lineNumber)
			exit(1)


		lineNumber = i+1


		has_equal_sign = 0
		if "=" in cond:
			has_equal_sign = 1

		if datatype is None:
			datatype = ""



		range_line_begin, range_line_end = self.generateRangeLine()

		var_list_line_before, var_list_line_after, var_list_line_end = self.generateVarLine()
		# var_list_internal_clause = self.generateVarListInternalClause()

		devicePtrClause = self.generateDevicePtrClause()

		pragmaACCRegion, pragmaOMPRegion, reduction_stmt_omp, privateVarsString = self.generatePragmaAccOmpClauseNewApproach(devicePtrClause, collapse, independent_loop)



		all_lines_in_kernel_acc = copy.deepcopy(all_lines_in_kernel)
		all_lines_in_kernel_omp = copy.deepcopy(all_lines_in_kernel)

		all_lines_in_kernel_acc.insert(line_index, pragmaACCRegion+"\n")
		all_lines_in_kernel_omp.insert(line_index, pragmaOMPRegion+"\n")


		for_line = "for(%s %s = %s;%s %s %s;%s)" % (datatype, varname, "beginLI", varcond, cond, "endLI", inc)
		if paranthesis is None:
			for_line += "\n"
		else:
			for_line += " {\n"

		del all_lines_in_kernel_acc[line_index+1]
		del all_lines_in_kernel_omp[line_index+1]
		all_lines_in_kernel_acc.insert(line_index+1, for_line)
		all_lines_in_kernel_omp.insert(line_index+1, for_line)

		line = '{\n'
		line += "int *beginLoopIndex=NULL, *endLoopIndex=NULL, jobCount, devCount, devIndex;\n"
		line += "GeckoLocation **dev = NULL;\n"
		line += range_line_begin		# this line contains 'ranges_count' and 'ranges'
		line += var_list_line_before
		line += 'GeckoError err = %sRegion(%s, %s, %s, %s, %d, %d, &devCount, &beginLoopIndex, &endLoopIndex, &dev, ranges_count, ranges, var_count, var_list, %s);\n' \
				 % (pragma_prefix_funcname, self.exec_pol, self.at, initval, boundary, incremental_direction, has_equal_sign, self.intensity)

		line += "if(err != GECKO_ERR_TOTAL_ITERATIONS_ZERO) {\n"
		line += var_list_line_after
		line += range_line_end
		line += "#pragma omp parallel num_threads(devCount) %s %s\n" % (reduction_stmt_omp, privateVarsString)
		line += "{\n"
		line += "int devIndex = omp_get_thread_num();\n"
		line += "if(dev[devIndex] != NULL) {\n"
		line += "int beginLI = beginLoopIndex[devIndex], endLI = endLoopIndex[devIndex];\n"
		line += "int asyncID = dev[devIndex]->getAsyncID();\n"
		line += "int locType = (int) dev[devIndex]->getLocationType().type;\n"


		# generating OpenMP
		line += "if(locType == GECKO_X32 || locType == GECKO_X64) {\n"
		line += ' '.join(all_lines_in_kernel_omp)
		line += "} else if(locType == GECKO_NVIDIA) {\n"
		# generating OpenACC
		line += ' '.join(all_lines_in_kernel_acc)
		line += "}\n"

		var_list_line_before, var_list_line_after, var_list_line_end = self.generateVarLine()

		# line += "#pragma acc wait(asyncID)\n"
		line += "} // end of if(dev[devIndex]!=NULL)\n"
		line += "#pragma acc wait\n"
		line += "} // end of OpenMP pragma \n"
		# line += "#pragma acc wait\n"
		line += var_list_line_end
		line += "} // end of checking: err != GECKO_ERR_TOTAL_ITERATIONS_ZERO \n"
		line += "geckoFreeRegionTemp(beginLoopIndex, endLoopIndex, devCount, dev, var_count, var_list, old_var_list);\n"
		line += "}\n"


		return line, lineNumber












	def processDraw(self, keywords, lineNumber):
		root = ""
		filename = '"gecko.dot"'

		i = 3
		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "root":
				root = k[1][:-1]
			elif k[0] == "filename":
				filename = k[1][:-1]
			i += 1

		if len(root) == 0:
			print "Line: %d - A root should be chosen." % (lineNumber)
			exit(1)


		line = "%sDrawHierarchyTree(%s, %s);" % (pragma_prefix_funcname, root, filename)

		return line



	def processConfig(self, keywords, lineNumber):
		root = ""
		filename = '"gecko.conf"'
		is_file_method_chosen = False
		env = '"GECKO_CONFIG_FILE"'
		is_env_method_chosen = False

		i = 3
		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "file":
				is_file_method_chosen = True
				if len(k) > 1:
					filename = k[1][:-1]
			elif k[0] == "env":
				is_env_method_chosen = True

			i += 1

		if is_file_method_chosen and is_env_method_chosen:
			print "Line: %d - The 'file' and 'env' methods could not be chosen at the same time." % (lineNumber)
			exit(1)


		if is_file_method_chosen:
			line = "%sLoadConfigWithFile(%s);" % (pragma_prefix_funcname, filename)
		elif is_env_method_chosen:
			line = "%sLoadConfigWithEnv();" % (pragma_prefix_funcname)
		else:
			print "Error in 'config' clause. Available options are: 'env' and 'file'."
			exit(1)

		return line


	def __process_location_range(self, combo):
		#      location, 	variable_name, 		start, 		count
		ret = ["", 			"", 				"", 		""]
		tmp = combo.split(".")
		ret[0] = tmp[0]
		tmp = tmp[1]
		if tmp[-1] != ']':
			print "Unable to retreive the variable properties"
			exit(1)
		tmp = tmp[:-1]
		tmp = tmp.split("[")
		ret[1] = tmp[0]
		tmp = tmp[1]
		tmp = tmp.split(":")
		ret[2] = tmp[0]
		ret[3] = tmp[1]

		return ret


	def processPut(self, keywords, lineNumber):
		if keywords[2][3] != "(" or keywords[2][-1] != ")":
			print "Error in PUT statement."
			exit(1)

		stmt = keywords[2][4:-1]
		from_to = stmt.split(",")
		from_arr = self.__process_location_range(from_to[0])
		to_arr = self.__process_location_range(from_to[1])

		line = 'chamPut("%s", (void**)&%s, %s, %s, "%s", (void**)&%s, %s, %s);\n' % (from_arr[0], from_arr[1], from_arr[2], from_arr[3],
			to_arr[0], to_arr[1], to_arr[2], to_arr[3])

		return line


	def processLine(self, line, lineNumber, real_line, lines_list):
		# keywords = line.split()
		keywords = line

		if self.parsing_region_state in [1, 2]:
			return self.parseRegionKernel(keywords, lineNumber), lineNumber

		if keywords[2] == "loctype":
			return self.processLocationType(keywords, lineNumber), lineNumber
		elif keywords[2] == "location":
			return self.processLocation(keywords, lineNumber), lineNumber
		elif keywords[2] == "hierarchy":
			return self.processHierarchy(keywords, lineNumber), lineNumber
		elif keywords[2] == "memory":
			return self.processMemory(keywords, lineNumber), lineNumber
		elif keywords[2] == "region":
			# return self.processRegion(keywords, lineNumber), lineNumber
			return self.processRegionNewApproach(keywords, lineNumber, lines_list)
		elif keywords[2] == "draw":
			return self.processDraw(keywords, lineNumber), lineNumber
		elif keywords[2] == "config":
			return self.processConfig(keywords, lineNumber), lineNumber
		elif keywords[2][0:3] == "put":
			return self.processPut(keywords, lineNumber), lineNumber
		else:
			print "Unrecognized %s clause - Line (%d): %s" % (pragma_keyword, lineNumber, real_line.strip())
			exit(1)
		return "\n", lineNumber

	def omitProblematicSpaces(self, line_to_process, lineNumber):
		keywords = line_to_process.split()

		i = 0
		n = len(keywords)
		finalKeywords = list()
		while i<n:
			kw = keywords[i]

			countOpen = 0
			while True:
				for w in kw:
					if w == '(':
						countOpen += 1
					elif w == ')':
						countOpen -= 1

					if countOpen < 0:
						print "Paranthesis do not match at line %d" % (lineNumber)
						exit(1)

				if countOpen == 0:
					break

				if countOpen > 0:
					i += 1
					if i >= n:
						print "Paranthesis do not match at line %d" % (lineNumber)
						exit(1)

					kw += (keywords[i] + " ")
					countOpen = 0


			if countOpen == 0:
				finalKeywords.append(kw)

			i += 1

		return finalKeywords
	
	def processFile(self, overwrite=False):
		f = open(self.filename, "r")
		lines = f.readlines()
		f.close()
		outputLines = list()
		lenLines = len(lines)
		i=0

		while i < lenLines:
			real_line = lines[i]
			l = lines[i]
			l = l.strip()

			if len(l) == 0 or l == "" or l == None:
				outputLines.append(real_line)
				i += 1
				continue

			if l[0:2]=="//":
				outputLines.append(real_line)
				i += 1
				continue

			# appending multiple lines ending with '\' with each other
			line_to_process = l
			if l[-1] == "\\":
				line_to_process = line_to_process[:-1]
				while True:
					line_to_process += lines[i+1].strip()
					if line_to_process[-1] == '\\':
						line_to_process = line_to_process[:-1]
						i += 1
					else:
						i += 1
						break

			line_to_process_list = line_to_process.split()						
			if line_to_process_list[0] == "#pragma" and line_to_process_list[1] == pragma_keyword:
				if preserveOriginalCodeAsComment:
					outputLines.append("//%s\n" % (line_to_process))
				# line_to_process_list = self.omitProblematicSpaces(line_to_process, i+1)
				line_to_append, i = self.processLine(line_to_process_list, i+1, real_line, lines)
				i += -1
				outputLines.append(line_to_append)
			elif self.parsing_region_state in [1, 2]:
				if preserveOriginalCodeAsComment:
					outputLines.append("//%s\n" % (line_to_process))
				# line_to_process_list = self.omitProblematicSpaces(line_to_process, i+1)
				line_to_append, i = self.processLine(line_to_process_list, i+1, real_line, lines)
				i += -1
				outputLines.append(line_to_append)
			else:
				outputLines.append(real_line)
			i += 1

		outputLines.insert(0, '#include "geckoRuntime.h"\n')

		output_filename = "output_"+self.filename
		if overwrite:
			output_filename = self.filename
		f = open(output_filename, "w")
		for l in outputLines:
			f.write(l)
		f.close()




gecko_orig_folder_name = "__GECKO__ORIG__"
gecko_conv_folder_name = "__GECKO__CONVERT__"
do_not_remove = False


def get_list_of_files(folder):
	ext_list = ["*.h", "*.hpp", "*.c", "*.cpp"]
	file_list = []
	for ext in ext_list:
		for f in glob.glob(folder+"/"+ext):
			if os.path.isfile(f):
				file_list.append(f)
	return file_list


def prune_file_list(file_list):
	pattern = r"#pragma[ ]+%s[ ]+" % (pragma_keyword)
	pruned_list = []
	for fname in file_list:
		f = open(fname, "r")
		lines = f.readlines()
		f.close()


		found = False
		for l in lines:
			match = re.search(pattern, l)
			if match is not None:
				found = True
				break

		if found:
			pruned_list.append(fname)

	return pruned_list


def forward_conversion(folder):
	global do_not_remove

	folder = os.path.relpath(folder)

	if os.path.exists("%s/%s" % (folder, gecko_orig_folder_name)) or os.path.exists("%s/%s" % (folder, gecko_conv_folder_name)):
		do_not_remove = True
		print "\n\nPlease revert back the changes with backward command.\n"
		return
	
	file_list = get_list_of_files(folder)
	file_list = prune_file_list(file_list)

	os.system("mkdir -p %s" % (gecko_orig_folder_name))
	for f in file_list:
		os.system("cp %s %s/" % (f, gecko_orig_folder_name))
	os.system("cp -r %s %s/" % (gecko_orig_folder_name, gecko_conv_folder_name))

	# os.system("rm -f log_astyle.log")
	for f in file_list:
		print "Converting %s..." % (f)
		new_filename = "%s/%s" % (gecko_conv_folder_name, f)
		src = SourceFile(new_filename)
		src.processFile(True)
		os.system("astyle %s &> /dev/null" % (new_filename))

	do_not_remove = True

	for f in file_list:
		os.system("cp %s/%s ./" % (gecko_conv_folder_name, f))




def backward_conversion(folder):
	global do_not_remove

	folder = os.path.relpath(folder)

	if not os.path.exists("%s/%s" % (folder, gecko_orig_folder_name)):
		do_not_remove = True
		print "\n\nPlease apply the changes with forward command.\n"
		return

	file_list = get_list_of_files("%s/%s" % (folder, gecko_orig_folder_name))
	for f in file_list:
		print "Converting %s..." % (f)
		os.system("cp %s ./" % (f))

	os.system("rm -rf %s/%s" % (folder, gecko_orig_folder_name))
	os.system("rm -rf %s/%s" % (folder, gecko_conv_folder_name))



# @atexit.register
# def check_for_exit():
# 	if do_not_remove:
# 		return
# 	os.system("rm -rf ./%s" % (gecko_orig_folder_name))
# 	os.system("rm -rf ./%s" % (gecko_conv_folder_name))





def usage():
	print "---------------------------------------------------------------------------------------------------------"
	print "\n%s script should be used as one of the following:\n" % (sys.argv[0])
	print "\t1. python %s  :  This will search hardcoded files within script and converts them." % (sys.argv[0])
	print "\t2. python %s {forward,backward} <folder_path>  :  This will convert files with a folder in forward/backward direction" % (sys.argv[0])
	print "\n"
	print "---------------------------------------------------------------------------------------------------------"


def main():
	if len(sys.argv) > 1:
		action = sys.argv[1]
		if action not in ["forward", "backward"]:
			usage()
			exit(1)
		if len(sys.argv) == 1:
			usage()
			exit(1)

		folder = sys.argv[2]
		if folder == "":
			folder = "./"

		folder = os.path.abspath(folder)
		if not os.path.exists(folder):
			usage()
			exit(1)

		folder_list = []
		gFU.findFolders(folder,folder_list)
		folder_list.append(folder)

		if action == "forward":
			for folder in folder_list:
				if "__GECKO__" in folder:
					continue
				os.chdir(folder)
				forward_conversion(folder)
		elif action == "backward":
			for folder in folder_list:
				if "__GECKO__" in folder:
					continue
				os.chdir(folder)
				backward_conversion(folder)

		return


	listOfFiles = glob.glob("*.h")
	listOfFiles += glob.glob("*.cpp")
	listOfFiles += glob.glob("*.cu")

	# listOfFiles = ["test.cpp"]
	listOfFiles = ["test.cpp", "test_with_config.cpp"]
	# listOfFiles = ["stencil.cpp", "dot_product.cpp", "matrix_mul.cpp"]
	print listOfFiles

	for f in listOfFiles:
		print "Processing file:", f
		src = SourceFile(f)
		src.processFile()
		os.system("astyle output_%s" % (f))



if __name__ == '__main__':
	main()
