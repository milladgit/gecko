
import os,sys,re,glob,atexit
import geckoREUtilities as gREU

pragma_keyword 				= "gecko"
pragma_prefix  				= "GECKO"
pragma_prefix_funcname 		= "gecko"


preserveOriginalCodeAsComment = False


class SourceFile(object):
	"""docstring for SourceFile"""
	def __init__(self, filename):
		super(SourceFile, self).__init__()
		self.filename = filename
		self.line_for_end = ""

		self.parsing_region_state = 0
		self.pragmaForRegion = ""

		self.var_list = ""
		self.exec_pol = ""
		self.exec_pol_type = ""
		self.exec_pol_option = ""
		self.at = ""


	def processLocationType(self, keywords, lineNumber):
		name = "-"
		family = "X64"
		micro_arch = '"-"'
		num_cores = "0"
		mem_size = '"-"'
		mem_type = '"-"'
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

				if family in ["CC2.0", "CC3.0", "CC4.0", "CC5.0"]:
					family = "CUDA"

				if family not in ("X64", "X32", "CUDA", "UNIFIED_MEMORY"):
					print "Line %d - Error in kind of locationtype - Unknown family (%s)" % (lineNumber, family)
					exit(1)
			elif k[0] == "num_cores":
				num_cores = k[1][:-1]
			elif k[0] == "mem":
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
			i += 1

		line = '%sLocationtypeDeclare(%s, %s_%s, %s, %s, %s, %s);\n' % (pragma_prefix_funcname, name, pragma_prefix, family, micro_arch, num_cores, mem_size, mem_type)
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
		loc = ""
		distribute = False
		duplicate = False
		tile = ""
		varToFree = ""
		i = 3
		while i < len(keywords):
			k = keywords[i].split("(")
			if k[0] == "allocate":
				name = k[1][:-1]
			elif k[0] == "type":
				_type = k[1][:-1]
			elif k[0] == "location":
				loc = k[1][:-1]
			elif k[0] == "free":
				varToFree = k[1][:-1]

			i += 1


		if varToFree != "":
			line = "%sFree(%s);\n" % (pragma_prefix_funcname, varToFree)
			return line

		if duplicate and distribute and tile:
			print "Line %d - Cannot choose <duplicate>, <distribute>, <tile> at the same time!" % (lineNumber)
			exit(1)

		# if not duplicate and not distribute and not tile:
		# 	distribute = True

		# variable_create = False
		name_list = name.strip().split("[")
		if len(name_list) == 1:
			name = name_list[0]
			count = "1"
			# variable_create = True
		elif len(name_list) != 2:
			print "Line %d - Cannot recognize the variable %s" % (lineNumber, name)
			exit(1)	
		else:	
			name = name_list[0]
			prop = name_list[1][:-1]
			prop = prop.split(":")
			if len(prop) != 2:
				print "Line: %d - Cannot extract length of variable %s as in (%s)" % (lineNumber, name, ' '.join(prop))
				exit(1)
			count = prop[1]


		tileDict = dict()
		tile = tile.split(",")
		for t in tile:
			t = t.strip()
			if t=="" or len(t) == 0:
				break
			t = t.split("[")
			t[-1] = t[-1][:-1]
			tileDict[t[0]] = t[1]

		# type_of_distribution = "CHAMELEON_DISTRIB_NONE"
		# if distribute:
		# 	type_of_distribution = "CHAMELEON_DISTRIB_DISTRIBUTE"
		# elif duplicate:
		# 	type_of_distribution = "CHAMELEON_DISTRIB_DUPLICATE"
		# elif len(tileDict) > 0:
		# 	type_of_distribution = "CHAMELEON_DISTRIB_TILE"
		# else:
		# 	type_of_distribution = "CHAMELEON_DISTRIB_UNKNOWN"


		line = ""
		# if not variable_create:
		# 	line = '%sVariableDeclare((void**)&%s, sizeof(%s), %s, %s);\n' % (pragma_prefix_funcname, name, _type, count, loc)
		line = '%sMemoryDeclare((void**)&%s, sizeof(%s), %s, %s);\n' % (pragma_prefix_funcname, name, _type, count, loc)

		# if len(tileDict) > 0:
		# 	for machine, indexes in tileDict.items():
		# 		ind = indexes.split(":")
		# 		line += '%sVariableTile(%s, "%s", %s, %s);\n' % (pragma_prefix_funcname, name, machine, ind[0], ind[1])

		return line

	def processRegion(self, keywords, lineNumber):
		at = ""
		exec_pol = ""
		exec_pol_int = ""
		varList = ""
		end = False
		wait = False

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
				varList = k[1][:-1]
			elif k[0] == "end":
				end = True
			elif k[0] == "pause":
				wait = True
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


		self.var_list = varList

		self.exec_pol = exec_pol
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
				if "," not in pol:
					print "Line: %d - 'Percentage' execution policy should include the percentages." % (lineNumber)
					exit(1)
				self.exec_pol_type = "runtime"
			else:
				self.exec_pol_type = "array"

			self.exec_pol = '"percentage"'
			self.exec_pol_option = pol



		self.parsing_region_state = 1

		return ""


	def parseRegionKernel(self, keywords, lineNumber):
		if self.parsing_region_state == -1:
			# we are dealing with the OpenACC pragma line after our directive

			ret = ""
			# ret  += "#pragma acc wait(devIndex)\ngeckoUnsetBusy(dev[devIndex]);\n"
			ret += "}\ngeckoFreeRegionTemp(beginLoopIndex, endLoopIndex, devCount, dev);\n}\n"

			self.parsing_region_state = 0

			return ret

		elif self.parsing_region_state == 1:
			# we are dealing with the OpenACC pragma line after our directive

			line = ' '.join(keywords)
			restOfPragma = gREU.parsePragmaACC(line)
			if restOfPragma is None or restOfPragma == "":
				print "Line: %d - After '#pragma %s', an OpenACC pragma should exist." % (lineNumber, pragma_prefix)
				exit(1)

			self.parsing_region_state = 2

			self.pragmaForRegion = "#pragma acc %s" % (restOfPragma)

			return ""

		elif self.parsing_region_state == 2:
			# we are dealing with for loop after OpenACC pragma and after our directive

			line = ' '.join(keywords)
			for_loop = gREU.parseForLoop(line)
			if for_loop is None:
				print "Line: %d - Unrecognizable for-loop format." % (lineNumber)
			
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

			range_line_begin = ""
			range_line_end = ""
			if self.exec_pol == '"range"':
				if self.exec_pol_type == "array":
					range_arr = self.exec_pol_option.split(',')
					ranges_count = "%d" % (len(range_arr))
					range_line_begin = "int ranges_count = %s;\n" % (ranges_count)
					range_line_begin += "int *ranges = (int*) malloc(sizeof(int) * ranges_count);\n" 
					for i, a in enumerate(range_arr):
						range_line_begin += "ranges[%d] = %s;\n" % (i, a)
					range_line_end = "free(ranges);\n"

				elif self.exec_pol_type == "runtime":
					arr = self.exec_pol_option.split(',')
					ranges_count = arr[0]
					range_name = arr[1]
					range_line_begin = "int ranges_count = %s;\n" % (ranges_count)
					range_line_begin += "int *ranges = &%s[0];\n" % (range_name)

			elif self.exec_pol == '"percentage"':
				if self.exec_pol_type == "array":
					range_arr = self.exec_pol_option.split(',')
					ranges_count = "%d" % (len(range_arr))
					range_line_begin = "int ranges_count = %s;\n" % (ranges_count)
					range_line_begin += "int *ranges = (int*) malloc(sizeof(int) * ranges_count);\n" 
					for i, a in enumerate(range_arr):
						range_line_begin += "ranges[%d] = %s;\n" % (i, a)
					range_line_end = "free(ranges);\n"

				elif self.exec_pol_type == "runtime":
					arr = self.exec_pol_option.split(',')
					ranges_count = arr[0]
					range_name = arr[1]
					range_line_begin = "int ranges_count = %s;\n" % (ranges_count)
					range_line_begin += "int *ranges = &%s[0];\n" % (range_name)
			else:
				range_line_begin = "int ranges_count = 0;\n"
				range_line_begin += "int *ranges = NULL;\n"



			line = '{\n'
			line += "int *beginLoopIndex, *endLoopIndex, jobCount, devCount, devIndex;\n"
			line += "GeckoLocation **dev;\n"
			line += range_line_begin		# this line contains 'ranges_count' and 'ranges'
			line += '%sRegion(%s, %s, %s, %s, %d, &devCount, &beginLoopIndex, &endLoopIndex, &dev, ranges_count, ranges);\n' \
					 % (pragma_prefix_funcname, self.exec_pol, self.at, initval, boundary, incremental_direction)
			line += range_line_end
			# line += "geckoRegionDistribute(&devCount, beingID, endID);\n"
			if self.exec_pol in ['"range"', '"percentage"']:
				line += "jobCount = ranges_count;"
			else:
				line += "jobCount = devCount;"

			# line += "for(devIndex=0;devIndex < jobCount;devIndex++) \n"
			line += "#pragma omp parallel num_threads(jobCount)\n"
			line += "{\n"
			line += "int devIndex = omp_get_thread_num();\n"
			line += "%sSetDevice(dev[devIndex]);\n" % (pragma_prefix_funcname)
			line += "%s deviceptr(%s) async(dev[devIndex]->getAsyncID())\n" % (self.pragmaForRegion, self.var_list)
			if datatype is None:
				datatype = ""
			line += "for(%s %s = %s;%s %s %s;%s)" % (datatype, varname, "beginLoopIndex[devIndex]", varcond, cond, "endLoopIndex[devIndex]", inc)

			if paranthesis is None:
				line += "\n"
			else:
				line += " {\n"

			self.parsing_region_state = 0

			return line

		return None



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

		i = 3
		while i < len(keywords):
			k = keywords[i].split("(")
			if len(k) > 1 and k[0] == "file":
				filename = k[1][:-1]

			i += 1


		line = "%sLoadConfigWithFile(%s);" % (pragma_prefix_funcname, filename)

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


	def processLine(self, line, lineNumber, real_line):
		# keywords = line.split()
		keywords = line

		if self.parsing_region_state in [1, 2]:
			return self.parseRegionKernel(keywords, lineNumber)

		if keywords[2] == "loctype":
			return self.processLocationType(keywords, lineNumber)
		elif keywords[2] == "location":
			return self.processLocation(keywords, lineNumber)
		elif keywords[2] == "hierarchy":
			return self.processHierarchy(keywords, lineNumber)
		elif keywords[2] == "memory":
			return self.processMemory(keywords, lineNumber)
		elif keywords[2] == "region":
			return self.processRegion(keywords, lineNumber)
		elif keywords[2] == "draw":
			return self.processDraw(keywords, lineNumber)
		elif keywords[2] == "config":
			return self.processConfig(keywords, lineNumber)
		elif keywords[2][0:3] == "put":
			return self.processPut(keywords, lineNumber)
		else:
			print "Unrecognized %s clause - Line (%d): %s" % (pragma_keyword, lineNumber, real_line.strip())
			exit(1)
		return "\n"

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

					kw += keywords[i]
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
				line_to_process_list = self.omitProblematicSpaces(line_to_process, i+1)
				# outputLines.append(self.processLine(' '.join(line_to_process), i+1))
				outputLines.append(self.processLine(line_to_process_list, i+1, real_line))
			elif self.parsing_region_state in [1, 2]:
				if preserveOriginalCodeAsComment:
					outputLines.append("//%s\n" % (line_to_process))
				line_to_process_list = self.omitProblematicSpaces(line_to_process, i+1)
				# outputLines.append(self.processLine(' '.join(line_to_process), i+1))
				outputLines.append(self.processLine(line_to_process_list, i+1, real_line))
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


def convert_based_on_actions(action, folder):
	if action == "forward":
		forward_conversion(folder)
	elif action == "backward":
		backward_conversion(folder)





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
			usage()
			exit(1)
		if not os.path.exists(folder):
			usage()
			exit(1)

		convert_based_on_actions(action, folder)

		return


	listOfFiles = glob.glob("*.h")
	listOfFiles += glob.glob("*.cpp")
	listOfFiles += glob.glob("*.cu")

	# listOfFiles = ["test.cpp"]
	listOfFiles = ["test.cpp", "test_with_config.cpp"]
	# listOfFiles = ["stencil.cpp", "dot_product.cpp", "matrix_mul.cpp"]
	print listOfFiles

	for f in listOfFiles:
		src = SourceFile(f)
		src.processFile()
		os.system("astyle output_%s" % (f))



if __name__ == '__main__':
	main()
