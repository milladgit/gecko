
import os, sys, glob

def findFiles(folder, file_list):
	folder_list = []

	lst = glob.glob("%s/*" % (folder))
	for f in lst:
		if os.path.isfile(f):
			file_list.append(f)
		else:
			folder_list.append(f)


	for fol in folder_list:
		findFiles(fol, file_list)	


def findFolders(folder, intput_folder_list):
	tmp_folder_list = []

	lst = glob.glob("%s/*" % (folder))
	for f in lst:
		if not os.path.isfile(f):
			intput_folder_list.append(f)
			tmp_folder_list.append(f)


	for fol in tmp_folder_list:
		findFolders(fol, intput_folder_list)	


def findFoldersInLeaf(folder, intput_folder_list):
	tmp_folder_list = []

	lst = glob.glob("%s/*" % (folder))
	for f in lst:
		if not os.path.isfile(f):
			tmp_folder_list.append(f)

	if len(tmp_folder_list) == 0:
		intput_folder_list.append(folder)

	for fol in tmp_folder_list:
		findFoldersInLeaf(fol, intput_folder_list)	
