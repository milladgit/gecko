
import os,sys


gecko_translate_folder = sys.argv[0]
gecko_translate_folder = gecko_translate_folder[::-1]
ind = gecko_translate_folder.find("/")
gecko_translate_folder = gecko_translate_folder[ind:]
gecko_translate_folder = gecko_translate_folder[::-1]


forward = "python %sgeckoTranslate.py forward ." % (gecko_translate_folder)
backward = "python %sgeckoTranslate.py backward ." % (gecko_translate_folder)


make_cmd = ' '.join(sys.argv[1:])

os.system(forward)
os.system(make_cmd)
os.system(backward)

