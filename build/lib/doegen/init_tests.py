"""
Run this initialisation of .yaml and .xlsx files after installation of doegen.

Creates config yaml files and excel templates if not already existing in current working directory.
"""
import os
import sys
import shutil
import doegen

#Current working directory
cwd = os.getcwd()

#Directory where package doegen is installed:
pckd = doegen.__path__[0] 

# Create settings_design.yaml
if os.path.exists('test'):
	print("directory already exist: test")
else:
	source = os.path.join(pckd,'test')
	target = os.path.join(cwd,'test')
	try:
		shutil.copytree(source, target)
		print("Generated directory: test")
	except IOError as e:
		print("Unable to copy directory. %s" % e)
	except:
		print("Unexpected error:", sys.exc_info())