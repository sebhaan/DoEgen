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
if os.path.isfile('settings_design.yaml'):
	print('File settings_design.yaml already exists.')
else:
	source = os.path.join(pckd,'settings_design.yaml')
	target = os.path.join(cwd,'settings_design.yaml')
	try:
	    shutil.copy(source, target)
	    print('Please edit settings_design.yaml')
	except IOError as e:
	    print("Unable to copy file. %s" % e)
	except:
	    print("Unexpected error:", sys.exc_info())

# Create Experiment_setup.xlsx
if os.path.isfile('Experiment_setup.xlsx'):
	print('File Experiment_setup.xlsx already exist.')
else:
	source = os.path.join(pckd,'Experiment_setup.xlsx')
	target = os.path.join(cwd,'Experiment_setup.xlsx')
	try:
	    shutil.copy(source, target)
	    print('Please add your experiment settings in Experiment_setup.xlsx')
	except IOError as e:
	    print("Unable to copy file. %s" % e)
	except:
	    print("Unexpected error:", sys.exc_info())


# Create settings_expresults.yaml
if os.path.isfile('settings_expresults.yaml'):
	print('File settings_expresults.yaml already exists.')
else:
	source = os.path.join(pckd,'settings_expresults.yaml')
	target = os.path.join(cwd,'settings_expresults.yaml')
	try:
	    shutil.copy(source, target)
	    print('Please edit settings_expresults.yaml after running the experiment.')
	except IOError as e:
	    print("Unable to copy file. %s" % e)
	except:
	    print("Unexpected error:", sys.exc_info())

# Create Experiment_results.xlsx
if os.path.isfile('Experiment_results.xlsx'):
	print('File Experiment_results.xlsx already exist.')
else:
	source = os.path.join(pckd,'Experiment_results.xlsx')
	target = os.path.join(cwd,'Experiment_results.xlsx')
	try:
	    shutil.copy(source, target)
	    print('Please add your experiment results in Experiment_results.xlsx after running the experiment.')
	except IOError as e:
	    print("Unable to copy file. %s" % e)
	except:
	    print("Unexpected error:", sys.exc_info())