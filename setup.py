from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension
from os import path
import os
import subprocess
import io

## in development set version
PYPI_VERSION = '0.4.7'

# Return the git revision as a string (from numpy)
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


if PYPI_VERSION is None:
    PYPI_VERSION = git_version()


this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

packages = find_packages()

if __name__ == "__main__":
    setup(name = 'DoEgen',
          author            = "Sebastian Haan",
          author_email      = "sebastian.haan@sydney.edu.au",
          url               = "https://github.com/sebhaan/DoEgen",
          version           = PYPI_VERSION,
          description       = "DoEgen: A Python Library for Optimised Design of Experiment Generation and Evaluation",
          long_description  = long_description,
          long_description_content_type='text/markdown',
          install_requires  = ['numpy>=1.16.3', 
                              'xlrd==1.2.0',
                              'pandas>=1.0.3',
                              'XlsxWriter>=1.2.8',
                              'openpyxl>=3.0.7'
                              'seaborn>=0.11.1',
                              'OApackage==2.6.6',
                              'tabulate==0.8.7',
                              'matplotlib>=3.1.0',
                              'PyYAML>=5.3.1',
                              'scikit_learn>=0.22.2.post1'],
          python_requires   = '>=3.6',
          setup_requires    = ["pytest-runner", 'webdav'],
          tests_require     = ["pytest", 'webdav'],
          packages          = ['doegen'],
          package_data      = {'doegen': ['*.yaml',
                                          '*.xlsx',
                                        'test/Experiment_setup_test.xlsx',
                                        'test/settings_design_test.yaml',
                                        'test/settings_expresults_test.yaml',
                                        'test/results/experiment_results_Nrun72.xlsx',
                                        'test/results/Designtable_optimal_Nrun72.csv']},
          include_package_data = False,
          classifiers       = ['Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.6',
                               'Programming Language :: Python :: 3.7'
                               ]
          )