#!/bin/env python
#DoEgen: A Python Library for Optimised Design of Experiment Generation and Evaluation
#
#Copyright 2020 Sebastian Haan
# 
#DoEgen is free software made available under the AGPL License. 
#For details see the LICENSE file.
#
#@author: Sebastian Haan

"""
DoEgen is a Python library aiming to assist in generating optimised
Design of Experiments (DoE), evaluating design efficiencies, and
analysing experiment results.

In a first step, optimised designs can be automatically generated and
efficiencies evaluated for any mixture of factor-levels for numeric and
categorical factors. Designs are automatically evaluated as function of
number of experiment runs and the most efficient designs are suggested.
In particular DoEgen provides computation of a wide range of design
efficiencies and allows to import and evaluate externally generated
designs as well.

The second part of DoEgen assists in analysing any derived experiment
results in terms of factor importance, correlations, and response
analysis for best parameter space selection.

Definitions
-----------

An Experiment Design is typically defined by:

-   Number of Factors: the parameters or variates of the experiment
-   Number of Runs: the number of experiments
-   Levels: The number of value options for each factor, which can be
    either numeric values (discrete or continuous) or categorical.
    Discrete levels for continuous factors can be obtained by providing
    the minimum and maximum of the factor range and the number of
    levels. The more levels, the more fine-grained the experiment will
    evaluate this factor, but also more experimental runs are required.

The goal of optimising an experimental design is to provide an efficient
design that is near-optimal in terms of, e.g., orthogonality, level
balance, and two-way interaction coverage, yet can be performed with a
minimum number of experimental runs, which are often costly or
time-consuming.

Functionality
-------------

If you would like to jumpstart a new experiment and to skip the
technical details, you can find a summary of the main usage of DoEgen in
Case Study Use Case in the README.

Currently, the (preliminary) release contains several functions for
generating and evaluating designs. Importing and evaluating external
designs is supported (e.g. for comparison to other DoE generator tools).
DoE also implements several functions for experiment result analysis and
visualisation of parameter space.

The main functionalities are (sorted in order of typical experiment
process):

-   Reading Experiment Setup Table and Settings (Parameter Name, Levels
    for each factor, Maximum number of runs, Min/Max etc)
-   Generating optimised design arrays for a range of runs (given
    maximum number of runs, and optional computation-time constrains,
    see `settings_design.yaml`).
-   Evaluation and visualisation of more than ten design efficiencies
    such as level balance, orthogonality, D-efficiencies etc (see
    [Design Efficiencies](#design-efficiencies) for the complete list).
-   Automatic suggestion of minimum, optimal, and best designs within a
    given range of experiment runs.
-   Import and evaluation of externally generated design arrays.
-   Experiment result analysis: Template table for experiment results,
    multi-variant RMSE computation, best model/parameter selection,
    Factor Importance computation, pairwise response surface and
    correlation computation, factor correlation analysis and Two-way
    interaction response plots.
-   Visualisation of experiment results.

Installation And Requirements
-----------------------------

### Requirements

-   Python >= 3.6
-   OApackage
-   xlrd
-   XlsxWriter
-   Numpy
-   Pandas
-   PyYAML
-   scikit_learn
-   matplotlib
-   seaborn

The DoEgen package is currently considered experimental and has been
tested with the libraries specified in `requirements.txt`.

Installation instructions and documentation for OApackage (tested with
OApackage 2.6.6) can be found at https://pypi.org/project/OApackage/ or
can be installed with

pip install OAPackage

Please see for more details the README.
"""

__version__ = "0.4.6"
__author__ = "Sebastian Haan"