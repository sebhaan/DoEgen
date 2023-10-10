#!/bin/env python
"""
Package for creation and evaluation of experiment design for any mixed-level exp design setup.
Designs are evaluated in regard to more than 10 design efficiency criteria.
This package also includes a function (eval_extarray) that allows to import externally created designs
, e.g. with SAS tools, and to evaluate their design efficiencies.

Author: Sebastian Haan
Affiliation: Sydney Informatics Hub (SIH), The University of Sydney
License: LGPL-3.0

Tested with Python 3.7, see requirements.txt
"""

import argparse
import os
import sys
from pathlib import Path
import time
from contextlib import redirect_stdout
from collections import namedtuple
import numpy as np
import pandas as pd
import itertools
#import tabulate
#import xlrd
import oapackage
from multiprocessing import Pool 
from functools import partial

# for oapckage installation and docs see https://oapackage.readthedocs.io/en/latest/oapackage.html
# from sklearn.cross_decomposition import CCA ## redundant?

# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", color_codes=True)
import yaml


def gen_highD(setup, arrayclass, nkeep=2, printopt=True, outpath=None):
    """
	(In work) Extends arrays and filter based on weighted D-efficiency
	Works computationally only for small arrays and small number of runs.
	EXPERIMENTAL. NOT DEFAULT OPTMIMZATION METHOD.
	
	INPUT
	:param setup: ExperimentalSetup
	:param arrayclass: arrayclass
	:param nkeep: Number of designs to keep at each stage
	:param printout: if True prints array stats

	RETURN
	Design array and efficiencies
	"""
    alpha = [5, 5, 15]
    if printopt:
        print(arrayclass)
    arraylist = [arrayclass.create_root()]
    options = oapackage.OAextend()
    options.setAlgorithmAuto(arrayclass)
    for extension_column in range(2, setup.number_of_factors):
        print(
            f"extend {len(arraylist)} arrays with {arraylist[0].n_columns} columns with a single column"
        )
        arraylist_extensions = oapackage.extend_arraylist(
            arraylist, arrayclass, options
        )
        if arraylist_extensions:
            # Select the best arrays based on the D-efficiency
            design_efficiencies = np.array(
                [a.Defficiencies() for a in arraylist_extensions]
            )
            design_efficiencies[~np.isfinite(design_efficiencies)] = 0.0
            # dd = np.array([a.Defficiency() for a in arraylist_extensions])
            dd = (
                np.nansum(design_efficiencies * np.asarray(alpha), axis=1)
                / np.asarray(alpha).sum()
            )
            ind = np.argsort(dd)[::-1]
            ind = ind[0:nkeep]
            selection = [arraylist_extensions[ii] for ii in ind]
            dd = dd[ind]
            if printopt:
                print(
                    "  generated %d arrays, selected %d arrays with D-efficiency %.4f to %.4f"
                    % (len(arraylist_extensions), len(ind), dd.min(), dd.max())
                )
            arraylist = selection
            success = 1
        else:
            print("Failure to find array extension")
            break
            success = 0
    # show the best array
    if success:
        Asel = selection[0]
        if printopt:
            print(
                "Generated a design in OA(%d, %d, 2^%d) with D-efficiency %.4f"
                % (Asel.n_rows, arrayclass.strength, Asel.n_columns, dd[0])
            )
            print("The array is (in transposed form):\n")
            Asel.transposed().showarraycompact()
        effs = evaluate_design2(setup, np.array(Asel), printopt=True)
        return Asel, effs
    else:
        return None, None


def test_genhighD(runsize,setup,nkeep=2, outpath=None):
    """
	For some testing of function gen_highD().

    INPUT
    :param runsize: number of runs
    :param setup: ExperimentalSetup
    :param nkeep: Number of designs to keep at each stage
    :param outpath: output directory for design files
	"""
    maxfact = np.max(setup.factor_levels)
    minfact = np.min(setup.factor_levels)
    if runsize % np.lcm(minfact, maxfact) > 0:
        print(
            "Number of Experiments (Runsize) must be lowest common multiple of factor levels!"
        )
    else:
        arrayclass = oapackage.arraydata_t(
            setup.factor_levels, runsize, 2, setup.number_of_factors
        )
        Asel, effs = gen_highD(setup, arrayclass, nkeep=nkeep, outpath=outpath)
        if outpath is not None:
            os.makedirs(outpath, exist_ok=True)
            fname = (
                "Oarray_" + str(setup.factor_levels) + "_Nrun" + str(runsize) + ".csv"
            )
            np.savetxt(os.path.join(outpath,fname), np.asarray(Asel), delimiter=",", fmt="%i")
            fname = (
                "Efficiencies_"
                + str(setup.factor_levels)
                + "_Nrun"
                + str(runsize)
                + ".csv"
            )
            np.savetxt(os.path.join(outpath, fname), effs, delimiter=",")
        return Asel, effs


def calc_twofactorbalance(setup, Array):
    """ 
	Computation of two-way interaction balance of exp design array and any missing	missing two-way intearctions
	INPUT
	setup: ExperimentalSetup
	Array: experiment design array with shape (N exp runs, M factors)

	RETURN
	lb2: Two-way interaction balance level, from zero (worst) to one (optimal)
	bl2_least1: fraction of missing combinations with minimum of one occurence
	Alevelbal: (Mfactors times Mfactors) Matrix of interaction counts
	"""
    runsize, nfac = Array.shape
    # fac_levels = [len(np.unique(Array[:, col].astype(int))) for col in range(Array.shape[1])]
    colname = ["Param_" + str(i) for i in range(nfac)]
    df = pd.DataFrame(data=Array, columns=colname)
    fac_levels = df.nunique(axis=0).values
    ncount = 0
    lb2_err = 0.0
    bl2_least1_err = 0.0
    Alevelbal = np.zeros((nfac, nfac))
    for i, j in itertools.combinations(range(nfac), 2):
        ncount += 1
        nperfect = runsize / (fac_levels[i] * fac_levels[j])
        df_group = (
            df.groupby(["Param_" + str(i), "Param_" + str(j)])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
        )
        # Be aware that this doesn't take into account combinations with no occurrence
        cvals = df_group["count"].values
        ilb2_err = abs(cvals - nperfect)
        ilb2_err[ilb2_err > nperfect] = nperfect
        # count combinations that do not occure
        nmiss = (fac_levels[i] * fac_levels[j]) - len(cvals)
        # Add all errors
        lb2_err += (ilb2_err.sum() + nperfect * nmiss) / runsize
        bl2_least1_err += nmiss / (fac_levels[i] * fac_levels[j])
        # print(df_group)
        # print('nmiss', nmiss)
        # print('lb2_err', lb2_err)
        # print('bl2_least1_err', bl2_least1_err)
    # Normalize balance calculation
    bl2_balance = 1 - lb2_err / ncount
    bl2_least1 = 1 - bl2_least1_err / ncount

    # Create two-level interaction balance pivot table
    dfcat = (
        pd.DataFrame(data=Array, columns=setup.factor_names)
        .astype("int")
        .astype("category")
    )
    # Alevelbal = pd.get_dummies(dfcat.set_index(setup.factor_names[0]), prefix_sep = ' _').sum(level=0)
    """
	Creates list that contains for each level in each factor the  number of interaction with other factors levels
	The aim is to check if pariwise combinations occur at least one and if the number of ouccurence is evenly spread
	"""
    Alist = []
    factors = sorted(setup.factor_names)
    for factor_name in factors:
        Alist.append(
            pd.get_dummies(dfcat.set_index(factor_name), prefix_sep=" _").groupby(level=0)
            .sum()
            .astype(int)
        )
       #.sum(level=0) keyword was deprecated in pandas > 2.0
#            pd.get_dummies(dfcat.set_index(factor_name), prefix_sep=" _")
#            .sum(level=0)

    # print(Alist)
    # Alevelbal = pd.concat(Alist, keys=factors, sort='True')
    return bl2_balance, bl2_least1, Alist  # Alevelbal


def normalize_array(Array):
    """
	Normalize array from -1 to 1

    INPUT
    Array: experiment design array with shape (N exp runs, M factors)
	"""
    colmax = np.max(Array, axis=0)
    colmin = np.min(Array, axis=0)
    coldelta = colmax - colmin
    colmean = (colmax + colmin) / 2.0
    if len(coldelta)-np.count_nonzero(coldelta)>0: #check for zeros if we get NaNs for coldelta it breaks something.
        print("This result causes a problem. Try increasing nrun_min=X to nrun_min+ndelta or tweaking Level Numbers in the exp. design file")
        raise ValueError("nrun_min variation or Level Number changes will fix this")
    return 2 * (Array - colmean) / coldelta


def evaluate_design2(setup, Array, printopt=False, dir_out=None, plotgrid=True):
    """
	Computes multiple characteristics of a given design array, such as 
	Level Balance, Center Balance, Orthogonality, D-, D_s-, D1-, A-Efficiencies
	Most indicators have a range from 0 (worst possible) to 1 (optimal)
	Creates also correlation plots for evaluation.

	INPUT
	setup: ExperimentalSetup
	Array: experiment design array with shape (N exp runs, M factors)
	printopt: (optional) boolean, by default output results 
	path_out: (optional) output dirtectoy name for diagnostis tables
	plotgrid: (optional) plots pairwise correlation of design as cornerplot

	RETURN
	list of 11 efficiencies (normalized from 0:worst to 1:optimal):
	Center Balance
	Level Balance
	Orthogonality
	Two-way Interaction Balance
	Two-way Interaction with at least one occurence
	D Efficiency (main term and quadratic)
	D1 Eff (only main terms)
	D2 Eff (main, quadratic, and interaction terms)
	A-Eff  (main term and quadratic)
	A1-Eff (only main terms)
	A2-Eff (main, quadratic, and interaction terms)
	If dir_out is not None: 
	Tables of corraltions, level balance and two-main interaction balance
	Plot of pairwise relationships
	"""
    runsize, number_of_factors = Array.shape
    # fac_levels = [len(np.unique(Array[:, col].astype(int))) for col in range(Array.shape[1])]
    fac_levels = [len(np.unique(col.astype(int))) for col in Array.T]
    # Normalize Array
    Anorm = normalize_array(Array)
    # Create Model Matrix X and Eff
    X, _ = create_model(Anorm, mode=2, norm=False)
    Deff = calc_Deff(X)
    Aeff = calc_Aeff(X)
    X1, _ = create_model(Anorm, mode=1, norm=False)
    D1eff = calc_Deff(X1)
    A1eff = calc_Aeff(X1)
    X2, _ = create_model(Anorm, mode=4, norm=False)
    D2eff = calc_Deff(X2)
    A2eff = calc_Aeff(X2)
    # Calculate center  balance for continous variables (Optimal is at 1)
    colsum = np.sum(Anorm, axis=0)
    centereff = 100 * (1 - np.sum(abs(colsum)) / np.sum(abs(Anorm)))
    # Calculate level balance
    sumdiff = 0.0
    for Ai, nlevel in zip(Array.T, fac_levels):
        Ai = Ai.astype(int)
        nfac = runsize / nlevel
        for j in np.unique(Ai):
            diff = abs(nfac - len(Ai[Ai == j]))
            sumdiff += np.min([abs(diff), nfac])

    # Set to one for perfect balance:
    leveleff = 100 * (1 - sumdiff / np.size(Array))
    # Pearson Correlation
    Acor_pearson = np.corrcoef(Anorm.T)
    # calculate Orthogonality :
    Aortho = np.dot(Anorm.T, Anorm)
    Aortho /= np.diagonal(Aortho)
    orthoeff = 100 * np.linalg.det(Aortho)
    ### Calculate Two-Level Interaction Balance:
    twoleveleff, twolevelmin, Alevel2bal = calc_twofactorbalance(setup, Array)
    twoleveleff, twolevelmin = 100 * twoleveleff, 100 * twolevelmin
    """
    # Calculate canonical correlation
    Acor_can = np.full((number_of_factors, number_of_factors), np.nan)
    try:
        for i, j in itertools.combinations(range(number_of_factors), 2):
            cca = CCA(n_components=1)
            Xnorm_c1, Xnorm_c2 = cca.fit_transform(
                Anorm[:, i].reshape(-1, 1).tolist(), Anorm[:, j].reshape(-1, 1).tolist()
            )
            Acor_can[i, j] = np.corrcoef(Xnorm_c1[:, 0].reshape(-1,1), Xnorm_c2)[0, 1] # seems to be a issue with numpy 1.20
        Acor_can_avg = np.nanmean(abs(Acor_can))
        Acor_can_max = np.nanmax(Acor_can)
    except:
        Acor_can_avg = np.nan
        Acor_can_max = np.nan
    """

    if printopt:
        print("Runsize: " +str(runsize) +" OrthoBal: %.2f" % orthoeff  +"  2LvlBal: %.2f" % twoleveleff)
       # print("Center Balance : %.2f" % centereff)
       # print("Level Balance : %.2f" % leveleff)
       # print("Orthogonal Balance: %.2f" % orthoeff)
       # print("Two Level interaction Balance: %.2f" % twoleveleff)
       # print("Two Level Interaction Minimum One: %.2f" % twolevelmin)
       # print("D-efficiency : %.2f" % Deff)
       # print("D1-efficiency : %.2f" % D1eff)
       # print("D2-efficiency : %.2f" % D2eff)
       # print("A-efficiency : %.2f" % Aeff)
       # print("A1-efficiency : %.2f" % A1eff)
       # print("A2-efficiency : %.2f" % A2eff)
        #print("Average Canonical Corr: %.2f" % Acor_can_avg)
        #print("Maximum Canonical Corr : %.2f" % Acor_can_max)

    # Save output daignostic tables
    if dir_out is not None:
        os.makedirs(dir_out, exist_ok=True)
        # Canonical Correlation
        """
        df_Acor_can = pd.DataFrame(
            data=np.round(Acor_can, 4),
            columns=setup.factor_names,
            index=setup.factor_names,
        )
        df_Acor_can.to_csv(dir_out + "Table_Canonical_Correlation.csv")
        """
        df_pearson = pd.DataFrame(
            data=np.round(Acor_pearson, 4),
            columns=setup.factor_names,
            index=setup.factor_names,
        )
        df_pearson.to_csv(dir_out + "Table_Pearson_Correlation.csv")
        # Alevel2bal.to_csv(dir_out + 'Table_Interaction_Balance.csv')
        with open(dir_out + "Table_Interaction_Balance.txt", "w") as outfile:
            outfile.write("\n".join(str(item) for item in Alevel2bal))
        # with open(dir_out + 'Table_Interaction_Balance.txt', 'w') as f:
        # 	for item in Alevel2bal:
        # 		f.write("%s\n" % item)
        # Make pairwise realtionship plot
        if plotgrid:
            dfarray = pd.DataFrame(Array, columns=setup.factor_names)
            # Define opaque value for scatter points so that one can identify point balance
            # Additional the plot will fit linear regression, which incdicates pariwise orthogonality
            # alphaval = 1/(np.nanmean(Alevel2bal) + 2 * np.nanstd(Alevel2bal) + 1)
            plt.ioff()  # automatic disables display of figures

            def hide_current_axis(*args, **kwds):
                # hide upper triangle
                plt.gca().set_visible(False)

            ax = sns.pairplot(
                dfarray, kind="reg", plot_kws=dict(scatter_kws={"alpha": 0.2})
            )
            ax.map_upper(hide_current_axis)
            plt.savefig(dir_out + "pairwise_correlation.png", dpi=150)
    efficiencies = (
        centereff,
        leveleff,
        orthoeff,
        twoleveleff,
        twolevelmin,
        Deff,
        D1eff,
        D2eff,
        Aeff,
        A1eff,
        A2eff,
        #Acor_can_avg,
        #Acor_can_max,
    )
    return efficiencies


def create_model(Array, mode=2, norm=True, intercept=1):
    """ Construct X matrix model from design array

    INPUT
    Array: experiment design array with shape (N exp runs, M factors)
    mode: 1: main effects only, 2: main and quadratic effects
    norm: if True, normalize array
    intercept: if True, add intercept to model

    RETURN
    X: Model matrix
    header: list of column names
    """
    nrun, nfac = Array.shape
    # Normalize Array
    if norm:
        Array = normalize_array(Array)
    # Define Model Matrix
    # Elements for two- way interaction model
    if mode == 1:
        # Model with intercept and main factor only, no quadratic or interaction terms
        p = nfac + 1
        X = np.zeros((nrun, p))
        # Set intercept to 1
        X[:, 0] = intercept
        X[:, 1:] = Array
        header_main = ["X" + str(i + 1) for i in range(nfac)]
        header = ["Intercept"] + header_main
    elif mode == 2:
        # intercept, main, and main quadratic terms w/o interaction terms
        # Same as JMP calculation
        Array_2way = []  # np.zeros((nrun, nfac * (nfac - 1) / 2))
        header_2way = []
        for i in range(nfac):
            Array_2way.append(Array[:, i] * Array[:, i])
            header_2way.append("X" + str(i + 1) + "X" + str(i + 1))
    elif mode == 3:
        # intercept, main and two-way interaction terms only
        Array_2way = []  # np.zeros((nrun, nfac * (nfac - 1) / 2))
        header_2way = []
        for i in range(nfac - 1):
            for j in range(i + 1, nfac):
                Array_2way.append(Array[:, i] * Array[:, j])
                header_2way.append("X" + str(i + 1) + "X" + str(j + 1))
    elif mode == 4:
        # intercept, main, qadratric and two way interaction terms
        Array_2way = []  # np.zeros((nrun, nfac * (nfac - 1) / 2))
        header_2way = []
        for i in range(nfac):
            for j in range(i, nfac):
                Array_2way.append(Array[:, i] * Array[:, j])
                header_2way.append("X" + str(i + 1) + "X" + str(j + 1))
    if mode > 1:
        Array_2way = np.stack(Array_2way).T
        p = 1 + nfac + Array_2way.shape[1]
        X = np.zeros((nrun, p))
        X[:, 0] = intercept
        X[:, 1 : nfac + 1] = Array
        X[:, nfac + 1 :] = Array_2way
        header_main = ["X" + str(i + 1) for i in range(nfac)]
        header = ["Intercept"] + header_main + header_2way
    return X, header


def calc_Deff(X):
    # Calculation of D-efficiency according to SAS JMP
    XX = np.dot(X.T, X)
    det = np.linalg.det(XX)
    if det > 1e-18:
        det = np.power(det, 1 / X.shape[1])
    else:
        det = 0
    return 100 / X.shape[0] * det


def calc_Aeff(X):
    # Calculation of A-efficiency according to SAS JMP
    XX = np.dot(X.T, X)
    trace = np.trace(np.linalg.pinv(XX))
    aeff = 100 * X.shape[1] / (X.shape[0] * trace)
    if aeff > 100:
        return 100
    else:
        return aeff

#setup,outpath,runtime,delta
def optimize_design(setup,outpath,runtime,delta,runsize,printopt=True,nrestarts=10,niter=None):
    """ 
	Optimizes design for given design specification and  array length (runsize)
	This optimization leverages part of the the oapackage.Doptimize package. 
	See for more details https://oapackage.readthedocs.io/en/latest/index.html
	Parameters for oapackage have been finetuned through testing various design setups.
	The oapackage returns multiple designs and the best design is selected based on 
	center balance effciency, orthogonality, and two-level balance (see function evaluate_design2)

	INPUT
	setup: ExperimentalSetup
	runsize: Number of experiments limited to a runsize of 500
	outpath_nrun: path for output directory (If None, no files are saved nor plotted)
	runtime: Maximum time for optimization (Default 100 seconds)
	printopt: (Default True) Prints status messages
	nrestart: Number of restrats for optimization (Default 10)
	niter: (Default None) Number of iterations for optimization. If None are given iterations 
	are approximated by runtime 

    RETURN
    efficiencies: list of 11 efficiencies (normalized from 0:worst to 1:optimal):
    Center Balance
    Level Balance
    Orthogonality
    Two-way Interaction Balance
    Two-way Interaction with at least one occurence
    D Efficiency (main term and quadratic)
    D1 Eff (only main terms)
    D2 Eff (main, quadratic, and interaction terms)
    A-Eff  (main term and quadratic)
    A1-Eff (only main terms)
    A2-Eff (main, quadratic, and interaction terms)
    If dir_out is not None:
    Tables of corraltions, level balance and two-main interaction balance
	"""

    runsize=int(runsize)
    outpath_nrun = os.path.join(outpath, "DesignArray_Nrun" + str(int(runsize)) + "/") 
    
    # Setting for oapackage optimisation weighting for D, Ds, D1 efficiencies:
    alpha = [5, 5, 15]
    arrayclass = oapackage.arraydata_t(
        setup.factor_levels, runsize, 0, setup.number_of_factors
    )

    # First estimate time
    # First caculate number of iterations for given time
    start_time = time.time()
    devnull = open(os.devnull, "w")
    with redirect_stdout(devnull):
        scores, design_efficiencies, designs, ngenerated = oapackage.Doptim.Doptimize(
            arrayclass, nrestarts=10, niter=100, optimfunc=alpha
        )
    delta_time1 = time.time() - start_time  # in seconds
    start_time = time.time()
    with redirect_stdout(devnull):
        scores, design_efficiencies, designs, ngenerated = oapackage.Doptim.Doptimize(
            arrayclass, nrestarts=10, niter=200, optimfunc=alpha
        )
    delta_time2 = time.time() - start_time  # in seconds
    delta_time = delta_time2 - delta_time1
    fac_time = runtime / delta_time
    # print("delta_time: " +str(delta_time))
    niter = int(100 * fac_time)
    #delta_time3 = time.time() - delta_time2  # in seconds
    #print(" Runsize: "+str(runsize)+" Niteration_: "+str(niter)+" factime: "+str(fac_time))
        
    with redirect_stdout(devnull):
        scores, design_efficiencies, designs, ngenerated = oapackage.Doptim.Doptimize(
            arrayclass,
            nrestarts=10,
            niter=niter,
            optimfunc=alpha,
            nabort=3000,
            maxtime=runtime,
        )

    print("Runsize: "+str(runsize)+" Niterations: "+str(niter)+" Doptimize_NumOut: "+str(len(designs)))
    # Evaluate D efficiencies as weighted mean over D, Ds and D1 with weights alpha
    # Deff=[np.sum(d.Defficiencies() * np.asarray(alpha) / np.asarray(alpha).sum()) for d in designs]
    # Make evaluation based on center balance, orthogonality, and two-levelbaldance
    score = []
    
    for i in range(len(designs)):
        effs = evaluate_design2(setup, np.asarray(designs[i]), printopt=False)
        # score = centereff + orthoeff + 0.5 * twoleveleff
        score.append(effs[0] + effs[2] + 0.5 * effs[3])
    best = np.argmax(score)
    # design_efficiencies[~np.isfinite(design_efficiencies)] = 0.
    # Deffm_list = np.nansum(design_efficiencies * np.asarray(alpha), axis =1) / np.asarray(alpha).sum()
    # best=np.argmax(Deffm_list)
    Asel = designs[best]
    A = np.asarray(Asel)
    # Calculate efficiencies:
    efficiencies = evaluate_design2(
        setup, np.asarray(A), dir_out=outpath_nrun, printopt=True
    )
    # Save results if outpath_nrun is not None
    if outpath_nrun is not None:
        os.makedirs(outpath_nrun, exist_ok=True)
        fname = "EDarray_" + str(setup.factor_levels) + "_Nrun" + str(runsize) + ".csv"
        np.savetxt(os.path.join(outpath_nrun, fname), A, delimiter=",", fmt="%i")
        fname = (
            "Efficiencies_" + str(setup.factor_levels) + "_Nrun" + str(runsize) + ".csv"
        )
        np.savetxt(os.path.join(outpath_nrun, fname), efficiencies, delimiter=",")
    #return Asel, efficiencies
    return efficiencies

def eval_extarray(setup, path, infname):
    """
	Evaluates any given design array (e.g. externally created by SAS) 
	and saves design efficiencies as csv file.

	INPUT:
	setup: ExperimentalSetup
	path, infname: input path and filename of array saved as csv file (without header or index)

    OUTPUT:
    Saves design efficiencies as csv file
	"""
    A = np.loadtxt(os.path.join(path, infname), delimiter=",")
    effs = evaluate_design2(setup, A)
    effs = np.round(effs, 3)
    # centereff, leveleff, orthoeff, twoleveleff, twolevelmin, Deff, D1eff, D2eff, Aeff, A1eff, A2eff = effs
    # Save efficienices as Pandas and csv
    header = [
        "Center Balance",
        "Level Balance",
        "Orthogonality",
        "Two-level Balance",
        "Two-level Min-Eff",
        "D-Eff",
        "D1-Eff",
        "D2-Eff",
        "A-Eff",
        "A1-Eff",
        "A2-Eff",
        #"Canonical-Eff",
        #"Canonical-Corr-max",
    ]
    df_eff = pd.DataFrame(data=effs.reshape(1, -1), columns=header)
    # Assuming infname for input array ends alerady with .csv :
    outfname = "Efficiencies_" + infname
    df_eff.to_csv(os.path.join(path,outfname), index=False)


class ExperimentalSetup:
    def __init__(self, factor_levels, level_values, factor_names):
        self.factor_levels = factor_levels
        self.level_values = level_values
        self.factor_names = factor_names
        self.number_of_factors = len(self.factor_levels)

    @classmethod
    def read(cls, fname_setup):
        return cls(*read_setup_new(fname_setup))


def read_setup(fname_setup):
    """
	Reading in experiment design setup file
	(assume format same as template that is created with create_setupfile.py)

	INPUT
	fname_setup: path + filename for experiment setup excel file

	RETURN
	list of number of factor levels, list of all factor levels
	"""
    df = pd.read_excel(fname_setup)
    nlevel = df["Level Number"].values
    nfactor = len(nlevel)
    factor_names = df["Parameter Name"].values
    factype = df["Parameter Type"].values
    lmin = df["Minimum"].values
    lmax = df["Maximum"].values
    level_vals = []
    for i in range(nfactor):
        if factype[i] == "Categorical":
            level = ["L" + str(k + 1) for k in range(nlevel[i])]
        elif factype[i] == "Continuous":
            if np.isfinite(lmin[i]) & np.isfinite(lmax[i]):
                level = np.linspace(lmin[i], lmax[i], nlevel[i]).tolist()
            else:
                print(
                    "Mimimum or Maximum not specified for Parameter "
                    + str(factor_names[i])
                    + ". Assuming default range from -1 to 1."
                )
        elif factype[i] == "Discrete":
            if np.isfinite(lmin[i]) & np.isfinite(lmax[i]):
                level = np.linspace(lmin[i], lmax[i], nlevel[i], dtype="int8").tolist()
            else:
                print(
                    "Mimimum or Maximum not specified for Parameter "
                    + str(factor_names[i])
                    + ". Assuming default range from -1 to 1."
                )
        else:
            print("Error: Parameter Type unknown for " + str(factor_names[i]))
        level_vals.append(level)
    return nlevel.tolist(), level_vals, factor_names


def read_setup_new(fname_setup):
    """
    Reading in experiment design setup file
    (assume format same as template that is created with create_setupfile.py)

    INPUT
    fname_setup: path + filename for experiment setup excel file

    RETURN
    list of number of factor levels, list of all factor levels
    """
    df = pd.read_excel(fname_setup, na_filter = False)
    if 'Include (Y/N)' in list(df):
        df = df[df['Include (Y/N)'] == 'Yes']
    nlevel = df["Level Number"].values
    if 'Levels' in list(df):
        levels = df["Levels"].values
    else:
        levels = [''] * len(df)
    factor_names = df["Parameter Name"].values
    nfactor = len(factor_names)
    factype = df["Parameter Type"].values
    lmin = df["Minimum"].values
    lmax = df["Maximum"].values
    level_vals = []
    for i in range(nfactor):
        if (levels[i] != '') | (',' in levels[i]):
            level = [ x.strip() for x in levels[i].split(',') ]
            check_numeric = np.asarray([ x.isnumeric() for x in level ])
            if check_numeric.all():
                try:
                    level = np.asarray(level).astype(float).tolist()
                except:
                    factype[i] == "Categorical"
            if len(level) != nlevel[i]:
                print('Levels: ', levels[i])
                print('WARNING: Number of levels ' + str(nlevel[i]) + ' is not consistent with levels given')
                levels[i] = len(level)
            level_given = True
        else:
            level_given = False
        if not level_given:
            if factype[i] == "Categorical":
                print('WARNING: Level names not given for factor ' + str(factor_names[i]))
                print('Setting level names to L1 ... L' + str(nlevel[i]))
                level = ["L" + str(k + 1) for k in range(nlevel[i])]
            elif factype[i] == "Continuous":
                if np.isfinite(lmin[i]) & np.isfinite(lmax[i]):
                    level = np.linspace(lmin[i], lmax[i], nlevel[i]).tolist()
                else:
                    print(
                        "Mimimum or Maximum not specified for Parameter "
                        + str(factor_names[i])
                        + ". Assuming default range from -1 to 1."
                    )
            elif factype[i] == "Discrete":
                if np.isfinite(lmin[i]) & np.isfinite(lmax[i]):
                    level = np.linspace(lmin[i], lmax[i], nlevel[i], dtype="int8").tolist()
                else:
                    print(
                        "Mimimum or Maximum not specified for Parameter "
                        + str(factor_names[i])
                        + ". Assuming default range from -1 to 1."
                    )
            else:
                print("Error: Parameter Type unknown for " + str(factor_names[i]))
        level_vals.append(level)
    return nlevel.tolist(), level_vals, factor_names


def array2valuetable(setup, fname_array, fname_out):
    """
	Generates experiment design table with level values from optimised experiment array.
	level values and factor names are obtained with ExperimentalSetup
	
	INPUT
	setup: ExperimentalSetup
	fname_array: input path + filename for exp design array in csv format (no header in csv)
	fname_out: output path + filename for converted experimental table with actual level values for each run

	RETURN
	Experiment design table saved as csv file
	"""
    # Read in exp design csv file:
    dfarray = pd.read_csv(fname_array, names=setup.factor_names, header=None)
    dfnew = dfarray.copy()
    # Check again that experiment design aray has the correct number of factors:
    assert dfnew.shape[1] == len(setup.factor_names)
    # replace array levels with actual values
    for i in range(len(setup.factor_names)):
        # loop over each factor
        facname = setup.factor_names[i]
        levels_array = dfarray[facname].unique()
        lvals = setup.level_values[i]
        for nl, level in enumerate(levels_array):
            # loop over each level and replace array level with level value
            dfnew.loc[dfarray[facname] == level, facname] = lvals[nl]
    # Add experiment run number
    dfnew.index += 1
    dfnew.to_csv(fname_out, index_label="Nexp")


def post_evaluate(setup, inpath, outpath, nmin, nmax, ndelta):
    """
	Evaluates design for given range of exp run numbers 
	Returns design efficiency plots and saves arrays as csv in defulat outpath directoy

	INPUT
	setup: ExperimentalSetup
	inpath: input directory name
	nmin: from number of runs to start with
	namx: to number of runs
	ndelta: run n umber interval

	RETURN:
	efficiency array
	"""
    # Need updated parameter and efficiencies calculation
    xrun = np.arange(nmin, nmax, ndelta)
    effs_array = np.zeros((len(xrun), 11))
    for i, irun, in enumerate(xrun):
        print("--------------------------------")
        print("Evaluation of array for " + str(irun) + " runs:")
        A = np.loadtxt(
            os.path.join(
                inpath,
                "EDarray_"
                + str(setup.factor_levels)
                + "_Nrun"
                + str(int(irun))
                + ".csv",
            ),
            delimiter=",",
        )
        effs = evaluate_design2(setup, A)
        effs_array[i] = effs
    header = [
        "Center Balance",
        "Level Balance",
        "Orthogonality",
        "Two-way Level Balance",
        "Two-way Min-Eff",
        "D-Eff",
        "D1-Eff",
        "D2-Eff",
        "A-Eff",
        "A1-Eff",
        "A2-Eff",
        #"Canonical-Eff",
        #"Canonical-Corr-max",
    ]
    df_eff = pd.DataFrame(data=effs_array, columns=header)
    dfout = np.round(df_eff, 3)
    dfout["Nexp"] = xrun
    fname = "Efficiencies_" + str(setup.factor_levels) + "_combined.csv"
    dfout.to_csv(os.path.join(outpath, fname), index=False)
    plt.ioff()
#sort values instead of columns
    dfout=dfout.sort_values(by=['Nexp']) #replaces the sort_columns bit in the plot part of this which is deprecated in pandas 2
    dfout.plot(
        "Nexp",
        [
            "Center Balance",
            "Level Balance",
            "Orthogonality",
            "Two-way Level Balance",
            "Two-way Min-Eff",
            "D-Eff",
            "D1-Eff",
        ],
#       deprecated 2.0
#        sort_columns=True,
    )
    plt.savefig(os.path.join(outpath, "Efficiencies_" + str(setup.factor_levels) + ".png"), dpi=300)
    plt.close()
    return effs_array


def print_designselection_summary(results, fname_out=None):
    """
	print summary evalutaion of three selected designs
	fout: output path + filename. If None prints to sys.stdout (Default)
	"""
    if fname_out is None:
        fout = sys.stdout
        closing = False
    else:
        fout = open(fname_out, "a")
        closing = True

    print("", file=fout)
    print("RESULTS OVERVIEW:", file=fout)
    print("--------------------------------", file=fout)
    for result in results.values():
        print(
            result.name.title() + " Exp Design Runsize: " + str(result.runsize),
            file=fout,
        )
    print("--------------------------------", file=fout)
    print("", file=fout)
    print("", file=fout)
    print("Efficiencies: ", file=fout)
    print(
        "------------------------------------------------------------------------------",
        file=fout,
    )
    columns = [
        result.name.title() + " Design" for result in results.values()
    ]  # ["Min Design", "Opt Design", "Best Desig"]
    names = [
        "Center Balance",
        "Level Balance",
        "Orthogonality",
        "Two-Way Interact Bal",
        "D Efficieny",
        "D1 Efficieny",
    ]
    dfsummary = pd.DataFrame(
        np.round(
            np.array([result.effs[[0, 1, 2, 3, 5, 6]] for result in results.values()]),
            3,
        ).T,
        index=names,
        columns=columns,
    )
    print(dfsummary.to_markdown(), file=fout)
    if closing:
        fout.close()


def main(
    fname_setup,
    path=None,
    outpath=None,
    nrun_max=150,
    maxtime_per_run=100,
    delta_nrun=None,
    nrun_min=None
):
    if nrun_max > 500:
        print("nrun_max of > 500 : OApackage < 2.7.7 does not support a runsize of > 500")
      
    if outpath is None:
        outpath = path = Path(path)
    else:
        outpath = Path(outpath)

    # Read-in experimental design specifications:
    setup = ExperimentalSetup.read(os.path.join(path,fname_setup))
    # print(setup.level_values)
    print('setup:', setup)

    # Calculate number of total possible combinations of all factors:
    Ncombinations_total = np.product(np.asarray(setup.factor_levels))
    print("Number of total combinations (Full Factorial):", Ncombinations_total)

    ### Make designs for multiple run sizes:
    maxfact = np.max(setup.factor_levels)
    minfact = np.min(setup.factor_levels)
    # Set run number by factor of lowest common multiple
    if (delta_nrun is None) | (delta_nrun=='None'):
        ndelta = np.lcm(minfact, maxfact)
    else:
        ndelta=delta_nrun
    # Find mimimum number of runs if not specified in input:
    if nrun_min==None:
        nrun, nrun_min = 0, 0
        while nrun_min == 0:
            nrun += ndelta
            if nrun >= setup.number_of_factors + 1:
                nrun_min = nrun

    #Just set the nrun as the min 
    else: 
        nrun=nrun_min
    #Print the starting runsize and increment 
    print ("starting numb of exp. nrun_min: "+str(nrun_min))    
    print ("increment numb of exp. by delta_nrun: "+str(ndelta))
    
    # Generate optimised design array and calculate efficiencies for each runsize in range of:
    xrun = np.arange(nrun_min, nrun_max, ndelta)
    # Total run time estimate given 100s per run
    minutes = np.round(len(xrun) * 100 / 60, 2)/os.cpu_count() #divide by number of parallel processes
    # seconds = np.round((60 * (len(xrun) * 100 / 60 - minutes)))
    if minutes < 3:
        print("Hold down your bagel.")
    elif (minutes >= 3) & (minutes < 15):
        print("Perfect time for a cup of tea.")
    elif (minutes >= 15) & (minutes < 180):
        print("Sit back and relax. This may take a while.")
    elif minutes >= 180:
        print("You may want to consider a smaller runsize.")
    print("Total estimated runtime:  " + str(round(minutes,2)) + " minutes")
    # xrun = np.arange(201,300,ndelta)
    effs_array = np.zeros((len(xrun), 11))
    
    #Replace the previous for loop with a multiprocessing alternative.
    multi_effs = optimize_design_multi(setup, xrun, outpath, maxtime_per_run, ndelta)

    #effs_array=multi_effs
    #convert the output from multiproc to the expected array format.
    for i in range(0,len(xrun)):
    #    effs_array[i]=multi_effs.get()[i] #pool.map_async output change to expected array, async returns unordered results
        effs_array[i]=multi_effs[i]        #pool.map output change to expected array, map should* return ordered results
 
    #print(effs_array)
    print("-------------------------------------------")
    print("Finished optimising all possible run sizes.")

    print("Saving efficiencies and design arrays as csv files....")
    # Save efficienices as Pandas and csv
    # header = ['Center Balance', 'Level Balance', 'Orthogonality', 'Two-level Balance', 'Two-level Min-Eff',  'avgD-Eff', 'D-Eff', 'Ds-Eff', 'D1-Eff', 'A-Eff', 'E-Eff']
    header = [
        "Center Balance",
        "Level Balance",
        "Orthogonality",
        "Two-level Balance",
        "Two-level Min-Eff",
        "D-Eff",
        "D1-Eff",
        "D2-Eff",
        "A-Eff",
        "A1-Eff",
        "A2-Eff",
        #"Canonical-Eff",
        #"Canonical-Corr-max",
    ]
    df_eff = pd.DataFrame(data=effs_array, columns=header)
    dfout = np.round(df_eff, 3)
    dfout["Nexp"] = xrun
    fname = "Efficiencies_" + str(setup.factor_levels) + "_all3.csv"
    dfout.to_csv(os.path.join(outpath, fname), index=False)

    # dfout.plot("Nexp", ['Center Balance', 'Level Balance', 'Orthogonality', 'D-Eff', 'D1-Eff', 'A-Eff', 'E-Eff'], sort_columns = True)
#   pandas.df.plot(sort_columns) is deprecated so sort the columns before plotting. Overwriting this is probably terrible.
    dfout=dfout.sort_values(by=['Nexp'])
    dfout.plot(
        "Nexp",
        [
            "Center Balance",
            "Level Balance",
            "Orthogonality",
            "Two-level Balance",
            "Two-level Min-Eff",
            "D1-Eff",
        ],
#        sort_columns=True, #deprecated in pandas 2.
    )
    plt.ioff()
    plt.savefig(os.path.join(outpath, "Efficiencies_" + str(setup.factor_levels) + ".png"), dpi=300)
    plt.close()
    # seems to be correlation  between orthogonality(levelbvalance) and D1 and
    # plt.scatter(effs_array[:,1], effs_array[:,5])

    ###### Identify minimum, optimal, and best runsize for experiment:
    Result = namedtuple("Result", ["name", "runsize", "effs"])
    results = {}
    """Minimum run criteria:
	runsize > Nfactor + 1
	center balance > 95
	level balance > 95
	Orthogonal Balance > 90 
	Two Level interaction Balance > 90
	Two Level Interaction Minimum One = 100
	"""
    print("Finding minimum, optimal and best designs...")
    effs_minsel = np.where(
        (effs_array[:, 0] >= 95)
        & (effs_array[:, 1] >= 95)
        & (effs_array[:, 2] >= 90)
        & (effs_array[:, 3] >= 90)
        & (effs_array[:, 4] == 100)
    )[0]
    if len(effs_minsel) > 0:
        ix_minsel = effs_minsel[0]
        results["min"] = Result("minimum", xrun[ix_minsel], effs_array[ix_minsel])

    """Optimal run criteria:
	center balance > 98
	level balance > 98
	Orthogonal Balance > 95 
	Two Level interaction Balance > 95
	Two Level Interaction Minimum One = 100
	Best score = center balance + Orthogonal Balance + Two Level interaction Balance + mean(D eff, D1 eff) - 4/run_min * nruns
	The penalty score is 4percent/minimum_runsize per additional run. 
	That means every time an additional minimum runsize is added, the score must be at least by 1 percent better
	on the average efficiency (center, orthogonality, two-way level balance, D efficiency)
	"""
    effs_optsel = np.where(
        (effs_array[:, 0] >= 98)
        & (effs_array[:, 1] >= 98)
        & (effs_array[:, 2] >= 95)
        & (effs_array[:, 3] >= 95)
        & (effs_array[:, 4] == 100)
    )[0]
    runs_sel = xrun[effs_optsel]
    if len(effs_optsel) > 0:
        score = (
            effs_array[effs_optsel, 0]
            + effs_array[effs_optsel, 2]
            + effs_array[effs_optsel, 3]
            + 0.5 * effs_array[effs_optsel, 5]
            + 0.5 * effs_array[effs_optsel, 6]
            - 4 / results["min"].runsize * runs_sel
        )
        irun_optsel = np.argmax(score)
        run_optsel = runs_sel[irun_optsel]
        ixopt = np.where(xrun == run_optsel)
        results["opt"] = Result("optimal", run_optsel, effs_array[ixopt].flatten())

    """
	Best run:
	score based on sum of efficiencies and includes a small penalty for runsize relative to maximum runsize
	"""
    bestscore = (
        effs_array[:, 0]
        + effs_array[:, 2]
        + effs_array[:, 3]
        + 100 * (effs_array[:, 4] - 100)
        + 0.5 * effs_array[:, 5]
        + 0.5 * effs_array[:, 6]
        - 1 / nrun_max * nrun
    )
    irun_bestsel = np.argmax(bestscore)
    run_bestsel = xrun[irun_bestsel]
    ixbest = np.where(xrun == run_bestsel)
    results["best"] = Result("best", run_bestsel, effs_array[ixbest].flatten())

    # Convert exp arrays into design tables with user level values
    print("Saving minimum, optimal, and best design as experiment design tables...")
    dforig = pd.read_excel(os.path.join(path,fname_setup), na_filter = False)
    append_orig = False
    if 'Include (Y/N)' in list(dforig):
        dforig = dforig[dforig['Include (Y/N)'] == 'No'].copy()
        if len(dforig) > 0:
            append_orig = True
    for result in results.values():
        fname_array = (os.path.join(outpath,
            "DesignArray_Nrun"
            + str(result.runsize)
            + "/"
            + "EDarray_"
            + str(setup.factor_levels)
            + "_Nrun"
            + str(result.runsize)
            + ".csv")
        )
        fname_out = (os.path.join(outpath,
            "Designtable_"
            + result.name
            + "_Nrun"
            + str(result.runsize)
            + ".csv")
        )
        array2valuetable(setup, fname_array, fname_out)
        # Append original factors that are not included in design variation ('Include (Y/N)' = No)   
        if append_orig:
            dfnew = pd.read_csv(fname_out) 
            names_const = dforig[dforig['Include (Y/N)'] == 'No']['Parameter Name'].values
            level_const = dforig[dforig['Include (Y/N)'] == 'No']['Levels'].values
            for i in range(len(names_const)):
                dfnew[names_const[i]] = level_const[i] 
            dfnew.to_csv(fname_out, index = False)

    # print summary in terminal:
    print_designselection_summary(results)
    # and in addition print summary to file:
    print_designselection_summary(
        results, fname_out=os.path.join(outpath, "Experiment_Design_selection_summary.txt")
    )
    print("")
    print("FINISHED")


### Possible add-on later: subsequent longer optimisation for the three final designs: min, opt, and best


# Generate a full factorial design table
def full_factorial_design(fname_setup,outfile='full_factorial_design_table.csv'):
    #use doegen functions to read in and initialise the design table / pandas thing from the excel input file.
    setup = ExperimentalSetup.read(fname_setup)
    design=read_setup_new(fname_setup)
    design_levels={}
    # how many experiments are there in a full factorial design? print this.
    DoE_setup=ExperimentalSetup.read(fname_setup) 
    Ncombinations_total = np.product(np.asarray(DoE_setup.factor_levels)) 
    print("Generate Full Factorial experiment design table: ",Ncombinations_total," experiments")

    #create and fill in the design table with the variable parameters. These are the ones flagged as "Yes" in the input excel file.
    for j in range(0,len(design[2])):
        design_levels[design[2][j]] = design[1][j]
    ffact=[]
    for item in itertools.product(*design_levels.values()):
        ffact.append(item)
    cols=list(design_levels.keys())
    df = pd.DataFrame(ffact,columns=cols)
    df.index += 1
    
    #iterate through the non-varying parameters (flagged them as "No" in the excel input file) and append those to the columns to the design table.
    dforig=pd.read_excel(fname_setup)
    level_const = dforig[dforig['Include (Y/N)'] == 'No']['Levels'].values
    names_const = dforig[dforig['Include (Y/N)'] == 'No']['Parameter Name'].values
    for i in range(len(names_const)):
        df[names_const[i]] = level_const[i]
    df.to_csv(outfile, index_label="Nexp") #save to the specified or default output design table csv file.
    return(setup)


# Multiprocessing replacement for main optimization loop.
def optimize_design_multi(setup, runsize, outpath, runtime, delta):
    proc = os.cpu_count()
    with Pool(processes = proc) as p:
        print('start multiproc')
        start = time.time()
        ordered_result = p.map(partial(optimize_design,setup,outpath,runtime,delta),runsize) #the multiproc start bit
        p.close()
        p.join()
        print('Simulations completed; total processing time: ' + str(round((time.time() - start)/60, 2)) + ' minutes')
        return ordered_result
  
def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("settings_path", nargs="?", default="settings_design.yaml")
    args = ap.parse_args()
    print(f"using settings in: {args.settings_path!r}")
    with open(args.settings_path) as f:
        cfg = yaml.safe_load(f)
    main(**cfg)


if __name__ == "__main__":
    main_cli()
