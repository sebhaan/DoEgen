"""
Package to evaluate the response and factor effectiveness of experiment results.

Author: Sebastian Haan
Affiliation: Sydney Informatics Hub (SIH), THe University of Sydney
Version: Experimental
License: APGL-3.0

Tested with Python 3.7

Main Capabilities:
- Multi-variant RMSE computation and best model/parameter selection
- Factor Importance computation
- Pairwise response surface and correlation computation
- Factor correlation analysis and Two -way intearction response plots
- Visualisation plots

ToDo:
Change to Pathlib

Changes to previous version:
- replace configloader with function arguments
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# Load settings parameters


def merge_expresults(fname_result, fname_design, y_label=None):
    """
	Reads experiment results into pandas dataframe 
	and merges with paramater file

	INPUT
	fname_result: path + filenmae of experimental results (see excel template)
	fname_design: path + filenmae of experimental design setip (see excel template)
	y_label: (Default None) Column name for precited y property
	"""
    dfres = pd.read_excel(fname_result)
    dfdes = pd.read_excel(fname_design)
    if y_label is not None:
        dfres = dfres[dfres["Y Label"] == y_label]
    # Merge two files:
    dfcomb = dfres.merge(dfdes, on="Nexp", how="left")
    return dfcomb


def create_testdata(outpath, fname_out, Nexp):
    """
	# Script for creating random set of results for testing
	
	INPUT
	outpath: output directory
	fname_out: filename in format '*.xlsx'
	Nexp: NUmber of experiments
	"""
    os.makedirs(outpath, exist_ok=True)
    PID = np.arange(1, 11)
    Yexp = np.random.rand(Nexp, len(PID)).flatten()
    Ytruth = np.random.rand(Nexp, len(PID)).flatten()
    Ylabel = np.ones(len(Yexp))
    aNexp = np.zeros_like(Yexp)
    aPID = np.zeros_like(Yexp)
    for i in range(Nexp):
        aNexp[10 * i : 10 * i + 10] = i + 1
        aPID[10 * i : 10 * i + 10] = PID
    array = np.vstack(
        (
            aNexp,
            aPID,
            Ylabel,
            Yexp,
            Ytruth,
            Ytruth * np.nan,
            Ytruth * np.nan,
            Ytruth * np.nan,
        )
    )
    header = [
        "Nexp",
        "PID",
        "Y Label",
        "Y Exp",
        "Y Truth",
        "Std Y Exp",
        "Std Y Truth",
        "Weight PID",
    ]
    df = pd.DataFrame(array.T, columns=header)
    df.to_excel(os.path.join(outpath,fname_out), index=False)


def weighted_avg_and_std(values, weights):
    """
    Returns weighted average and standard deviation.

    INPUT
    values, weights -- arrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def calc_expresults_stats(ylabels, dfdes, dfres, outpath):
    """
	Computation of statistical evaluation of experimetal results for each predicted y:
	1) Parameter importance, which is defined by maximum y range over parameter levels (y in averaged for each level)
	Results are visualized in bar plot and saved as csv, including, min, max, std devioation across all levels
	2) Computes RMSE and saves results as csv
	3) Computes list of top experiments and their parameters
	4) Computes average and variance of best parameters weighted with RMSE; saved to csv file

	INPUT
	ylabels: label ID for each target variable.
	dfdes: experiment design dataframe (inlcudes one column Nexp and the other columns the factor names)
	dfres: experiment result dataframe (Ids in Nexp column must match design array)
    outpath: path for output files
	"""
    npar = len(list(dfdes)) - 1
    nexp = len(dfdes)
    params = list(dfdes)[1:]
    for ylabel in ylabels:
        dfdes_y = dfdes.copy()
        # Initialise array for factor results
        ymin_par = np.full(npar, np.nan)
        ymax_par = np.zeros(npar) * np.nan
        ystd_par = np.zeros(npar) * np.nan
        ymean_par = np.zeros(npar) * np.nan
        # Select Y data and to dfdes overall stats in dfdes
        ydf = dfres[dfres["Y Label"] == ylabel].copy()
        ymean = ydf.fillna(0).groupby("Nexp")["Y Exp"].mean()
        ystd = ydf.fillna(0).groupby("Nexp")["Y Exp"].std()
        ytruemean = ydf.fillna(0).groupby("Nexp")["Y Truth"].mean()
        ytruestd = ydf.fillna(0).groupby("Nexp")["Y Truth"].std()
        assert len(ymean) == dfdes_y.shape[0]
        dfdes_y["Y Exp Mean"] = ymean.values
        dfdes_y["Y Exp Std"] = ystd.values
        dfdes_y["Y Truth Mean"] = ytruemean.values
        dfdes_y["Y Truth Std"] = ytruestd.values
        # Loop over parameter to caluclate factor range, min, max, mean and stddev:
        for i, param in enumerate(params):
            levels = dfdes[param].unique()
            ylevel = [
                np.nanmean(dfdes_y.loc[dfdes[param] == level, "Y Exp Mean"])
                for level in levels
            ]
            ylevelstd = [
                np.nanmean(dfdes_y.loc[dfdes[param] == level, "Y Exp Std"])
                for level in levels
            ]
            ymin_par[i] = np.nanmin(ylevel)
            ymax_par[i] = np.nanmax(ylevel)
            ystd_par[i] = np.nanstd(ylevel)
            ymean_par[i] = np.nanmean(ylevel)
        ypos = np.arange(npar)
        width = ymax_par - ymin_par
        sort = np.argsort(width)
        # Plot factor importance as barplot
        plt.ioff()
        plt.figure(figsize=(8, 5))
        plt.barh(
            ypos,
            width=width[sort],
            left=ymin_par[sort],
            tick_label=np.asarray(params)[sort],
            color="red",
        )
        plt.title("Range " + str(ylabel))
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, "Ybarplot_" + str(ylabel) + ".png"), dpi=300)
        plt.close()
        # Save factor importance to csv:
        res = np.vstack((width, ymin_par, ymax_par, ymean_par, ystd_par))
        dfrange = pd.DataFrame(
            res.T, columns=["Yrange", "Ymin", "Ymax", "Ymean", "Ystd"], index=params
        )
        dfrange.to_csv(
            os.path.join(outpath, "Experiment_" + str(ylabel) + "_Factorimportance.csv")
        )

        # Calculate RMSE and best parameter space:
        if ydf["Y Truth"].notnull().any():
            rmse = np.zeros(nexp)
            ytrue = np.zeros(nexp)
            for i in range(nexp):
                resid = (
                    ydf.loc[ydf["Nexp"] == i + 1, "Y Exp"]
                    - ydf.loc[ydf["Nexp"] == i + 1, "Y Truth"]
                )
                rmse[i] = np.sqrt(np.nanmean(resid ** 2))
            dfdes_y["RMSE"] = rmse
            # Save overall results to csv with sorted RMSE
            dfdes_y.to_csv(os.path.join(outpath, "Experiment_" + str(ylabel) + "_RMSE.csv"))

            # Calculate best parameters (for only nueric parameters)
            if nexp >= 20:
                nsel = 10
            elif (nsel >= 10) & (nsel < 20):
                nsel = 5
            else:
                nsel = 3
            dfsort = dfdes_y.sort_values(["RMSE"], ascending=True)
            print(
                "Top "
                + str(nsel)
                + " experiments with best RMSE for "
                + str(ylabel)
                + " :"
            )
            print(dfsort.head(nsel))
            dfsort.iloc[0:nsel].to_csv(
                os.path.join(outpath,
                "Experiment_"
                + str(ylabel)
                + "_RMSE_Top"
                + str(nsel)
                + "_sorted.csv")
            )
            """ takingh out best parameter weighting since averaging might be misleading
            # best parameter space is based on weighted RMSE of top results
            # Note that these are average parameter estimaets and are not considering multi-modal distributions
            # For multi-modal see list of top resulst
            # Select only numeric parameters
            params_num = dfsort[params]._get_numeric_data().columns
            param_wmean = np.zeros(len(params_num))  # weigthed mean
            param_wstd = np.zeros(len(params_num))  # weighted std
            dfsel = dfsort.iloc[0:nsel]
            for i, param in enumerate(params_num):
                param_wmean[i], param_wstd[i] = weighted_avg_and_std(
                    dfsel[param].values, 1 / (dfsel["RMSE"].values ** 2)
                )
            params_stats = np.vstack((param_wmean, param_wstd))
            # Save to csv
            dfparam_avg = pd.DataFrame(
                params_stats.T,
                index=params_num,
                columns=["Weighted Average", "Weigted Stddev"],
            )
            dfparam_avg.to_csv(
                cfg.outpath + "Experiment_" + str(ylabel) + "_Best-Parameter-Avg.csv"
            )
            # plot dataframe table
            plot_table(dfparam_avg, cfg.outpath, "BestFactor_Avg_" + str(ylabel) + ".png")
            """
            """
            dfparam_avg.plot(kind="bar", y="Weighted Average", yerr="Weigted Stddev")
            plt.tight_layout()
            """


# Make 3d correlation plot with heatmap
# (Make 3d scatter to image plot (works only for continous))
def plot_3dmap(df, params, target_name, fname_out):
    """
	Plots Y value or RMSE as function of two differnt X variates for each pairwise combination of factors
	The plot is using a gridded heatmap which enablesto visualise also categorical factors 
	and not just numerical data

	INPUT
	df: pandas dataframe
	params: list of factor names
    target_name: 'Y Exp Mean' or 'RMSE'
	dfname_out: output path + filename for image

	OUTPUT
	Cornerplot of Y-PairwiseCorrelation Images 
	"""
    print('Plotting Y as function of pairwise covariates ...')
    nfac = len(params)
    # Check first for max and min value
    ymin0 = df[target_name].max()
    ymax0 = df[target_name].min()
    for i in range(nfac - 1):
        for j in range(i + 1, nfac):
            table = pd.pivot_table(
                df,
                values=target_name,
                index=[params[j]],
                columns=[params[i]],
                aggfunc=np.nanmean,
            )
            if np.min(table.min()) < ymin0:
                ymin0 = np.min(table.min())
            if np.max(table.max()) > ymax0:
                ymax0 = np.max(table.max())
    # Make corner plot
    # sns.set_style("whitegrid")
    plt.ioff()  # automatic disables display of figures
    # fig, axs = plt.subplots(nfac-1, nfac-1, sharex=True, sharey=True, figsize=(nfac*2, nfac*2))
    fig, axs = plt.subplots(nfac - 1, nfac - 1, figsize=(nfac * 2, nfac * 2))
    for i in range(nfac - 1):
        for j in range(i + 1, nfac):
            table = pd.pivot_table(
                df,
                values=target_name,
                index=[params[j]],
                columns=[params[i]],
                aggfunc=np.nanmean,
            )
            g = sns.heatmap(
                table,
                cmap="viridis",
                annot=False,
                ax=axs[j - 1, i],
                vmin=ymin0,
                vmax=ymax0,
                square=True,
                cbar=False,
            )
            if i > 0:
                g.set_ylabel("")
                g.set(yticklabels=[])
            if j < nfac - 1:
                g.set_xlabel("")
                g.set(xticklabels=[])
    # remove remaining plots:
    for i in range(1, nfac - 1):
        for j in range(1, i + 1):
            g = sns.heatmap(
                table * np.nan,
                cmap="viridis",
                annot=False,
                ax=axs[j - 1, i],
                vmin=ymin0,
                vmax=ymax0,
                square=True,
                cbar=False,
            )
            g.set_ylabel("")
            g.set(yticklabels=[])
            g.set_xlabel("")
            g.set(xticklabels=[])
    # Make colorbar
    g = sns.heatmap(
        table * np.nan,
        cmap="viridis",
        annot=False,
        ax=axs[0, 1],
        vmin=ymin0,
        vmax=ymax0,
        square=True,
        cbar=True,
    )
    g.set_xlabel("")
    g.set_ylabel("")
    g.set(yticklabels=[])
    g.set(xticklabels=[])
    fig.suptitle("Pair-Variate Plot with " + target_name)
    plt.savefig(fname_out, dpi=300)


def plot_3dmap_rmse(df, params,  fname_out):
    """
    Plots RMSE value as function of two differnt X variates for each pairwise combination of factors
    The plot is using a gridded heatmap which enablesto visualise also categorical factors 
    and not just numerical data

    INPUT
    df: pandas dataframe
    params: list of factor names
    target_name: 
    dfname_out: output path + filename for image

    OUTPUT
    Cornerplot of Y-PairwiseCorrelation Images 
    """
    print('Plotting RMSE as function of pairwise covariates ...')
    nfac = len(params)
    # Check first for max and min value
    ymin0 = df["RMSE"].max()
    ymax0 = df["RMSE"].min()
    for i in range(nfac - 1):
        for j in range(i + 1, nfac):
            table = pd.pivot_table(
                df,
                values="RMSE",
                index=[params[j]],
                columns=[params[i]],
                aggfunc=np.nanmean,
            )
            if np.min(table.min()) < ymin0:
                ymin0 = np.min(table.min())
            if np.max(table.max()) > ymax0:
                ymax0 = np.max(table.max())
    # Make corner plot
    # sns.set_style("whitegrid")
    plt.ioff()  # automatic disables display of figures
    # fig, axs = plt.subplots(nfac-1, nfac-1, sharex=True, sharey=True, figsize=(nfac*2, nfac*2))
    fig, axs = plt.subplots(nfac - 1, nfac - 1, figsize=(nfac * 2, nfac * 2))
    for i in range(nfac - 1):
        for j in range(i + 1, nfac):
            table = pd.pivot_table(
                df,
                values="RMSE",
                index=[params[j]],
                columns=[params[i]],
                aggfunc=np.nanmean,
            )
            g = sns.heatmap(
                table,
                cmap="viridis",
                annot=False,
                ax=axs[j - 1, i],
                vmin=ymin0,
                vmax=ymax0,
                square=True,
                cbar=False,
            )
            if i > 0:
                g.set_ylabel("")
                g.set(yticklabels=[])
            if j < nfac - 1:
                g.set_xlabel("")
                g.set(xticklabels=[])
    # remove remaining plots:
    for i in range(1, nfac - 1):
        for j in range(1, i + 1):
            g = sns.heatmap(
                table * np.nan,
                cmap="viridis",
                annot=False,
                ax=axs[j - 1, i],
                vmin=ymin0,
                vmax=ymax0,
                square=True,
                cbar=False,
            )
            g.set_ylabel("")
            g.set(yticklabels=[])
            g.set_xlabel("")
            g.set(xticklabels=[])
    # Make colorbar
    g = sns.heatmap(
        table * np.nan,
        cmap="viridis",
        annot=False,
        ax=axs[0, 1],
        vmin=ymin0,
        vmax=ymax0,
        square=True,
        cbar=True,
    )
    g.set_xlabel("")
    g.set_ylabel("")
    g.set(yticklabels=[])
    g.set(xticklabels=[])
    fig.suptitle("Pair-Variate Plot for RMSE Function")
    plt.savefig(fname_out, dpi=300)


def plot_regression(df, params, target_name, fname_out):
    """
	Creates Correlation plot with Y or RMSE for each numeric Variate
	Note that only numeric data is selected for this plot

	INPUT
	df: dataframe
	params: list of factor names
    target_name: 'Y Exp Mean' or 'RMSE'

	OUTPUT
	Image with Correlations
	"""
    # Select numeric variates:
    columns = df[params]._get_numeric_data().columns
    nfac = len(columns)
    nax1 = int(np.sqrt(nfac))
    nax2 = int(np.ceil(nfac / int(np.sqrt(nfac))))
    # fig, axs = plt.subplots(nax1, nax2, figsize=(nax1 * 3, nax2 * 3))
    plt.ioff()  # automatic disables display of figures
    fig = plt.figure(figsize=(nax1 * 5, nax2 * 4))
    for i in range(nfac):
        r = df[columns[i]].corr(df[target_name])
        plt.subplot(nax2, nax1, i + 1)
        # sns.lmplot( x = columns[0], y = 'Y Exp Mean', data = df)
        ax = sns.regplot(x=columns[i], y=target_name, data=df)
        ax.annotate("r = {:.3f}".format(r), xy=(0.1, 0.9), xycoords=ax.transAxes)
        plt.savefig(fname_out, dpi=300)

def plot_factordis(df, params, target_name, fname_out):
    """
    Creates distribution plot of Y or RMSE for each numeric Variate
    Note that only numeric data is selected for this plot

    INPUT
    df: dataframe
    params: list of factor names
    target_name: 'Y Exp Mean' or 'RMSE'

    OUTPUT
    Image with Correlations
    """
    # Select numeric variates:
    columns = df[params]._get_numeric_data().columns
    nfac = len(columns)
    nax1 = int(np.sqrt(nfac))
    nax2 = int(np.ceil(nfac / int(np.sqrt(nfac))))
    # fig, axs = plt.subplots(nax1, nax2, figsize=(nax1 * 3, nax2 * 3))
    plt.ioff()  # automatic disables display of figures
    fig = plt.figure(figsize=(nax1 * 5, nax2 * 4))
    for i in range(nfac):
        plt.subplot(nax2, nax1, i + 1)
        ax = sns.violinplot(y=df[target_name], x=df[columns[i]])
        plt.savefig(fname_out, dpi=300)

def plot_table(df_table, outpath, fname_out):
    """
    Plot Dataframe as formatted table

    INPUT
    df_table: dataframe
    outpath: output path
    fname_out: image output filename
    """
    # Format table

    #df_table.to_string(float_format=lambda x: '%.3f' % x)
    plt.ioff()
    plt.figure(linewidth=2,
           tight_layout={'pad':40},
           figsize=(7,4)
          )
    # Set colors for row and column headers
    rcolors = plt.cm.BuPu(np.full(len(df_table), 0.15))
    ccolors = plt.cm.BuPu(np.full(len(list(df_table)), 0.15))

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    table = pd.plotting.table(ax, df_table, loc = 'center', 
        rowLoc='left',
        colLoc = 'center',
        rowColours=rcolors,
        colColours=ccolors)
    table.scale(0.6, 1.3)
    table.set_fontsize(7)
    plt.box(on=None)
   # plt.tight_layout(pad = 40)
    plt.draw()
    plt.savefig(os.path.join(outpath, fname_out), dpi=300)
    plt.close()
    

def main(inpath, fname_results, fname_design, outpath = None):

    if outpath is None:
        outpath = inpath = Path(inpath)
    else:
        outpath = Path(outpath)
    os.makedirs(outpath, exist_ok = True)
    # 1) Read in experiment result data
    if fname_results.endswith('.xlsx'):
        dfres = pd.read_excel(os.path.join(inpath, fname_results))
    elif fname_results.endswith('.csv'):
        dfres = pd.read_csv(os.path.join(inpath, fname_results))
    # ['Nexp' 'PID', 'Y Label', 'Y Exp', 'Y Truth', 'Std Y Exp', 'Std Y Truth', 'Weight PID']
    # 2) Read in experiment design setup table with parameter specifications
    dfdes = pd.read_csv(os.path.join(inpath, fname_design))
    # dfdes = pd.read_csv('designs_Danial/' + 'designtable_Nrun36.csv' )

    # Filter out design parameters that are constant
    dfdes = dfdes[dfdes.columns[dfdes.nunique() > 1]].copy() 

    # List of different predictable Y properties:
    try:
        ylabels = dfres["Y Label"].unique()
    except:
        print("No column with name 'Y Label' found in results file. Default results name 'Y1' added.")
        dfres["Y Label"] = 'Y1'
        ylabels = dfres["Y Label"].unique()
    params = list(dfdes)[1:]
    npar = len(params)
    nexp = dfdes.shape[0]

    # Calculating main stats (RMSE, parameter importance, best parameters)
    calc_expresults_stats(ylabels, dfdes, dfres, outpath)

    # Visualise correlation results for each Y predictable
    for ylabel in ylabels:
        print("Plotting correlation plots for Ylabel:" + str(ylabel) + " ...")
        dfname = os.path.join(outpath, "Experiment_" + str(ylabel) + "_RMSE.csv")
        df = pd.read_csv(dfname)
        # Plot Pairwise X correlation for Y:
        fname_out1 = (os.path.join(
            outpath, "Y-pairwise-correlation_" + str(ylabel) + ".png")
        )
        plot_3dmap(df, params, "Y Exp Mean", fname_out1)
        # Plot Pairwise X correlation for RMSE
        fname_out2 = (os.path.join(
            outpath, "RMSE-pairwise-correlation_" + str(ylabel) + ".png")
        )
        plot_3dmap(df, params, "RMSE", fname_out2)
        # Plot Main factor correlation plot with Y:
        fname_out3 = os.path.join(outpath, "Expresult_correlation_X-Y_" + str(ylabel) + ".png")
        plot_regression(df, params, 'Y Exp Mean', fname_out3)
        fname_out4 = os.path.join(outpath, "Expresult_distribution_X-RMSE_" + str(ylabel) + ".png")
        plot_factordis(df, params, 'RMSE', fname_out4)

    print("FINISHED")


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('settings_path', nargs='?', default='settings_expresults.yaml')
    args = ap.parse_args()
    print(f"using settings in: {args.settings_path!r}")
    with open(args.settings_path) as f:
        cfg = yaml.safe_load(f)
    main(**cfg)


if __name__ == "__main__":
    #from doegen import configloader_results as cfg
    main_cli()