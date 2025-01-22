#!/usr/bin/env python3.10

"""
Created on Mon Dec 6, 2021  
Updated August 9, 2022
@author: Alec Kercheval
adjusted: Jacob Lan
"""

'''
program to run simulations demonstrating average correlation and fraction of variance explained by PC1

'''

import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
import simulationJSE as sjse


def UpperTriMasking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


def CorrAverage(corr):
    upperTrilArray = UpperTriMasking(corr.to_numpy())
    trilAvg = np.average(upperTrilArray)

    return trilAvg

def PC_FractionEigsh(cov, FactorFrac):
    p = cov.shape[1]
    vals, vecs = eigsh(np.array(cov), p, which='LM')
    vals = vals[::-1]
    pcFraction = vals[:FactorFrac].sum() / np.trace(cov)  # choose how many (FactorFrac) leading eigenvalues are summed for calculation

    return pcFraction

# The result shows the delta^2 is minor, not enough to make up the difference between AvgCorr and FracVar
def PC_FractionEigshAdjust(n, cov, k):
    p = cov.shape[1]
    vals, vecs = eigsh(np.array(cov), p, which='LM')
    vals = vals[::-1]

    # Estimate delta squared and adjust fraction of PC1
    l_squared = (np.sum(vals) - vals[:k].sum()) / (n - k)
    delta_squared = l_squared * (n / p)
    delta_squared_fraction = delta_squared / np.trace(cov)
    # print("Delta_squared:", delta_squared)

    pcFraction = (vals[:k].sum() - k * delta_squared) / np.trace(cov)  # choose how many (k) leading eigenvalues are summed for calculation

    return pcFraction, delta_squared_fraction

def estimate_market_returns(returns_data):
    """
    Estimate market returns as the equal-weighted average of all securities.

    :param returns_data: DataFrame with securities as columns and returns as values
    :return: Series of estimated market returns
    """
    return returns_data.mean(axis=1)

def OneSetComparison(RandSeed, NormalFlag, MaxAssets, NumExperiments, NumPeriods, BetaMean, BetaStDev, Factor1StDev,
                     SpecificStDev, FactorFlag, FactorFrac, Factor2StDev, Factor3StDev, Factor4StDev):

    rng = np.random.default_rng(RandSeed)  # makes a random number generator, random seed
    sim = sjse.SimulationJSE(rng, NormalFlag, MaxAssets, NumExperiments, NumPeriods, BetaMean, BetaStDev, Factor1StDev,
                             SpecificStDev, FactorFlag, Factor2StDev, Factor3StDev, Factor4StDev)

    # Get returns matrix (assets x periods x numExperiments) from sim object
    Rtot = sim.GetReturnsMatrix()

    CorrAvgList = np.zeros((NumExperiments))
    FracEigshList = np.zeros((NumExperiments))
    Delta2List = np.zeros((NumExperiments))
    DiffList = np.zeros((NumExperiments))
    for exper in range(NumExperiments):
        Y = Rtot[:, :, exper]  # matrix of returns for trial exper
        Y_df = pd.DataFrame(Y.transpose())

        cov = Y_df.cov()
        corr = Y_df.corr()

        # Calculate the average correlation and fraction of variance explained
        corrAvg = CorrAverage(corr)
        fractionEigsh, delta_squared_fraction = PC_FractionEigshAdjust(NumPeriods, cov, FactorFrac)

        # Calculate the difference between fraction of variance explained and the average correlation
        diff = fractionEigsh - corrAvg

        # Store results
        CorrAvgList[exper] = corrAvg
        FracEigshList[exper] = fractionEigsh
        Delta2List[exper] = delta_squared_fraction
        DiffList[exper] = diff

    return CorrAvgList, FracEigshList, Delta2List, DiffList

### end def

#####################################################################
### main program   ##################################################
#####################################################################

# set up parameters for input to SimulationGPS object

MaxAssets = 500  # default 500
NumExperiments = 30  # default 30
NumPeriods = 252  # default 252

# BetaMean = 1.0 / np.sqrt(NumPeriods)
BetaMean = 1.0
# BetaStDev = 0.5

Factor1StDev = 0.16 / np.sqrt(NumPeriods)  # default 0.16/sqrt(252)
Factor2StDev = 0.04 / np.sqrt(NumPeriods)
Factor3StDev = 0.04 / np.sqrt(NumPeriods)
Factor4StDev = 0.08 / np.sqrt(NumPeriods)

OneFactorFlag = 0  # 0 for one factor; 1 for four factors
MultiFactorFlag = 1  # 0 for one factor; 1 for four factors
NormalFlag = 0  # 0 for Normal specific returns, 1 for double exponential, 2 for student's t

BetaStDevNameList = [0.25, 0.2, 0.15, 0.1, 0.05]
BetaStDevValueList = BetaStDevNameList
SpecificStDevNameList = [0.5, 0.4, 0.3, 0.2, 0.1]
SpecificStDevValueList = SpecificStDevNameList / np.sqrt(NumPeriods)

###
FactorRandSeed = 1
result_folder = r'./results'
# One factor analysis
OneFactorFrac = 1

# 1.1: One sample test, set BetaStDev=0.25 and SpecificStDev=0.1/sqrt(252), to observe the relationship of AvgCorr and FracVar
betaChangeDf = pd.DataFrame(columns=BetaStDevNameList)
diffPerList = np.zeros((NumExperiments))

CorrAvgList_, FracEigshList_, Delta2List_, DiffPerList_ = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets, NumExperiments,
                                                           NumPeriods, BetaMean, BetaStDevValueList[0], Factor1StDev,
                                                           SpecificStDevValueList[-1], OneFactorFlag, OneFactorFrac,
                                                           Factor2StDev, Factor3StDev, Factor4StDev)

comparison_df_ = pd.DataFrame(
    {
        "PC1: Fraction of Variance Explained": FracEigshList_,
        "Average Correlation": CorrAvgList_,
        "Specific Variance": Delta2List_,
        "Diff": DiffPerList_
     })

_file_path = os.path.join(result_folder, 'Compare_AvgCorr_FracVar.xlsx')
comparison_df_.to_excel(_file_path, index=False)

# # 1.2： Single-factor one sample test, set 1 factor, BetaStDev=0.25 and SpecificStDev changes, to observe the relationship of AvgCorr and FracVar
# SingleFactorFrac_ = 1
# for (SpecificStDevName, SpecificStDevValue) in zip(SpecificStDevNameList, SpecificStDevValueList):
#     print("\n", "Sum", SingleFactorFrac_, "Factor, SpcStDev:", SpecificStDevName)
#     CorrAvgList_, FracEigshList_, Delta2List_, DiffPerList_ = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
#                                                                  NumExperiments, NumPeriods, BetaMean,
#                                                                  BetaStDevValueList[0], Factor1StDev,
#                                                                  SpecificStDevValue, OneFactorFlag, SingleFactorFrac_,
#                                                                  Factor2StDev, Factor3StDev, Factor4StDev)
#     comparison_df_ = pd.DataFrame(
#         {
#             "PC1: Fraction of Variance Explained": FracEigshList_,
#             "Average Correlation": CorrAvgList_,
#             "Specific Variance": Delta2List_,
#             "Diff": DiffPerList_
#          })
#     _file_path = os.path.join(result_folder, '1Factor_Compare_AvgCorr_FracVar_' + str(SpecificStDevName)[-1] + '.xlsx')
#     comparison_df_.to_excel(_file_path, index=False)
#
# # 1.3： Multi-factor one sample test, set 1 factor, BetaStDev=0.25 and SpecificStDev changes, to observe the relationship of AvgCorr and FracVar
# MultiFactorFrac_ = 1
# for (SpecificStDevName, SpecificStDevValue) in zip(SpecificStDevNameList, SpecificStDevValueList):
#     print("\n", "Sum", MultiFactorFrac_, "Factor, SpcStDev:", SpecificStDevName)
#     CorrAvgList_, FracEigshList_, Delta2List_, DiffPerList_ = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
#                                                                  NumExperiments, NumPeriods, BetaMean,
#                                                                  BetaStDevValueList[0], Factor1StDev,
#                                                                  SpecificStDevValue, MultiFactorFlag, MultiFactorFrac_,
#                                                                  Factor2StDev, Factor3StDev, Factor4StDev)
#     comparison_df_ = pd.DataFrame(
#         {
#             "PC1: Fraction of Variance Explained": FracEigshList_,
#             "Average Correlation": CorrAvgList_,
#             "Specific Variance": Delta2List_,
#             "Diff": DiffPerList_
#          })
#     _file_path = os.path.join(result_folder, '4Factor_Compare_AvgCorr_FracVar_' + str(SpecificStDevName)[-1] + '.xlsx')
#     comparison_df_.to_excel(_file_path, index=False)

# # 2: Reduce STD of Beta, to observe the change of difference between AvgCorr and FracVar
BetaChangeDf = pd.DataFrame(columns=BetaStDevNameList)
DiffPerList = np.zeros((NumExperiments))
for (BetaStDevName, BetaStDevValue) in zip(BetaStDevNameList, BetaStDevValueList):
    print("OneFactorBetaStDev: ", BetaStDevName)
    CorrAvgList_, FracEigshList_, Delta2List_, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
                                                                 NumExperiments, NumPeriods, BetaMean, BetaStDevValue,
                                                                 Factor1StDev, SpecificStDevValueList[-1],
                                                                 OneFactorFlag, OneFactorFrac,
                                                                 Factor2StDev, Factor3StDev, Factor4StDev)
    BetaChangeDf[BetaStDevName] = DiffPerList
_file_path = os.path.join(result_folder, 'OneBetaChangeDf.xlsx')
BetaChangeDf.to_excel(_file_path, index=False)

# 3: Reduce STD of Specific Risk, to observe the change of difference between AvgCorr and FracVar
SpcChangeDf = pd.DataFrame(columns=SpecificStDevNameList)
DiffPerList = np.zeros((NumExperiments))
for (SpecificStDevName, SpecificStDevValue) in zip(SpecificStDevNameList, SpecificStDevValueList):
    print("OneFactorSpecificStDev: ", SpecificStDevName)
    CorrAvgList_, FracEigshList_, Delta2List_, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
                                                                 NumExperiments, NumPeriods, BetaMean,
                                                                 BetaStDevValueList[-1], Factor1StDev,
                                                                 SpecificStDevValue, OneFactorFlag, OneFactorFrac,
                                                                 Factor2StDev, Factor3StDev, Factor4StDev)
    SpcChangeDf[SpecificStDevName] = DiffPerList
_file_path = os.path.join(result_folder, 'OneSpcChangeDf.xlsx')
SpcChangeDf.to_excel(_file_path, index=False)

###
# Multi factor analysis
MultiFactorFracList = [1, 2, 3, 4]  # iterate to sum 1, 2, 3, 4 leading eigenvalues as numerator in the calculation of sum(leading eigenvalues)/sum(all eigenvalues)
MultiFactorFracList_no_iter = [1]
# 4: Reduce STD of Beta, to observe the change of difference between AvgCorr and FracVar
BetaChangeDf = pd.DataFrame()
for MultiFactorFrac in MultiFactorFracList:
    DiffPerList = np.zeros((NumExperiments))
    for (BetaStDevName, BetaStDevValue) in zip(BetaStDevNameList, BetaStDevValueList):
        print("\n", "Sum", MultiFactorFrac, "Factor, BetaStDev:", BetaStDevName)
        CorrAvgList_, FracEigshList_, Delta2List_, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
                                                                     NumExperiments, NumPeriods, BetaMean, BetaStDevValue,
                                                                     Factor1StDev,
                                                                     SpecificStDevValueList[-1], MultiFactorFlag,
                                                                     MultiFactorFrac,
                                                                     Factor2StDev, Factor3StDev, Factor4StDev)
        BetaChangeDf["Sum " + str(MultiFactorFrac) + " Factor_BetaStDev " + str(BetaStDevName)] = DiffPerList
# BetaChangeDf.to_excel(r'./results/MultiBetaChangeDf.xlsx', index=False)
_file_path = os.path.join(result_folder, 'MultiBetaChangeDf.xlsx')
BetaChangeDf.to_excel(_file_path, index=False)

# 5: Reduce STD of Specific Risk, to observe the change of difference between AvgCorr and FracVar
SpcChangeDf = pd.DataFrame()
for MultiFactorFrac in MultiFactorFracList:
    DiffPerList = np.zeros((NumExperiments))
    for (SpecificStDevName, SpecificStDevValue) in zip(SpecificStDevNameList, SpecificStDevValueList):
        print("\n", "Sum", MultiFactorFrac, "Factor, SpcStDev:", SpecificStDevName)
        CorrAvgList_, FracEigshList_, Delta2List_, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
                                                                     NumExperiments, NumPeriods, BetaMean,
                                                                     BetaStDevValueList[-1], Factor1StDev,
                                                                     SpecificStDevValue, MultiFactorFlag, MultiFactorFrac,
                                                                     Factor2StDev, Factor3StDev, Factor4StDev)
        SpcChangeDf["Sum " + str(MultiFactorFrac) + " Factor_SpcStDev " + str(SpecificStDevName)] = DiffPerList
# SpcChangeDf.to_excel(r'./results/MultiSpcChangeDf.xlsx', index=False)
_file_path = os.path.join(result_folder, 'MultiSpcChangeDf.xlsx')
SpcChangeDf.to_excel(_file_path, index=False)