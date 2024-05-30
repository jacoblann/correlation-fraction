# correlation-fraction.py

"""
Created on Mon Dec 6, 2021  
Updated August 9, 2022
@author: Alec Kercheval
adjusted: Jacob Lan
"""

'''
program to run simulations demonstrating average correlation and fraction of variance explained by PC1

'''

import numpy as np
import pandas as pd
import simulationJSE as sjse
from scipy.sparse.linalg import eigsh
import numpy.linalg as nlg


def UpperTriMasking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


def CorrAverage(corr):
    upperTrilArray = UpperTriMasking(corr.to_numpy())
    trilAvg = np.average(upperTrilArray)

    return trilAvg


def CorrAvgMinusMethod(corr):
    p = corr.shape[1]
    trilSum = np.tril(corr).sum() - np.trace(corr)
    trilAvg = trilSum / ((p * p - p) / 2)

    return trilAvg


def PC_FractionNLG(cov):
    # An alternative NLG method to calculate eigenvalues
    eigValue, eigVector = nlg.eig(cov)
    NLGPc1Fraction = eigValue[0] / np.sum(eigValue)

    return NLGPc1Fraction


def PC_FractionEigsh(cov, FactorFrac):
    p = cov.shape[1]
    vals, vecs = eigsh(np.array(cov), p, which='LM')
    vals = vals[::-1]
    pcFraction = vals[:FactorFrac].sum() / np.trace(cov)  # choose how many (FactorFrac) leading eigenvalues are summed for calculation

    return pcFraction


def OneSetComparison(RandSeed, NormalFlag, MaxAssets, NumExperiments, NumPeriods, BetaMean, BetaStDev, Factor1StDev,
                     SpecificStDev, FactorFlag, FactorFrac, Factor2StDev, Factor3StDev, Factor4StDev):
    rng = np.random.default_rng(RandSeed)  # makes a random number generator, random seed

    sim = sjse.SimulationJSE(rng, NormalFlag, MaxAssets, NumExperiments, NumPeriods, BetaMean, BetaStDev, Factor1StDev,
                             SpecificStDev, FactorFlag, Factor2StDev, Factor3StDev, Factor4StDev)

    # get returns matrix (assets x periods x numExperiments) from sim object
    Rtot = sim.GetReturnsMatrix()

    CorrAvgList = np.zeros((NumExperiments))
    FracEigshList = np.zeros((NumExperiments))
    DiffPerList = np.zeros((NumExperiments))
    for exper in range(NumExperiments):
        Y = Rtot[:, :, exper]  # matrix of returns for trial exper
        Y_df = pd.DataFrame(Y.transpose())

        cov = Y_df.cov()
        corr = Y_df.corr()

        corrAvg = CorrAverage(corr)
        corrAvgMinus = CorrAvgMinusMethod(corr)  # an alternative way to calculate average correlations, should be same as corrAvg

        fractionEigsh = PC_FractionEigsh(cov, FactorFrac)
        fractionNLG = PC_FractionNLG(cov)  # an alternative NLG way to calculate eigenvalues, should be same as fractionEigsh in PC1

        diffPer = (fractionEigsh - corrAvg) / ((fractionEigsh + corrAvg) / 2)   # difference between AvgCorr and FractionVar in percentage form

        CorrAvgList[exper] = corrAvg
        FracEigshList[exper] = fractionEigsh
        DiffPerList[exper] = diffPer

    return CorrAvgList, FracEigshList, DiffPerList

### end def


#####################################################################
### main program   ##################################################
#####################################################################

# set up parameters for input to SimulationGPS object

MaxAssets = 500  # default 500
NumExperiments = 30  # default 400
NumPeriods = 252  # default 252

BetaMean = 1.0
# BetaStDev = 0.5

Factor1StDev = 0.16 / np.sqrt(NumPeriods)  # default 0.16/sqrt(252)
# SpecificStDev = 0.6 / np.sqrt(NumPeriods)  # daily vol from annual

Factor2StDev = .04 / np.sqrt(NumPeriods)
Factor3StDev = .04 / np.sqrt(NumPeriods)
Factor4StDev = .08 / np.sqrt(NumPeriods)

OneFactorFlag = 0  # 0 for one factor; 1 for four factors
MultiFactorFlag = 1  # 0 for one factor; 1 for four factors
NormalFlag = 0  # 0 for Normal specific returns, 1 for double exponential, 2 for student's t

BetaStDevList = [0.25, 0.2, 0.15, 0.1, 0.05]
SpecificStDevNameList = [0.5, 0.4, 0.3, 0.2, 0.1]
SpecificStDevValueList = SpecificStDevNameList / np.sqrt(NumPeriods)

###
FactorRandSeed = 1
# One factor analysis
OneFactorFrac = 1
# 1: One sample test, set BetaStDev=0.25 and SpecificStDev=0.1/sqrt(252), to observe the relationship of AvgCorr and FracVar
betaChangeDf = pd.DataFrame(columns=BetaStDevList)
diffPerList = np.zeros((NumExperiments))

CorrAvgList, FracEigshList, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets, NumExperiments,
                                                           NumPeriods, BetaMean, BetaStDevList[0], Factor1StDev,
                                                           SpecificStDevValueList[-1], OneFactorFlag, OneFactorFrac,
                                                           Factor2StDev, Factor3StDev, Factor4StDev)

comparison_df = pd.DataFrame(
    {"PC1: Fraction of Variance Explained": FracEigshList, "Average Correlation": CorrAvgList
        , "Diff Percentage": DiffPerList
     })
comparison_df.to_excel(r'./results/Compare_AvgCorr_FracVar.xlsx', index=False)

# 2: Reduce STD of Beta, to observe the change of difference between AvgCorr and FracVar
BetaChangeDf = pd.DataFrame(columns=BetaStDevList)
DiffPerList = np.zeros((NumExperiments))
for BetaStDev in BetaStDevList:
    print("OneFactorBetaStDev: ", BetaStDev)
    CorrAvgList_, FracEigshList_, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
                                                                 NumExperiments, NumPeriods, BetaMean, BetaStDev,
                                                                 Factor1StDev, SpecificStDevValueList[-1],
                                                                 OneFactorFlag, OneFactorFrac,
                                                                 Factor2StDev, Factor3StDev, Factor4StDev)
    BetaChangeDf[BetaStDev] = DiffPerList
BetaChangeDf.to_excel(r'./results/OneBetaChangeDf.xlsx', index=False)

# 3: Reduce STD of Specific Risk, to observe the change of difference between AvgCorr and FracVar
SpcChangeDf = pd.DataFrame(columns=SpecificStDevNameList)
DiffPerList = np.zeros((NumExperiments))
for (SpecificStDevName, SpecificStDevValue) in zip(SpecificStDevNameList, SpecificStDevValueList):
    print("OneFactorSpecificStDev: ", SpecificStDevName)
    CorrAvgList_, FracEigshList_, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
                                                                 NumExperiments, NumPeriods, BetaMean,
                                                                 BetaStDevList[-1], Factor1StDev,
                                                                 SpecificStDevValue, OneFactorFlag, OneFactorFrac,
                                                                 Factor2StDev, Factor3StDev, Factor4StDev)
    SpcChangeDf[SpecificStDevName] = DiffPerList
SpcChangeDf.to_excel(r'./results/OneSpcChangeDf.xlsx', index=False)


###
# Multi factor analysis
MultiFactorFracList = [1, 2, 3, 4]  # iterate to sum 1, 2, 3, 4 leading eigenvalues as numerator in the calculation of sum(leading eigenvalues)/sum(all eigenvalues)
# 4: Reduce STD of Beta, to observe the change of difference between AvgCorr and FracVar
BetaChangeDf = pd.DataFrame()
for MultiFactorFrac in MultiFactorFracList:
    DiffPerList = np.zeros((NumExperiments))
    for BetaStDev in BetaStDevList:
        print("Sum", MultiFactorFrac, "Factor, BetaStDev:", BetaStDev)
        CorrAvgList_, FracEigshList_, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
                                                                     NumExperiments, NumPeriods, BetaMean, BetaStDev,
                                                                     Factor1StDev,
                                                                     SpecificStDevValueList[-1], MultiFactorFlag,
                                                                     MultiFactorFrac,
                                                                     Factor2StDev, Factor3StDev, Factor4StDev)
        BetaChangeDf["Sum " + str(MultiFactorFrac) + " Factor_BetaStDev " + str(BetaStDev)] = DiffPerList
BetaChangeDf.to_excel(r'./results/MultiBetaChangeDf.xlsx', index=False)

# 5: Reduce STD of Specific Risk, to observe the change of difference between AvgCorr and FracVar
SpcChangeDf = pd.DataFrame()
for MultiFactorFrac in MultiFactorFracList:
    DiffPerList = np.zeros((NumExperiments))
    for (SpecificStDevName, SpecificStDevValue) in zip(SpecificStDevNameList, SpecificStDevValueList):
        print("Sum", MultiFactorFrac, "Factor, SpcStDev:", SpecificStDevName)
        CorrAvgList_, FracEigshList_, DiffPerList = OneSetComparison(FactorRandSeed, NormalFlag, MaxAssets,
                                                                     NumExperiments, NumPeriods, BetaMean,
                                                                     BetaStDevList[-1], Factor1StDev,
                                                                     SpecificStDevValue, MultiFactorFlag, MultiFactorFrac,
                                                                     Factor2StDev, Factor3StDev, Factor4StDev)
        SpcChangeDf["Sum " + str(MultiFactorFrac) + " Factor_SpcStDev " + str(SpecificStDevName)] = DiffPerList
SpcChangeDf.to_excel(r'./results/MultiSpcChangeDf.xlsx', index=False)
