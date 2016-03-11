# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:10:29 2016

Run Friedman test with multiple comparisons based on Conover 'Practical
Non-Parameteric Statistics, 3rd Ed.

@author: Michelangelo
"""
from scipy.stats import rankdata, f
from numpy import asarray, sum, empty, indices, array, isnan, logical_and
from numpy import nan, inf, logical_or
from numpy.random import randint

import numpy as n


def cfriedmanstat(mydata):
    """
    Return the improved Friedman statistic T2 (from Conover 1999)
    """
    mydata = asarray(mydata)  # Convert mydata to array

    # Size of array: nblocks: number of tuples/blocks  (b in Conover);
    # nts: number of treatments (k in Conover)
    [nblocks, nts] = mydata.shape

    ranks = empty([nblocks, nts])  # Initialize array for ranks

    # Create rank matrix, with tied ranks given average of expected rank
    for k in range(nblocks):
        ranks[k,:] = rankdata(mydata[k, :])  # , method='average')

    # summedranks: Sum of ranks for each treatment (Rj in Conover)
    summedranks = sum(ranks, axis=0)
    # sumsqrranks: Sum of squared ranks to correct for ties (A1 in Conover)
    sumsqrranks = sum(ranks**2)
    # correction: correction factor for ties (C1 in Conover)
    correction = nblocks*nts*((nts+1)**2)/4

    # T1: Friedman's T1 statistic; approximately follows Chi2 under null
    # hypothesis
    T1numerator = (nts-1) * sum((summedranks - nblocks * (nts + 1) / 2) ** 2)
    X = nblocks*(nts-1)
    if sumsqrranks == correction:
        if T1numerator <= 0:
            T1 = nan
            T2 = nan
        else:
            T1 = inf            
            T2 = inf
            
    else:
        T1 = T1numerator/(sumsqrranks - correction)
        # T2: Preferred statistic that follows F-distribution under the null.
        if X <= T1:
            if logical_and(T1 > 0, X == T1):
                T2 = inf
            else:
                T2 = nan
        else:
            T2 = ((nblocks-1)*T1/(X-T1))
    return([['T2', T2], ['nblocks', nblocks], ['nts', nts],
            ['summed ranks (along columns)', summedranks],
            ['sum squared ranks', sumsqrranks],
            ['correction factor for ties', correction], ['T1',T1]])


def cfriedman(mydata, verbose=True):
    """
    Perform Friedman test and multiple comparisons.
    It tests whether there is a difference among k treatments (columns) where
    data is grouped into similar blocks (e.g. a colony to which several
    treatments were applied to different zooids).
    This function uses Conover's recommendation for an improved version that
    compares to the F-distribution rather than the Chi-square.
    mydata: an array or list of equal sized lists.
    Columns of mydata: treatments; rows of mydata: blocks.
    """
    T2list = cfriedmanstat(mydata)
    if logical_and(logical_and(T2list[0][0] == 'T2',
             T2list[1][0] == 'nblocks'),
             T2list[2][0] == 'nts'):
        T2 = T2list[0][1]
        nblocks = T2list[1][1]
        nts = T2list[2][1]
        # Calculate p-value based on cdf of F-distribution
        Pfried = 1-f.cdf(T2, nts-1, (nblocks-1)*(nts-1))
        if verbose:
            print('P:', Pfried)
            print(T2list)

        return([['P', Pfried], T2list])
    else:
        return('Output of cfriedmanstat() not as expected')

# Test example for Friedman test from Conover
grassdata = [[4,3,2,1],[4,2,3,1],[3,1.5,1.5,4],[3,1,2,4],[4,2,1,3],[2,2,2,4],
             [1,3,2,4],[2,4,1,3],[3.5,1,2,3.5],[4,1,3,2],[4,2,3,1],
             [3.5,1,2,3.5]]
cfriedman(grassdata)
print('')
#  Expected Output:
#  Summed Ranks
#  [ 38.   23.5  24.5  34. ]
#  nblocks: 12 ; ntreatments: 4 ; A1: 356.5 ; C1: 300.0
#  T1: 8.09734513274 ; T2: 3.19219790676 ; P: 0.0362154746433


def resamplealongrows(mydata):
    """
    Create bootsrap of array, resampling along rows, but not columns
    Produces an array from mydata that has the same dimensions, but each row
    is formed by taking samples (with replacement) from that row. Order of
    rows is left unchanged.
    mydata: a 2D array or list of equal-length lists of numbers
    """
    mydata = asarray(mydata)  # Force data array to be array.
    [nblocks, nts] = mydata.shape  # Get dimensions of data
    rowinds = indices((nblocks, nts))[0]  # list of row indices
    colinds = randint(nts, size=[nblocks, nts])  # Resample column indices FOR
                                            # EACH ROW with replacement.
    return(mydata[rowinds, colinds])  # Return values from mydata at indices
                                # specified by rowinds and colinds

# Test Friedman for simple data set where it should be weak. Bootstrap
# maxk resamples (with replacement), and calculate Conover's version of
# Friedman test
simdata = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]  # ,[1,0,0]]
print('simdata: ', simdata)
nboot = 5000
p = empty(nboot)
l = 0
for k in range(nboot):
    bootdat = resamplealongrows(simdata)
    p[k] = cfriedman(bootdat, verbose=False)[0][1]
    if logical_and(n.isnan(p[k]), l < 5):
        print('Found a nan! Mmmm... naan: ')
        print(bootdat)
        l = l + 1

print('number of bootstrap resamples: ', nboot)
print('with nans')
for alpha in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0001]:
    print('Fraction with p<', alpha, ': ', sum(p < alpha) / p.size)
# nan's only appear when T1 becomes a nan, when all data values are the
# same. Therefore, should cound as p=1; shouldn't make a difference, but check.
print('without nans (set p to 1 for nans)')
p2 = array(p)
p2[isnan(p)] = 1
for alpha in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0001]:
    print('Fraction with p<', alpha, ': ', sum(p2 < alpha) / p2.size)
#  Expected output for both: fraction of p<alpha should be fairly close to
#  alpha, except at very low alpha. This suggests that even for low numbers of
#  replicates, Conover's version of the Friedman test seems pretty good,
#  although very low p-values are much too low, and intermediate (0.1 - 0.05)
#  are conservative.
#  Example output:
#  simdata:  [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
#  number of bootstrap resamples:  5000
#  with nans (p=1)
#  Fraction with p< 0.5 :  0.5532
#  Fraction with p< 0.2 :  0.183
#  Fraction with p< 0.1 :  0.0888
#  Fraction with p< 0.05 :  0.03
#  Fraction with p< 0.02 :  0.03
#  Fraction with p< 0.01 :  0.0112
#  Fraction with p< 0.005 :  0.0014
#  Fraction with p< 0.002 :  0.0014
#  Fraction with p< 0.001 :  0.0014
#  Fraction with p< 0.0001 :  0.0014


def bootsample2D(mydata):
    """
    Create bootsrap of array, resampling rows, and then resampling along rows.
    Produces an array from mydata that has the same dimensions, but each row
    is formed by first picking random rows from mydata, and then taking samples
    (with replacement) from each row. 
    mydata: a 2D array or list of equal-length lists of numbers
    """
    mydata = asarray(mydata)  # Force data array to be array.
    [nblocks, nts] = mydata.shape  # Get dimensions of data
    # rowinds: list of row indices, randomized by row
    rowinds = indices((nblocks, nts))[0][randint(nblocks, size=nblocks), :]
    colinds = randint(nts, size=[nblocks, nts])  # Resample column indices FOR
                                            # EACH ROW with replacement.
    return(mydata[rowinds, colinds])  # Return values from mydata at indices
                                # specified by rowinds and colinds


def bfriedman(mydata, nboot=5000, verbose=True):
    """
    Bootstrap version of Friedman test
    """
    mydata = asarray(mydata)  # Convert mydata to array

    T2list = cfriedmanstat(mydata)
    if T2list[0][0] == 'T2':
        T2 = T2list[0][1]
        if isnan(T2):
            T2 = 0
            print('T2 is nan; T2 set to 0.')
            return(nan)
        else:
            T2boot = empty(nboot)
            for k in range(nboot):
                T2boot[k] = cfriedmanstat(bootsample2D(mydata))[0][1]

            pboot = sum(T2boot >= T2)/T2boot.size
            return([['P', pboot], T2list])
    else:
        return('Output of cfriedmanstat() not as expected')

print('---')
print('Test bootstrap version of Friedman test')
maxb = 10000
print('p value for grass data from Conover; nboot = ', maxb)
print(bfriedman(grassdata, nboot=maxb))
