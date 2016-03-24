# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:10:29 2016

Run Friedman test with multiple comparisons based on Conover 1999 'Practical
Non-Parameteric Statistics, 3rd Ed. (Following Iman and Davenport 1980).

As an alternative for small sample sizes, there is also a bootstrap
implementation of Friedman's test.

The Friedman test tests whether there is a difference among k treatments
(columns) where data is grouped into similar blocks (e.g. a colony to which
several treatments were applied to different zooids). Specifically, the null
hypothesis is that all rankings of data within a block are equally likely

@author: Michelangelo
"""
from scipy.stats import rankdata, f
from numpy import asarray, sum, empty, indices, array, isnan, logical_and, nan
from numpy.random import randint


def resamplealongrows(mydata):
    """
    Create bootsrap of array, resampling along rows, but not columns
    Produces an array from mydata that has the same dimensions, but each row
    is formed by taking samples (with replacement) from that row. Order of
    rows is left unchanged.
    mydata: a 2D array or list of equal-length lists of numbers
    """
    mydata = asarray(mydata)  # Force data to be an array.
    [nblocks, nts] = mydata.shape  # Get dimensions of data
    rowinds = indices((nblocks, nts))[0]  # list of row indices
    # Resample column indices FOR EACH ROW with replacement.
    colinds = randint(nts, size=[nblocks, nts])  
    # Return values from mydata at indices specified by rowinds and colinds
    return(mydata[rowinds, colinds])  

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
    # Resample column indices FOR EACH ROW with replacement.
    colinds = randint(nts, size=[nblocks, nts])
    # Return values from mydata at indices specified by rowinds and colinds
    return(mydata[rowinds, colinds])


def TestFriedStuff():
    """
    function to run tests on functions above.
    """
    # Test example for Friedman test from Conover
    print('Grass data example from conover: ')
    grassdata = [[4,3,2,1], [4,2,3,1], [3,1.5,1.5,4], [3,1,2,4], [4,2,1,3],
                 [2,2,2,4], [1,3,2,4], [2,4,1,3], [3.5,1,2,3.5], [4,1,3,2],
                 [4,2,3,1], [3.5,1,2,3.5]]
    print('Output from criedman() function, which follows Conover 1999)')
    friedgrass = friedmanstat(grassdata)
    friedgrass.cfriedman()
    print(vars(friedgrass))
    print('')
    print('Expected Output: Summed Ranks: [ 38.   23.5  24.5  34. ]',
          'nblocks: 12 ; ntreatments: 4 ; A1: 356.5 ; C1: 300.0',
          'T1: 8.09734513274 ; T2: 3.19219790676 ; P: 0.0362154746433')

    # Test Friedman for simple data set where it should be weak. Bootstrap
    # maxk resamples (with replacement), and calculate Conover's version of
    # Friedman test
    print('Test cfriedman() for case in which it should perform poorly')
    simdata = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]  # ,[1,0,0]]
    print('simdata: ', simdata)
    nboot = 5000
    p = empty(nboot)
    l = 0
    for k in range(nboot):
        bootdat = resamplealongrows(simdata)
        bootl = friedmanstat(bootdat)
        bootl.cfriedman()
        p[k] = bootl.P
        if logical_and(isnan(p[k]), l < 5):
            print('Found a nan! Mmmm... naan: ')
            print(bootdat)
            l = l + 1

    print('number of bootstrap resamples: ', nboot)
    print('with nans')
    for alpha in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                  0.0001]:
        print('Fraction with p<', alpha, ': ', sum(p < alpha) / p.size)
    # nan's only appear when T1 becomes a nan, when all data values are the
    # same. Therefore, should cound as p=1; shouldn't make a difference, but
    # check.
    print('without nans (set p to 1 for nans)')
    p2 = array(p)
    p2[isnan(p)] = 1
    for alpha in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                  0.0001]:
        print('Fraction with p<', alpha, ': ', sum(p2 < alpha) / p2.size)
    #  Expected output for both: fraction of p<alpha should be fairly close to
    #  alpha, except at very low alpha. This suggests that even for low numbers
    #  of replicates, Conover's version of the Friedman test seems pretty good,
    #  although very low p-values are much too low, and intermediate
    #  (0.1 - 0.05) are conservative.
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

    print('---')
    print('Test bootstrap version (bfriedman() of Friedman test')
    maxb = 10000
    print('p value for grass data from Conover; nboot = ', maxb)
    friedgrass.bfriedman(nboot=maxb)
    print(vars(friedgrass))


def friedmantest(mydata, variant='Fdist', nboot = 5000, verbose=True):
    """
    mydata: 2-level list or dim-2 array (columns: treatments; rows: blocks)
    variant: 'Fdist' gives version from Conover (originally Iman & Davenport);
        'boot' gives bootstrap version
    """
    try:
        mystats = friedmanstat(mydata)
        if (variant == 'Fdist'):
            mystats.cfriedman()
        elif (variant == 'Boot'):
            mystats.bfriedman(nboot)
        else:
            mystats = None
            print('Variant not recognized.')

        if verbose and not (mystats is None):
            print(vars(mystats))

        return(mystats)
    except:
        print('Error in friedmantest')
        return(None)


class friedmanstat:
    """
    Object containing stats for Friedman test.
    """
    def __init__(self, mydata):
        """
        Initialize a friedmanstat object with statistic T2 (from Conover 1999)
        mydata: an array or list of equal sized lists.
        Columns of mydata are treatments; rows of mydata are blocks.
        """
        try:
            mydata = asarray(mydata)  # Convert mydata to array

            # Size of array: nblocks: number of tuples/blocks  (b in Conover);
            # nts: number of treatments (k in Conover)
            [nblocks, nts] = mydata.shape

            ranks = empty([nblocks, nts])  # Initialize array for ranks

            # Create rank matrix, with tied ranks given average of ranks
            for k in range(nblocks):
                ranks[k,:] = rankdata(mydata[k, :])  # , method='average')

            # summedranks: Sum of ranks for each treatment (Rj in Conover)
            summedranks = sum(ranks, axis=0)
            # sumsqrranks: Sum of squared ranks used to correct for ties (A1
            # in Conover)
            sumsqrranks = sum(ranks**2)
            # correction: correction factor for ties (C1 in Conover)
            correction = nblocks*nts*((nts+1)**2)/4

            # T1: Friedman's T1 statistic; approximately follows Chi2 under
            # null hypothesis. If statements to deal with cases (e.g. when all
            # columns are tied in all blocks, or some other error crops up.
            T1numerator = (nts-1)*sum((summedranks-nblocks*(nts+1)/2)**2)
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
                # T2: Preferred statistic (follows F-distribution under null)
                if X <= T1:
                    if logical_and(T1 > 0, X == T1):
                        T2 = inf
                    else:
                        T2 = nan
                else:
                    T2 = ((nblocks-1)*T1/(X-T1))

            self.T2 = T2  # Improved Friedman statistic; follows F-distribution 
            # see Conover 1999
            self.nblocks = nblocks # number of blocks
            self.nts = nts # number of treatments
            self.summedranks = summedranks # summed ranks along columns
            self.sumsqrranks = sumsqrranks # sum squared ranks
            self.tiecorrection = correction # correction factor for ties
            self.T1 = T1 #Standard Friedman statistic
            self.P = None # P-value determined by test functions below.
            # Distribution used in test. Fill in with test functions below.         
            self.distribution = None
            self.mydata = mydata  # Input data as an array

        except:
            self.T2 = None
            self.mydata = None
            print('Error in initializing friedmanstat structure')

    def cfriedman(self):
        """
        This function uses Conover's recommendation for an improved version
        that compares to the F-distribution rather than the Chi-square.
        """
        try:
            # Calculate p-value based on cdf of F-distribution for T2
            self.P = 1-f.cdf(self.T2,self.nts-1,(self.nblocks-1)*(self.nts-1))
            self.distribution = 'f.cdf'
        except:
            self.P = None
            self.distribution = None
            print('Error in cfriedman')

    def bfriedman(self, nboot=5000):
        """
        Bootstrap version of Friedman test
        """
        try:
            if isnan(self.T2):
                self.T2 = 0
                print('T2 is nan; T2 set to 0.')
                self.P = None
                self.distribution = None
            else:
                T2boot = empty(nboot)
                for k in range(nboot):
                    T2boot[k] = friedmanstat(bootsample2D(self.mydata)).T2

                # Calculate P value from bootstrap sample
                self.P = sum(T2boot >= self.T2)/T2boot.size
                # 'bootstrap' will tell the next function what to do.
                self.distribution = ['bootstrap', ['nboot', nboot],
                                     ['bootsamples', T2boot]]
        except:
            self.P = None
            self.distribution = None
            print('Error in bfriedman')
