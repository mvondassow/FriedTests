# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:41:45 2016

@author: Michelangelo
"""
from scipy.stats import rankdata, f, t
from numpy import asarray, sum, empty, indices, array, isnan, logical_and, nan
from numpy.random import randint, random
import numpy as np


def resamplealongrows(mydata):
    """
    Resample an array ONLY ALONG THE ROWS.

    Create resampling of an array, but resampling along rows only, not columns
    Produces an array from mydata that has the same dimensions, but each row
    is formed by taking samples (with replacement) from that row. Order of
    rows is left unchanged.

    Parameters
    ----------
    mydata : list or ndarray
        A 2D array or list of equal-length lists of numbers

    Returns
    -------
    ndarray of same size and shape as mydata, resampled along rows.
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
    Resample a 2D array once, to produce a new 2D array.

    First resamples rows, and then resamples along rows. Produces an array from
    mydata that has the same dimensions, but each row is formed by first
    picking random rows from mydata, and then taking samples (with replacement)
    from each row.

    Parameters
    ----------
    mydata : ndarray or list
        Either a 2D array or list of equal-length lists of numbers

    Returns
    -------
    ndarray, 2 dimensional array with random sampling of entries in mydata.
    """
    mydata = asarray(mydata)  # Force data array to be array.
    [nblocks, nts] = mydata.shape  # Get dimensions of data
    # rowinds: list of row indices, randomized by row
    rowinds = indices((nblocks, nts))[0][randint(nblocks, size=nblocks), :]
    # Resample column indices FOR EACH ROW with replacement.
    colinds = randint(nts, size=[nblocks, nts])
    # Return values from mydata at indices specified by rowinds and colinds
    return(mydata[rowinds, colinds])


def bootset2D(mydata, nreps):
    """
    Resample a 2D array multiple times, to produce a 3D array of multiple
    bootstrap samples.

    Produces 3D array from mydata in which each layer is formed by first
    picking random rows from mydata, and then taking random samples (with
    replacement) from each row. It is a 3D stack of bootstrap samples of the
    2D array, preserving the structure of nts measurements from nblocks blocks.

    Parameters
    ----------
    mydata : list or ndarray
        A 2D array or list of equal-length lists of numbers; each layer (along
        dim 0) contains one randomization.
    nreps : number of times to run resampling

    Returns
    -------
    ndarray
        Array is dim 3, numeric. Each layer (along dim 0) contains one
        resampling of mydata: nreps layers, by nblocks, by nts.
    """
    mydata = asarray(mydata)  # Force data array to be array.
    [nblocks, nts] = mydata.shape  # Get dimensions of data
    # rowinds: list of row indices, randomized by row
    rowinds = indices(
        (nblocks, nts))[0][randint(nblocks, size=[nreps, nblocks]), :]
    # Resample column indices FOR EACH ROW with replacement.
    colinds = randint(nts, size=[nreps, nblocks, nts])
    # Return values from mydata at indices specified by rowinds and colinds
    return(mydata[rowinds, colinds])


def randomizeAlongRows(mydata, nreps):
    """
    Randomize a 2D array multiple times along the rows, to produce a 3D array
    of multiple randomized samples.

    Produces an array from mydata that has the same dimensions, but each row
    is formed by taking a randome permutation of the data from the original
    row.

    Parameters
    ----------
    mydata : list or ndarray
        A 2D array or list of equal-length lists of numbers; each layer (along
        dim 0) contains one randomization.
    nreps : number of times to run randomization

    Returns
    -------
    ndarray
        Array is dim 3, numeric. Each layer (along dim 0) contains one
        randomization of mydata.
    """
    mydata = asarray(mydata)  # Force data array to be array.
    [nrows, ncolumns] = mydata.shape  # Get dimensions of data
    # Create 3D array of random floats
    myrands = random(size=(nreps, nrows, ncolumns))
    # Convert random floats to randomized column indices.
    colinds = np.argsort(myrands, 2)
    # Create 3D array of row indices (matching original column indices).
    rowinds = np.tile(
        np.reshape(np.arange(nrows), (1, nrows, 1)), (nreps, 1, ncolumns))
    # Return values from mydata at indices specified by rowinds and colinds
    return(mydata[rowinds, colinds])


def rankalongrows3D(mydata):
    """
    Ranks data along the rows of a 3-D, numeric array.
    Take average of ranks for ties.

    Parameters:
    -----------
    mydata : ndarray, dim 3, numeric

    Returns:
    --------
    rankavg : ndarray, dim 3, numeric
        Contains ranks of data, with average value for ties.
    """
    # Get length of each dimension of mydata
    nlayers, nrows, ncolumns = np.shape(mydata)
    # create 3D arrays containing row, layer, and column indices of all entries
    # in an array of same shape as mydata
    rowinds = np.tile(
        np.reshape(np.arange(nrows), (1, nrows, 1)), (nlayers, 1, ncolumns))
    layerinds = np.tile(
        np.reshape(np.arange(nlayers), (nlayers, 1, 1)), (1, nrows, ncolumns))
    colinds = np.tile(np.arange(ncolumns), (nlayers, nrows, 1))
    # Initialize array to set for rankings
    rankpos = empty((nlayers, nrows, ncolumns))
    rankneg = empty((nlayers, nrows, ncolumns))
    # Use argsort to find the indices corresponding to the 1st ranked, 2nd
    # ranked,... entries in mydata; uses ranking method such that rank of ties
    # is assigned by position in list.
    rankpos[layerinds, rowinds, np.argsort(mydata, 2)] = colinds
    # Redo as above but with negative value of data to get reverse ranking for
    # ties
    rankneg[layerinds, rowinds, np.argsort(-mydata, 2)] = colinds
    # number of columns (ncolumns) minus the reverse ranking gives the ranking
    # of the data if argsort had traversed the rows in the opposite direction.
    # Averaging these two rankings (rankpos and (ncolumns-rankneg-1)) gives the
    # ranking averaging by ties. Add 1 to set first rank equal to 1
    rankavg = (rankpos + ncolumns - 1 - rankneg) / 2 + 1
    return(rankavg)


def friedmantest(mydata, variant='Fdist', nreps=5000, verbose=True):
    """
    Run Friedman test and multiple comparisons.
    
    Parameters:
    -----------
    mydata : list or ndarray
        Data: 2-level list or dim-2 array (columns: treatments; rows: blocks)
    variant: Boolean, default: 'Fdist'
        'Fdist' gives version from Conover (originally Iman & Davenport);
        'Rand' gives randomization version; other values give error message
    nreps : int, default: 5000
        Number of replications to use if use 'Rand'
    verbose : Boolean, default: True
        Whether to print information in friemanstat object.
    """
    try:
        mystats = friedmanstat(mydata)
        if (variant == 'Fdist'):
            mystats.cfriedman()
            mystats.multcompare()
        elif (variant == 'Rand'):
            mystats.rfriedman(nreps)
            mystats.multcompare()
        else:
            mystats = None
            print('Variant not recognized.')

        if verbose and not (mystats is None):
            mystats.printfriedstuff(verbose=True)

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
        Initialize a friedmanstat object for use in Friedman tests.

        Parameters
        -----------
        mydata: list or array
            A list of equal sized lists or an array containing numeric data.
            Columns of mydata are treatments; rows of mydata are blocks.

        Attributes
        ----------
        self.mydata : ndarray
            Input data as an array
        self.T2 : float
            Improved Friedman statistic that follows F-distribution (see
            Conover 1999).
        self.nblocks : int
            number of blocks
        self.nts : int
            number of treatments
        self.summedranks : array, dim 1
            Summed ranks along columns (treatments)
        self.sumsqrranks : float
            Grand sum of squared ranks
        self.tiecorrection : float
            Used in correction for ties
        self.T1 : float
            Standard Friedman statistic
        self.P : None
            Will hold P-value determined by test methods below.
        self.distribution : None
            Will hold distribution used in test.
        self.pairwisePs : None
            Will hold information on pairwise comparisons
        """
        try:
            mydata = asarray(mydata)  # Convert mydata to array

            # Size of array: nblocks: number of tuples/blocks  (b in Conover);
            # nts: number of treatments (k in Conover)
            [nblocks, nts] = mydata.shape

            ranks = empty([nblocks, nts])  # Initialize array for ranks

            # Create rank matrix, with tied ranks given average of ranks
            for k in range(nblocks):
                ranks[k, :] = rankdata(mydata[k, :])  # , method='average')

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
                    T1 = float('inf')
                    T2 = float('inf')

            else:
                T1 = T1numerator/(sumsqrranks - correction)
                # T2: Preferred statistic (follows F-distribution under null)
                if X <= T1:
                    if logical_and(T1 > 0, X == T1):
                        T2 = float('inf')
                    else:
                        T2 = float('inf')
                else:
                    T2 = ((nblocks-1)*T1/(X-T1))

            self.T2 = T2  # Improved Friedman statistic; follows F-distribution
            # see Conover 1999
            self.nblocks = nblocks  # number of blocks
            self.nts = nts  # number of treatments
            self.summedranks = summedranks  # summed ranks along columns
            self.sumsqrranks = sumsqrranks  # sum squared ranks
            self.tiecorrection = correction  # correction factor for ties
            self.T1 = T1  # Standard Friedman statistic
            self.P = None  # P-value determined by test functions below.
            # Distribution used in test. Fill in with test functions below.      
            self.distribution = None
            self.mydata = mydata  # Input data as an array
            self.pairwisePs = None  # pairwise comparisons

        except:
            self.T2 = None
            self.mydata = None
            print('Error in initializing friedmanstat structure.')
            print('This is what you gave me for data:')
            print(mydata)

    def cfriedman(self):
        """        
        Friedman test based on Conover 1999.

        This method uses Conover's recommendation for an improved version
        that compares to the F-distribution rather than the Chi-square.
        Generates P value and distribution information.

        Parameters
        ----------
        none
   
        Sets property
        -------------
        self.P : float
                P value
        self.distribution : list
                String describing what distribution was used for test.
        self.pairwisePs : None
                Sets to None to prevent mismatches between testing methods

        Returns
        -------
        none
        """
        try:
            # Calculate p-value based on cdf of F-distribution for T2
            df1 = self.nts-1
            df2 = (self.nblocks-1)*(self.nts-1)
            self.P = 1-f.cdf(self.T2, df1, df2)
            self.distribution = ['f.cdf', [['df1', df1], ['df2', df2]]]
            self.pairwisePs = None
        except:
            self.P = None
            self.distribution = None
            self.pairwisePs = None
            print('Error in cfriedman')

    def rfriedman(self, nreps=1000):
        """
        Randomization version of Friedman test.
        
        This method generates the P value and distribution information for a
        randomization version of the Friedman test (see Manly 1997 for basic
        idea of randomization test)

        Parameters
        ----------
        nreps : integer
                number of randomized samples to create
   
        Sets property
        -------------
        self.P : float
                P value
        self.distribution : list
                zeroth and first elements contain strings indicating that the
                distribution is created by randomization and the number of
                randomizations; the second and third elements contain two
                elements 1st) a label string, 2nd) the values from the
                randomization
        self.pairwisePs : None
                Sets to None to prevent mismatches between testing methods

        Returns
        -------
        none
        """
        try:
            if isnan(self.T1):
                self.T1 = 0
                print('T1 is nan; T1 set to 0.')
                self.P = None
                self.distribution = None
            else:
                # Generate randomized data (take 2 D data array; generate 3 D
                # array of randomizations; each layer (along dim 0) contains
                # one randomization.
                RandomizedData = randomizeAlongRows(self.mydata, nreps)
                # Rank randomized data.
                RandDataRanks = rankalongrows3D(RandomizedData)

                # Calculate Friedman statistic T1 for each randomization
                SumRandRanks = sum(RandDataRanks, 1)
                SumSqrRandRanks = sum(sum(RandDataRanks**2, 1), 1)
                correction = self.tiecorrection
                randT1numer = (self.nts-1)*sum(
                            (SumRandRanks-self.nblocks*(self.nts+1)/2)**2,1)
                randT1denom = SumSqrRandRanks - correction
                # Create logical array for values of the denominator that are
                # greater than zero
                posvals = (randT1denom > 0)
                # Create array to store values of T1 for each randomizated
                # dataset
                randT1 = np.full(nreps, nan)
                # If the denominator is zero, set randT1 to 'inf'                
                randT1[randT1denom == 0] = float(np.inf)
                # If the denominator is > 0 calculate randT1; otherwise value
                # remains 'nan'
                randT1[posvals] = randT1numer[posvals] / randT1denom[posvals]
                # Calculate P value from bootstrap sample
                self.P = sum(randT1 >= self.T1)/nreps

                # Calculate values of pairwise statistic used for comparisons
                # between treatment pairs in Conover 1999: multcompare() will
                # use these as the randomized distribution of these statistics
                # to compare to the observed statistic.
                df = (self.nblocks-1)*(self.nts-1)  # degrees of freedom
                # Denominator for pairwise t-statistic from random data
                multdenoms = (2 *
                                (self.nblocks * SumSqrRandRanks -
                                   sum(SumRandRanks ** 2, 1)) / df) ** (1/2)
                # Calculate difference in summed ranks for an arbitrary pair of
                # columns in the randomized data
                multnumers = abs(SumRandRanks[:, 0] - SumRandRanks[:, 1])
                # calculate pairwise t (similar to calculation of randT1)
                posvals = (multdenoms > 0)
                pairwise = np.full(nreps, nan)
                pairwise[logical_and(multdenoms == 0, multnumers>0)] = float(
                    np.inf)
                pairwise[logical_and(multdenoms==0, multnumers==0)] = 0
                pairwise[posvals] = multnumers[posvals]/multdenoms[posvals]

                # Set the distribution property.
                # The string 'randomized' will tell the next function what to
                # do. The next list item stores the number of resamplings
                # (nreps); then the values of T1 for the resamplings; then the
                # values of |Rj-Ri|/... for multple comparisons
                self.distribution = ['randomized', ['nresamples', nreps],
                                     ['randomized T1s', randT1],
                                     ['pairwise', pairwise]]
                self.pairwisePs = None

        except:
            self.P = None
            self.distribution = None
            self.pairwisePs = None
            print('Error in rfriedman')

    def multcompare(self):
        """
        Do multiple comparisons following the Friedman test.

        Uses either of the following:
            For the Friedman test as described in Conover 1999, use the method
            discussed in Conover 1999.
            For the randomization version, use the same test statistic as in
            Conover 1999, but compare to an empirical distribution of that
            test statistic based on randomization (output of self.rfriedman()).

                # STILL NEEDS:
                Check with simulations that it gives appropriate values.
                Check for small sample sizes or large treatment numbers.

        Parameters
        ----------
        none

        Sets property
        -------------
        self.pairwisePs : List
                First elements of list give strings with descriptions;
                remaining elements of list give column indices for pairs of
                treatments, followed by P values

        Returns
        -------
        none
        """
        df = (self.nblocks-1)*(self.nts-1)
        denom = (2 *
                    (self.nblocks * self.sumsqrranks - sum(
                            self.summedranks ** 2)) / df) ** (1/2)    
        pairwisePs = [[
                'multiple comparisons between pairs of treatments'],
                ['treatment', 'treatment', 'two-tailed P']]
        for k in range(self.nts):
            for l in range(k+1, self.nts):
                if denom > 0:
                    tobs = abs(self.summedranks[k] - self.summedranks[l])/denom
                # Special cases if denominator goes to zero.
                elif denom == 0:
                    if abs(self.summedranks[k] - self.summedranks[l]) > 0:
                        tobs = float(np.inf)
                    elif abs(self.summedranks[k] - self.summedranks[l]) == 0:
                        tobs = 0
                    else:
                        tobs = np.nan
                        
                else:
                    tobs = np.nan

                # k & l indices of columns for treatments (start at 0);
                # 2-tailed P value = 2*(1-t.cdf(tobs, df))
                if self.distribution is None:
                    pairwisePs = 'No distribution property: run overall test.'
                elif self.distribution[0] == 'f.cdf':
                    pairwisePs = (
                        pairwisePs + [[k, l, 2*(1-t.cdf(tobs, df))]])
                elif (self.distribution[0] == 'randomized' and
                        self.distribution[3][0] == 'pairwise'):
                    # For this distribution treated as 1-sided test because
                    # used absolute value for both the statistic (which should
                    # approximate a t-statistic) and the values for the
                    # statistic from the randomized data. Therefore, only the
                    # upper tail of the distribution stored in
                    # self.distribution[3][1] (i.e. 'pairwise') represents
                    # extreme values.
                    pairwisePs = (
                                    pairwisePs + [[k, l, sum(
                                        tobs <= self.distribution[3][1]) /
                                            len(self.distribution[3][1])]]  )
                else:
                    pairwisePs = pairwisePs + ['Distribution not as expected']              
        self.pairwisePs = pairwisePs

    def printfriedstuff(self, verbose=False):
        """
        Prints friedstat object

        Cleaner printing of friedmanstat object attributes
        verbose = True: returns all attributes, but gives percentiles of the
        distributions for the randomization tests.

        Parameters:
        ----------
        verbose : boolean, default is True
                Whether or not to print all information
        """
        print('P = ', self.P)
        print('distribution method used: ', self.distribution[0])
        if self.distribution[0] == 'randomized':
            print(self.distribution[1])
        print('n blocks: ', self.nblocks, '; n treatments: ', self.nts)

        if self.pairwisePs is not None:
            print('\n', self.pairwisePs[0])
            print(self.pairwisePs[1])
            print(array(self.pairwisePs[2:]))

        if verbose:
            print('\nStandard Friedman statisic: T1 = ', self.T1)
            print('Improved statistic (Iman & Davenport 1980): ', self.T2)
            print('summed ranks')
            print(self.summedranks)
            print('grand sum of squared ranks: ', self.sumsqrranks)
            print('correction for ties: ', self.tiecorrection)
            print('\nData')
            print(self.mydata)
            if self.distribution[0] == 'randomized':
                print('\nDistribution of T1 (randomization)')
                print('min, 1st quartile, median, 3rd quartile, max')
                print(np.percentile(
                    self.distribution[2][1], [0, 25, 50, 75, 100]))
                print('\nDistribution of pairwise statistic (randomization)')
                print('min, 1st quartile, median, 3rd quartile, max')
                print(np.percentile(
                    self.distribution[3][1], [0, 25, 50, 75, 100]))

