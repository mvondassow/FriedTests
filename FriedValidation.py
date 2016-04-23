"""
Validation tests and simulation functions for FriedTests

Moved to separate file on Thu Apr 23, 2016

@author: Michelangelo
"""
from scipy.stats import rankdata, f, t
from numpy import asarray, sum, empty, indices, array, isnan, logical_and, nan
from numpy.random import randint, random
from time import clock
import numpy as np

def TestFriedStuff():
    """
    Function to run several tests on functions for Friedman test.
    
    Parameters
    ----------
    none
    
    Returns
    -------
    none
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

    print('number of resamples: ', nboot)
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
    print('Test randomization version (bfriedman() of Friedman test')
    maxb = 10000
    print('p value for grass data from Conover; nreps = ', maxb)
    friedgrass.rfriedman(nreps=maxb)
    print(friedgrass.P)
    print(vars(friedgrass))


def SimFriedStuff(nblocks=5, nts=3, distfunc=None, variant='Rand', nsim=1000,
                  nreps=1000, alpha=0.05, verbose=True):
    """
    Run simulations to test Friedman test and multiple comparisons
    
    Parameters
    ----------
    nblocks : int
        number of blocks
    nts : int
        number of treatments
    distfunc : function
        Transforms a 2 dimensional or 3 dimensional array of uniformly
        distributed random numbers (on range [0,1]) into user-defined
        distribution (e.g. rounding to generate ties, etc.)
    variant : string
        'Rand' or 'Fdist'
    nsim : int
        number of trials for the simulation
    nreps : int
        number of resamplings to do if using 'Rand' variant

    Returns
    -------
    none
    """
    t = clock()
    simdata = random((nblocks, nts))
    if distfunc is not None:
        try:
            simdata = distfunc(simdata)
        except:
            print(
                'distfunc is not a valid function on multidimensional arrays.'
                'Continuing using uniform distribution')

    temp = friedmantest(simdata, variant, nreps, verbose=False)
    testduration = round(nsim*(clock()-t))
    response = input(
        'This will take ~'+str(testduration)+' s. Continue? [y/n]')

    if response is 'n' or response is 'N':
        print('Simulation brutally cut down in its prime by the user')
    elif response is 'y' or response is 'Y':
        simdata = random((nsim, nblocks, nts))
        GroupTest = np.full(nsim, nan)
        for k in range(nsim):
            temp = friedmantest(
                simdata[k, :, :], variant, nreps, verbose=False)
            GroupTest[k] = temp.P<=alpha
            
        print('Percentile with P <= ' +str(alpha) + ': ' + 
            str(sum(GroupTest)/nsim))
    else:
        print('Invalid response. Simulation not done.')
