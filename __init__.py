"""
Run two versions of Friedman test with multiple comparisons.

The first version is based on Conover 1999 'Practical Non-Parameteric
Statistics, 3rd Ed. (Following Iman and Davenport 1980). As an alternative for
small sample sizes, there is a randomization implementation of Friedman's test,
based on Manly's 'Randomization, Bootstrap,...'1997.

The Friedman test tests whether there is a difference among k treatments
(columns) where data is grouped into b similar blocks (e.g. a colony to which
several treatments were applied to different zooids). Specifically, the null
hypothesis is that all rankings of data within a block are equally likely

Functions and classes
---------------------
friedmanstat : class
    Class used doing different versions of Friedman test
    methods: __init()__, cfriedman(), rfriedman(), multcompare(), & 
        printfriedstuff()
resamplealongrows : function
    Resample an array along rows only; used to test Friedman test performance
bootsample2D : function NOT USED BUT LEFT IN IN CASE USEFUL LATER
    Randomly resample an array once; unused
bootset2D : function NOT USED BUT LEFT IN IN CASE USEFUL LATER
    Resamples an array multiple times (vectorized), maintaining organization of
    data grouped in blocks (resamples blocks, then resamples within blocks)
randomizeAlongRows : function
    Randomize an array multiple times (vectorized) used for rfriedman()
    method of friedmanstat; randomly permutes order along rows
rankalongrows3D : function
    Rank along the rows of a 3D numeric array using average value for rank of
    ties. Used by rfriedman() method of friedmanstat.
friedmantest : function
    Run friedman test. Combines various steps using the class methods.

Dependencies
------------
numpy
numpy.stats
numpy.random
scipy.stats

Created on Thu Feb 18 11:10:29 2016

@author: Michelangelo
"""
from friedtestfuns import friedmantest, friedmanstat
from friedtestfuns import randomizeAlongRows, rankalongrows3D, resamplealongrows
from friedtestfuns import bootsample2D, bootset2D

from FriedValidation import TestFriedStuff, SimFriedStuff
