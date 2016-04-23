# FriedTests
Two ways to do the Friedman test.

This provides a class and other functions to run two versions of Friedman test with multiple comparisons.

The first version is based on Conover 1999 'Practical Non-Parameteric Statistics, 3rd Ed. (Following Iman and Davenport 1980). 
As an alternative for small sample sizes, there is a randomization implementation of Friedman's test.
Multiple comparisons (following the overall test) were implemented as described in Conover 1999 for the first version of the Friedman test.
Multiple comparisons were implemented using the same statistic, but compared to a distribution based on randomization of the data, for the
randomization test version.

The Friedman test tests whether there is a difference among k treatments (columns) where data is grouped into b similar blocks 
(e.g. a colony to which several treatments were applied to different zooids). Specifically, the null hypothesis is that all 
rankings of data within a block are equally likely
