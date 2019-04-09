import fractions
import functools
import numpy
import os
import shelve
import simulation
import time
import typing

import alg1_world

def allocations(cItem, cBucket, _start=None):
    """Given cItems indestinguishable items, return the list of all possible ways
    that they can be distinguishably allocated among cBuckets
    distinguishable buckets.

    Each entry in the returned list is another list that contains cBucket
    integers, with each integer representing the number of items in the
    corresponding bucket for that allocation.

    Not all items are necessarily allocated.

    e.g. allocations(2, 2) yields:
    [(0, 0),
    (1, 0), (0, 1),
    (2, 0), (1, 1), (0, 2)]

    _start -- for internal use only. A tuple that is added to all the
    allocations.

    """
    assert(cBucket > 0)
    assert(cItem >= 0)
    if cItem == 0:
        assert(_start is not None)
        gcd = functools.reduce(fractions.gcd, _start)
        if (gcd != 0): #equivalently, _start is the zero vector
            _start = tuple(weight // gcd for weight in _start)
        return set([_start])

    if _start is None:
        _start = (0,)*cBucket

    ret = allocations(cItem-1, cBucket, _start=_start)
    for i in range(cBucket):
        tmp = _start[:i] + (_start[i]+1,) + _start[i+1:]
        ret = ret.union(allocations(cItem-1, cBucket, _start=tmp))

    return ret

def computePercentiles(simulation: simulation.simulation, actions: typing.List[typing.Tuple[float]],
                       log=True, percentiles=range(0,101,1), cTrials=1000) -> typing.List[typing.Tuple[float]]:
    """ret[i][j] is the percentiles[j]-th percentile of the action actions[i]"""
    results = numpy.empty((cTrials, len(actions)))
    if log:
        print("Computing Quantiles")
    for i in range(cTrials):
        if log:
            print(f"Trial {i} / {cTrials}\r", end="")
        for j in range(len(actions)):
            results[i,j] = simulation.computeCost(actions[j])
        simulation.reinitialize()
    if log:
        print("")
    return numpy.percentile(results, percentiles).transpose()

def getStandardPercentiles(simulation: simulation.simulation):
    return alg1_world.getAlgOnePercentiles(simulation)

def getCachedVariable(dbfile, varName, constructor, depFNames=[]):
    kTime = varName + "_mTime"
    with shelve.open(dbfile) as shelf:
        if not (varName in shelf and kTime in shelf):
            shelf[kTime] = time.time()
            shelf[varName] = constructor()

        mTime = shelf[kTime]
        for fName in depFNames:
            if os.path.getmtime(fName) > mTime:
                print(f"WARNING: {fName} has been modified since {varName} was computed")

        return shelf[varName]
