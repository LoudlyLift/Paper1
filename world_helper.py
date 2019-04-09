import numpy
import os
import shelve
import simulation
import time
import typing

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
