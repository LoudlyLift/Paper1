import math
import statistics
import numpy

import world_helper
import simulation

ALGCONVENTIONAL_DBFILE = "./Persist/algConventional"

def run_trials(simulation, cTrials, frac):
    results = []

    num = round(simulation.cEquipment * float(frac))

    #import pdb; pdb.set_trace()
    for _ in range(cTrials):
        simulation.reinitialize()

        offloadability = numpy.array([ eq.cCycle / eq.cbInput for eq in simulation._equipment])
        indicies = offloadability.argsort()[-num:]

        action = numpy.zeros(simulation.cEquipment)
        action[indicies] = 1

        cost = simulation.computeCost(action)
        results.append(cost)
    return results

def computeBestFrac(simulation, fracs, cTrials=1000):
    results = []
    for frac in fracs:
        print(f"{frac}: ", end="")
        results_ = run_trials(simulation, cTrials=cTrials, frac=frac)
        result = statistics.median(results_)
        print(result)

        results.append(result)
    bestFracIndex = numpy.argmin(results)
    return fracs[bestFracIndex]

def run(simulation, cEpisodes):
    percentiles = world_helper.getStandardPercentiles(simulation)
    bestFrac = world_helper.getCachedVariable(ALGCONVENTIONAL_DBFILE,
                                              f"bestFrac_{simulation.cEquipment}",
                                              lambda: computeBestFrac(simulation, numpy.arange(0, 1.001, 0.1), cTrials=100000),
                                              depFNames=["conventional.py"])
    print(f"Using frac {bestFrac}")

    results = run_trials(simulation, cEpisodes, bestFrac)
    result = statistics.median(results)
    quantile = numpy.searchsorted(percentiles, result) / (len(percentiles)-1)

    print( "algorithm    │ actual │ quantile │")
    print( "─────────────┼────────┼──────────┼")
    print(f"conventional │ {result:6.2f} │ {quantile:8.2f} │")


    print("RESULTS:")
    print(f"Median of conventional algorithm is {result:.2f}; quantile {quantile}")
