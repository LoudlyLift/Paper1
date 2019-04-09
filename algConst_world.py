import fractions
import functools
import math
import numpy
import random
import typing

import simulation
import world_helper
import alg1_world

ALGCOST_DBFILE = "./Persist/algConst"

"""This trivial algorithm always takes the same action. That action is chosen by
performing all possible actions on a number of randomized simulations and taking
the action that had the optimal median performance.
"""
class algConst_world(alg1_world.alg1_world):
    def determineBestAction(self, log=True, cTrials=1000) -> typing.List[typing.Tuple[float]]:
        """ret[i][j] is the percentiles[j]-th percentile of the action actions[i]"""
        results = numpy.empty((cTrials, len(self.possibleActions)))
        if log:
            print("Determining Best Action")
        for i in range(cTrials):
            if log:
                print(f"Trial {i} / {cTrials}\r", end="")
            for j in range(len(self.possibleActions)):
                results[i,j] = self.simulation.computeCost(self.possibleActions[j])
            self.simulation.reinitialize()
        if log:
            print("")
        results = numpy.median(results, axis=0)
        return self.possibleActions[numpy.argmin(results)]

    def __init__(self, simulation: simulation.simulation, action=None, cTrials=1000):
        super().__init__(simulation)

        if action is None:
            action = world_helper.getCachedVariable(ALGCOST_DBFILE, "bestAction",
                                                    lambda: self.determineBestAction(cTrials=1000),
                                                    depFNames=["simulation.py", "equipment.py"])

        self.action = action
        print(f"Using the action: {self.action}")

    def reset(self):
        self.simulation.reinitialize()

        return tuple(0 for count in self.getStateMetadata())

    def getRandomMove(self):
        return 0

    def step(self, _):
        self._prior_cost = self.simulation.computeCost(self.action)

        reward = 0
        done = True
        state = self.getState()

        return (state, reward, done)

    def getState(self):
        return (0,)

    def getStateMetadata(self):
        return (1,)

    def getNumActions(self):
        return 1

    def getLegalMoves(self):
        return [True] * len(self.possibleActions)

    def closeEpisode(self):
        quantile = numpy.searchsorted(self.percentiles, self._prior_cost) / (len(self.percentiles)-1)
        self.localCost = self.simulation.computeCost([0] * self.simulation.cEquipment)

        return {"min": self.minCost, "max": self.maxCost, "local": self.localCost, "actual": self._prior_cost, "quantile": quantile}
