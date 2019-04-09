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
class algFullLocal_world(alg1_world.alg1_world):
    def __init__(self, simulation: simulation.simulation):
        super().__init__(simulation)

        self.action_local = (0,) * self.simulation.cEquipment
        self.legal_moves = [True] * self.getNumActions()

    def reset(self):
        self.simulation.reinitialize()

        return self.getState()

    def getRandomMove(self):
        return 0

    def step(self, _):
        self._prior_cost = self.simulation.computeCost(self.action_local)

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
        return self.legal_moves

    def closeEpisode(self):
        quantile = numpy.searchsorted(self.percentiles, self._prior_cost) / (len(self.percentiles)-1)
        self.localCost = self.simulation.computeCost([0] * self.simulation.cEquipment)

        return {"min": self.minCost, "max": self.maxCost, "local": self.localCost, "actual": self._prior_cost, "quantile": quantile}
