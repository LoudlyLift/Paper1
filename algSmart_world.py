import math
import random
import numpy
import world_helper

import simulation

ALGSMART_DBFILE="./Persist/algSmart"

"""Wrapper around simulation.py for my proposed algorithm"""
class algSmart_world:
    def __init__(self, simulation: simulation.simulation, equipmentToState, equipmentStateMetadata, weights = [0,1,2,4], maxIter=35):
        self.simulation = simulation

        self.allActions = weights

        self.offloadActions = [ action for action in self.allActions if action != 0 ]

        self.percentiles = world_helper.getStandardPercentiles(simulation)

        self.minCost = self.percentiles[0]
        self.maxCost = self.percentiles[100]

        self.maxIter = maxIter

        self.equipmentToState = equipmentToState
        self.equipmentStateMetadata = equipmentStateMetadata

        #"invalid" moves are made valid by the preclassification step
        self.legalMoves = numpy.array([True] * len(self.allActions))


    def reset(self):
        self.currentVector = numpy.array([0] * self.simulation.cEquipment)
        self.currentIndex = 0
        self.iteration = 0

        self.simulation.reinitialize()

        return self.getState()

    def getRandomMove(self):
        return random.randint(0, self.getNumActions()-1)

    def step(self, action):
        self.currentVector[self.currentIndex] = self.allActions[action]
        self.iteration += 1
        self.currentIndex = (self.currentIndex+1) % len(self.currentVector) #must be updated before getState

        state = self.getState()

        done = self.iteration == self.maxIter

        if done:
            self._prior_cost = self.simulation.computeCost(self.currentVector)
            self._prior_quantile = numpy.searchsorted(self.percentiles, self._prior_cost) / (len(self.percentiles)-1)

            reward = (100 * (1-self._prior_quantile)) ** 4
        else:
            reward = 0

        return (state, reward, done)

    def getStateMetadata(self):
        return self.equipmentStateMetadata + (3,) * len(self.offloadActions)

    def getState(self):
        equipment = self.simulation.getEquipment(self.currentIndex)

        equipmentState = self.equipmentToState(equipment)

        mecState = []
        for action in self.offloadActions:
            count = (self.currentVector == action).sum()
            frac = count/self.simulation.cEquipment

            expected = len(self.allActions)

            if frac <= expected:
                value = 0
            elif frac <= 2*expected:
                value = 1
            else:
                value = 2
            mecState.append(value)
        mecState = tuple(mecState)

        return equipmentState + mecState

    def getNumActions(self):
        return len(self.allActions)

    def getLegalMoves(self):
        return self.legalMoves

    def closeEpisode(self):
        localCost = self.simulation.computeCost([0] * self.simulation.cEquipment)

        return {"min": self.minCost, "max": self.maxCost, "local": localCost, "actual": self._prior_cost, "quantile": self._prior_quantile}
