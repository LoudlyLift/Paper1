import math
import random
import numpy
import world_helper

import simulation

ALGSMART_DBFILE="./Persist/algSmart"

"""Wrapper around simulation.py for my proposed algorithm"""
class algSmart_world:
    def __init__(self, simulation: simulation.simulation, equipmentToState, equipmentStateMetadata, weights = [0,2,3,4], maxIter=35):
        self.simulation = simulation

        self.allActions = weights

        self.offloadActions = [ action for action in self.allActions if action != 0 ]

        self.percentiles = world_helper.getStandardPercentiles(simulation)

        self.maxIter = maxIter

        self.equipmentToState = equipmentToState
        self.equipmentStateMetadata = equipmentStateMetadata

        #"invalid" moves are made valid by the preclassification step
        self.legalMoves = numpy.array([True] * len(self.allActions))

        self.isTrainable = True

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

        self._prior_cost = self.simulation.computeCost(self.currentVector)
        self._prior_quantile = numpy.searchsorted(self.percentiles, self._prior_cost) / (len(self.percentiles)-1)

        reward = ((10 * (1-self._prior_quantile)) ** 3)/10

        return (state, reward, done)

    def getStateMetadata(self):
        return self.equipmentStateMetadata + (5,) * len(self.offloadActions)

    def getState(self):
        equipment = self.simulation.getEquipment(self.currentIndex)

        equipmentState = self.equipmentToState(equipment)

        mecState = []
        for action in self.offloadActions:
            count = (self.currentVector == action).sum()
            if count <= 2:  #3xone to one
                value = count
            elif count <= 4:#two to one
                value = 3
            else:           #three to one
                value = 4
            mecState.append(value)
        mecState = tuple(mecState)

        return equipmentState + mecState

    def getNumActions(self):
        return len(self.allActions)

    def getLegalMoves(self):
        return self.legalMoves

    def closeEpisode(self):
        return {"actual": self._prior_cost, "quantile": self._prior_quantile}
