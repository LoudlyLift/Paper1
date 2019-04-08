import math
import random
import numpy

import simulation


"""Wrapper around simulation.py for my proposed algorithm"""
class algSmart_world:
    def getAllActionVectors(self, _index=0, _baseVector=()):
        assert(_index <= self.simulation.cEquipment)

        if _index == self.simulation.cEquipment:
            return [_baseVector]

        allActionVectors = []
        for action in self.allActions:
            allActionVectors += self.getAllActionVectors(_index=_index+1, _baseVector=_baseVector+(action,))
        return allActionVectors

    def __init__(self, simulation: simulation.simulation, equipmentToState, equipmentStateMetadata, weights = [0,1,2,4], maxIter=35):
        self.simulation = simulation

        self.allActions = weights
        self.allActionVectors = self.getAllActionVectors()
        self.offloadActions = [ action for action in self.allActions if action != 0 ]

        self.maxIter = maxIter

        self.equipmentToState = equipmentToState
        self.equipmentStateMetadata = equipmentStateMetadata

    def reset(self):
        self.currentVector = numpy.array([0] * self.simulation.cEquipment)
        self.currentIndex = 0
        self.iteration = 0

        self.simulation.reinitialize()

        self.localCost = self.simulation.computeCost([0] * self.simulation.cEquipment)

        self.minCost = math.inf
        self.maxCost = 0.0
        for actionVector in self.allActionVectors:
            cost = self.simulation.computeCost(list(actionVector))
            self.minCost = min(self.minCost, cost)
            self.maxCost = max(self.maxCost, cost)

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
            reward = -self._prior_cost
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
        return [True] * len(self.allActions)

    def closeEpisode(self):
        return {"min": self.minCost, "max": self.maxCost, "local": self.localCost, "actual": self._prior_cost}
