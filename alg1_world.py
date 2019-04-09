import math
import numpy
import random

import simulation
import world_helper

ALG1_DBFILE = "./Persist/alg1"

def getAlgOneActions(cEquipment: int):
    return world_helper.getCachedVariable(ALG1_DBFILE, f"possibleActions_{cEquipment}",
                                          lambda: list(world_helper.allocations(cItem=cEquipment, cBucket=cEquipment)),
                                          depFNames=["alg1_world.py"])

def getAlgOnePercentiles(simulation):
    actions = getAlgOneActions(simulation.cEquipment)
    return world_helper.getCachedVariable(ALG1_DBFILE, f"percentiles_{simulation.cEquipment}",
                                                      lambda: world_helper.computePercentiles(simulation, actions),
                                                      depFNames=["simulation.py", "equipment.py"])


"""Wrapper around simulation.py for Algorithm 1 as described in the paper

NOTE: this makes the trivial optimization of rescaling MEC server CPU
allocations. e.g. if two tasks are being offloaded, and the algorithm says to
grant each of them 20% of the CPU, they will both actually get 50%."""
class alg1_world:

    def __init__(self, simulation: simulation.simulation, granularityTC=8, granularityMEC = None):
        self.simulation = simulation

        if granularityMEC is None:
            granularityMEC = simulation.cEquipment
        self.granularityMEC = granularityMEC

        self.granularityTC = granularityTC

        self.possibleActions = getAlgOneActions(simulation.cEquipment)
        self.percentiles = getAlgOnePercentiles(simulation)

        self.minCost = self.percentiles[0]
        self.maxCost = self.percentiles[100]

        #"invalid" moves are made valid by the preclassification step
        self.legalMoves = numpy.array([True] * len(self.possibleActions))


    def reset(self):
        self.simulation.reinitialize()

        self.stateLast = None
        self.iterCount = 0

        self.localCost = self.simulation.computeCost([0] * self.simulation.cEquipment)

        # alg 1 says that initial state is random ¯\_(ツ)_/¯
        metadata = self.getStateMetadata()
        return tuple(random.randint(0, count - 1) for count in metadata)

    def getRandomMove(self):
        #"invalid" moves are made valid by the preclassification step
        return random.randint(0, self.getNumActions()-1)

    def actionFromIndexaction(self, indexaction:int):
        return list(self.possibleActions[indexaction])

    def step(self, action):
        action = self.actionFromIndexaction(action)

        cost = self.simulation.computeCost(action)

        costBucket = math.floor(self.granularityTC * (cost-self.minCost) / (self.maxCost - self.minCost))
        costBucket = min(costBucket, self.granularityTC - 1)

        acBucket = self.granularityMEC - sum(action)

        state = (costBucket, acBucket)

        #eq in section 4.A, under "reward" bullet
        reward = (self.localCost - cost) / (self.localCost)

        done = (self.iterCount == 1000) or (self.stateLast is not None and state == self.stateLast)

        self.stateLast = state
        self.iterCount += 1
        self._prior_cost = cost

        return (state, reward, done)

    def getStateMetadata(self):
        return (self.granularityTC, self.granularityMEC+1) #+1 because legal values are [0, granularityMEC]

    def getNumActions(self):
        return len(self.possibleActions)

    def getLegalMoves(self):
        return self.legalMoves

    def closeEpisode(self):
        quantile = numpy.searchsorted(self.percentiles, self._prior_cost) / (len(self.percentiles)-1)

        return {"min": self.minCost, "max": self.maxCost, "local": self.localCost, "actual": self._prior_cost, "quantile": quantile}
