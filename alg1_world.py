import math
import numpy
import random

import simulation


"""Wrapper around simulation.py for Algorithm 1 as described in the paper

NOTE: this makes the trivial optimization of rescaling MEC server CPU
allocations. e.g. if two tasks are being offloaded, and the algorithm says to
grant each of them 20% of the CPU, they will both actually get 50%."""
class alg1_world:
    @staticmethod
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
            return set([_start])

        if _start is None:
            _start = (0,)*cBucket

        ret = alg1_world.allocations(cItem-1, cBucket, _start=_start)
        for i in range(cBucket):
            tmp = _start[:i] + (_start[i]+1,) + _start[i+1:]
            ret = ret.union(alg1_world.allocations(cItem-1, cBucket, _start=tmp))

        return ret

    def __init__(self, simulation: simulation.simulation, granularityTC=8, granularityMEC = None):
        self.simulation = simulation

        if granularityMEC is None:
            granularityMEC = simulation.cEquipment
        self.granularityMEC = granularityMEC

        self.granularityTC = granularityTC

        self.possibleActions = list(alg1_world.allocations(cItem=simulation.cEquipment, cBucket=simulation.cEquipment))
        random.shuffle(self.possibleActions)

    def reset(self):
        self.simulation.reinitialize()

        self.stateLast = None
        self.iterCount = 0

        self.localCost = self.simulation.computeCost([0] * self.simulation.cEquipment)

        self.costs = numpy.empty(len(self.possibleActions))
        for (i, act) in enumerate(self.possibleActions):
            self.costs[i] = self.simulation.computeCost(list(act))


        self.minCost = self.costs.min()
        self.maxCost = self.costs.max()

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
        #"invalid" moves are made valid by the preclassification step
        return [True] * len(self.possibleActions)

    def closeEpisode(self):
        actual = self._prior_cost
        percentile = 100*numpy.sum(self.costs < actual) / self.costs.size

        return {"min": self.minCost, "max": self.maxCost, "local": self.localCost, "actual": actual, "percnt": percentile}
