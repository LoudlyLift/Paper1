import math
import numpy
import random
import typing

import equipment
import simulation

def newEquipment(args=None) -> equipment.equipment:
    # values given in the paper
    power=500*1e-3
    power_waiting=100*1e-3
    freq=1*1e9
    energyPerCycle=1e-27 * (freq**2)
    timeenergy_ratio = 0.5
    cbInput = random.randint(300, 500) * 1000
    cCycle = random.randint(900, 1100) * 1000000

    maxDistance = 200
    distance = maxDistance * math.sqrt(random.random())
    posTheta = random.random() * 2 * math.pi

    #TODO: the paper doesn't specify how to initalize these?
    gain=numpy.random.rayleigh(distance)
    sDelayMax = 500 #when offloading, just uploading can take 25+ seconds
                    #depending on randomness... (All tasks can be processed
                    #locally in < 1.5 sec...)

    return equipment.equipment(power=power, power_waiting=power_waiting,
                               gain=gain, frequency=freq,
                               energyPerCycle=energyPerCycle,
                               timeenergy_ratio=timeenergy_ratio,
                               cbInput=cbInput, cCycle=cCycle,
                               sDelayMax=sDelayMax)

class SmartSimulation(simulation.simulation):
    @staticmethod
    def weightedDistribution(additional: float, initialBuckets: typing.List[float], weights: typing.List[float]) -> typing.List[float]:
        """Metaphorically speaking, we have a list of buckets that each contains some
        quantity of water. We want the proportion of water in each bucket to
        match the proportions given in the weights list, but we cannot remove
        any water from the buckets. We have `additional` units of extra water,
        which we can add to any of the buckets. We must use all of this extra
        water.

        More formally:

        Args:

        -- additional: the number that is distributed among the buckets; additional ≥ 0

        -- initialBuckets: the initial values; buckets[i] ≥ 0 ∀ i

        -- weights: the desired ratio between the elements of buckets; weights[i] > 0 ∀ i

        let weightedQuantity = numpy.divide(result, weights)

        The list we return is the (unique) solution to this optimization
        problem:

        minimize (max(weightedQuantity) - min(weightedQuantity)) such that:

        => sum(result) == additional + sum(initialBuckets)

        => result[i] ≥ initialBuckets[i] ∀ i
        """
        buckets = numpy.array(initialBuckets, dtype=float)
        weights = numpy.array(weights, dtype=float)

        THRESHOLD = 1e-9

        assert(numpy.count_nonzero(buckets) == len(buckets))

        while additional > 0:
            ratio = buckets/weights

            minRatio = min(ratio)

            nonminRatios = [r for r in ratio if r > minRatio + THRESHOLD]
            if len(nonminRatios) == 0:
                targetRatio = math.inf
            else:
                targetRatio = min(nonminRatios)

            deltaRatio = targetRatio - minRatio

            selectedIndicies = numpy.where(ratio < minRatio + THRESHOLD)[0]
            sumWeights = sum(weights[i] for i in selectedIndicies)

            usedWater = deltaRatio * sumWeights
            usedWater = min(usedWater, additional)

            for i in selectedIndicies:
                buckets[i] += usedWater * (weights[i] / sumWeights)
            additional -= usedWater

        return buckets

    def computeCost(self, allocationWeights: typing.List[float]) -> float:
        # rescale such that weights sum to one
        initial_sum = sum(allocationWeights)
        assert(initial_sum >= 0)
        if initial_sum != 0:
            allocationWeights = [ v / initial_sum for v in allocationWeights ]

        forceOffload = [ False ] * self.cEquipment
        for i in range(self.cEquipment):
            eq = self._equipment[i]

            if allocationWeights[i] > 0 or eq.local_processing_time() > eq.sDelayMax:
                forceOffload[i] = True

        cOffload = sum(forceOffload)
        if cOffload == 0:
            return simulation.simulation.computeCost(self, allocationWeights)
        allocatedBandwidth = self.bandwidth / cOffload

        minFrac = [ 0.0 ] * self.cEquipment
        for i in range(self.cEquipment):
            if not forceOffload[i]:
                minFrac[i] = 0
                continue
            eq = self._equipment[i]
            upload_rate = eq.upload_rate(allocatedBandwidth, self.N0)
            max_proccessingSeconds = (eq.sDelayMax - (eq.cbInput/upload_rate))
            assert(max_proccessingSeconds > 0)
            minHtz = eq.cCycle / max_proccessingSeconds
            minFrac[i] = minHtz / self.mec_clockspeed

        offloadIndicies = numpy.array(forceOffload).nonzero()[0]

        offloadMins = [ minFrac[i] for i in offloadIndicies ]
        offloadWeights = [ allocationWeights[i] for i in offloadIndicies ]

        min_sum = sum(offloadMins)
        assert(min_sum <= 1)

        offloadWeights = SmartSimulation.weightedDistribution(1 - min_sum, offloadMins, offloadWeights)

        for (iOffload, iGlobal) in enumerate(offloadIndicies):
            allocationWeights[iGlobal] = offloadWeights[iOffload]

        return simulation.simulation.computeCost(self, allocationWeights)

def newSimulation() -> simulation.simulation:
    #given
    bandwidth=10e6
    mec_clockspeed=5e9

    #idk
    N0 = 1e-4

    #TMP
    cEquipment = 7

    return SmartSimulation(bandwidth=bandwidth, cEquipment=cEquipment,
                                 mec_clockspeed=mec_clockspeed, N0=N0,
                                 consEquipment=newEquipment)

qtableConfig={"learning_rate": 0.3}

future_discount=0.5

world_config_num_cost_buckets=10
