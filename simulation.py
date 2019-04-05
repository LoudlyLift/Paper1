import typing
import math
import equipment


"""Constructed once, re-randomized for each run.
"""
class simulation:
    def __init__(self, bandwidth: float, cEquipment: int, mec_clockspeed: float,
                 N0: float, consEquipment, consEquipmentArgs=None):
        """args:
        bandwidth -- ???
        cEquipment -- the number of UEs
        mec_clockspeed -- MEC server's CPU clockspeed in GHz
        N0 -- ???
        consEquipment -- a constructor that returns an equipment; one positional argument which is constEquipmentArgs
        """
        assert(cEquipment>0)
        self.bandwidth = bandwidth
        self.mec_clockspeed = mec_clockspeed
        self.N0 = N0
        self.cEquipment = cEquipment
        self.consEquipment = consEquipment
        self.consEquipmentArgs = consEquipmentArgs

        self.reinitialize()

    def reinitialize(self):
        """Start a new scenario; reinitialize random variables.

        """
        self._equipment = [self.consEquipment(self.consEquipmentArgs) for i in range(self.cEquipment)]

    def getEquipment(self, i):
        """Returns the i-th equipment
        """
        return self._equipment[i]

    def computeCost(self, allocationWeights) -> float:
        """Computes the cost of allocating the MEC's CPU in the given proportion for the
        current scenario.

        allocationWeights is a list of weights of what fraction of the MEC
        server's CPU a given task should get. Zero means compute
        local. Otherwise is offloaded using (weight / sum(allocationWeights)) of
        the MEC's CPU.

        """
        assert(len(allocationWeights) == self.cEquipment)

        cOffloaded = self.cEquipment - allocationWeights.count(0)
        totalWeight = sum(allocationWeights)

        if cOffloaded != 0:
            allocatedBandwidth = self.bandwidth / cOffloaded

        #eq. (12), only with weights instead of ƒ
        total = 0
        for (equipment, weight) in zip(self._equipment, allocationWeights):
            cost = 0
            if weight == 0:
                cost = equipment.cost_local()
            else:
                assert(weight > 0)
                assert(totalWeight >= weight)
                allocatedFreq = (weight / totalWeight) * self.mec_clockspeed
                cost = equipment.cost_offload(allocatedBandwidth, allocatedFreq, self.N0)
            total += cost

        return total
