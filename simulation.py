import typing
import config
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
        consEquipment -- a constructor that returns an equipment; first arg is world, keyword arg Args=consEquipmentArgs.
        """
        assert(cEquipment>0)
        self.bandwidth = bandwidth
        self.mec_clockspeed = mec_clockspeed
        self.N0 = N0
        self.cEquipment = cEquipment
        self.consEquipment = consEqupment
        self.consEquipmentArgs = consEquipmentArgs

        self.reinitialize()

    def reinitialize(self):
        """Start a new scenario; reinitialize random variables.

        """
        self._equipment = [self.consEquipment(self, self.consEquipmentArgs) for i in range(self.cEquipment)]

    def getEquipment(self, i):
        """Returns the i-th equipment
        """
        return self._equipment[i]

    def computeCost(self, allocationWeights) -> float:
        """Computes the cost of allocating the MEC's CPU in the given proportion for the
        current scenario.

        allocationWeights is a list of weights of what fraction of the MEC
        server's CPU a given task should get. Zero means compute
        local. Otherwise it gets (weight / sum(allocationWeights)) of the MEC's
        CPU.

        """

        cOffloaded = 0
        totalWeight = 0
        for (isOffloaded, weight) in zip(shouldOffloads, allocationWeights):
            if isOffloaded:
                cOffloaded += 1
                totalWeight += weight

        #eq. (12), only with weights instead of ƒ
        total = 0
        for (equipment, isOffloaded, allocationWeight) \
            in zip(self._equipment, shouldOffloads, allocationWeights):
            allocatedFreq = (allocationWeight / totalWeight) * \
                self.mec_clockspeed
            total += isOffloaded * equipment.cost_offload(cOffloaded,
                                                          allocatedFreq)
            total += (1-isOffloaded) * equipment.cost_local()

        return total