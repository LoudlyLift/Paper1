import config
import math
import equipment

class world:
    def __init__(self, bandwidth: float, cEquipment: int, mec_clockspeed: float,
                 N0: float):
        assert(cEquipment>0)
        self.bandwidth = bandwidth
        self.mec_clockspeed = mec_clockspeed
        self.N0 = N0
        self.cEquipment = cEquipment

        self.equipment = [config.newEquipment(self) for i in range(cEquipment)]


    def computeCost(self, shouldOffloads, allocationWeights) -> float:
        """shouldOffloads is a boolean list of length cEquipment stating whether or not
        a particular task should be offloaded.

        allocationWeights is a list of what fraction of the MEC server's CPU a
        given task should get
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
            in zip(self.equipment, shouldOffloads, allocationWeights):
            allocatedFreq = (allocationWeight / totalWeight) * \
                self.mec_clockspeed
            total += isOffloaded * equipment.cost_offload(cOffloaded,
                                                          allocatedFreq)
            total += (1-isOffloaded) * equipment.cost_local()

        return total
