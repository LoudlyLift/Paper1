import config
import math
import equipment

class world:
    def __init__(self, bandwidth: float, cEquipment: int, N0: float):
        assert(cEquipment>0)
        self.bandwidth = bandwidth
        self.N0 = N0
        self.cEquipment = cEquipment

        self.equipment = [config.newEquipment(self) for i in range(cEquipment)]
