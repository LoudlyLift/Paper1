import config
import math
import equipment

class world:
    def __init__(self, bandwidth: float, cEquipment: int, N0: float):
        assert(cEquipment>0)
        self._bandwidth = bandwidth
        self._N0 = N0
        self._cEquipment = cEquipment

        self._equipment = [config.newEquipment(self) for i in range(cEquipment)]
