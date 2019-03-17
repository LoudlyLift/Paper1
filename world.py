import config
import math
import equipment
import numpy

class world:
    def __init__(self, bandwidth: float, cEquipment: int, N0: float):
        assert(cEquipment>0)
        self._bandwidth = bandwidth
        self._N0 = N0
        self._cEquipment = cEquipment

        self._equipment = [config.newEquipment() for i in range(cEquipment)]

    def _compute_upload_rate(world, effective_bandwidth, equipment):
        """Computes the unscaled upload rate"""
        num=eq._power*eq._gain
        dem=effective_bandwidth*world._N0
        quotient=num/dem

        #TODO: log_e, I presume?
        return effective_bandwidth * math.log(1+quotient)

    def _compute_upload_rates(self, offloaders):
        cOffload = len(offloaders)
        effective_bandwidth = self._bandwidth / cOffload

        fun = lambda eq: self._compute_upload_rate(self,
                                                   effective_bandwidth=effective_bandwidth,
                                                   eq)

        return list(map(fun, offloaders))
