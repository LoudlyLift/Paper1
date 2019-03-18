from __future__ import annotations

import random
import task
import world

class equipment:
    def __init__(self, world_: world.world, power, gain, task_, frequency,
                 energyPerCycle):
        self.power = power
        self.gain = gain
        self.task_ = task_
        self.frequency = frequency
        self.energyPerCycle = energyPerCycle
        self.N0 = N0
        self.world_ = world_

    def _compute_upload_rate(self, effective_bandwidth):
        """Computes the unscaled upload rate using the formula given in eq. (1)"""
        num=self._power*self._gain
        dem=effective_bandwidth*self.world_._N0
        quotient=num/dem

        #TODO: log_e, I presume?
        return effective_bandwidth * math.log(1+quotient)

    def cost_local(self):
        #eq. (2)
        time   = self.task_.cCycle / self.frequency
        #eq. (3)
        energy = self.task_.cCycle * self.energyPerCycle

        #eq. after (3) but before (4)
        return time*self.task_.timeenergy_ratio \
            + energy*(1-self.task_.timeenergy_ratio)

    def cost_offload(self, cOffloaders: int, fracProcessor):
        effective_bandwidth = self.world_.bandwidth / cOffloaders

        # eq. (4)
        time_offload = self.task_.cbInput / \
            self._compute_upload_rate(effective_bandwidth)

        # eq. (5)
        energy_offload = self.power * time_offload

        # eq. (6)
        time_processing = self.fracProcessor *
