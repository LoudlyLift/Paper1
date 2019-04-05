from __future__ import annotations

import random
import task
import world

"""

cbInput -- the byte count of the input cCycle

size -- the count of the number of CPU cycles it will take to compute this task

sDelayMax -- maximum tolerable delay of the task, in seconds

timeenergy_ratio -- the linear tradeoff to make between optimizing energy
consumption and optimizing the time until completion. One means complete the
task as soon as possible without regard to energy costs. Zero means that
equipment's task is to be completed in the most energy efficient manner possible
(as long as it is still completed within sDelayMax seconds).
"""
class equipment:
    def __init__(self, power: float, power_waiting: float, gain, frequency:
                 float, energyPerCycle: float, timeenergy_ratio: float, cbInput:
                 int, cCycle: int, sDelayMax: float):
        self.power = power
        self.power_waiting = power_waiting
        self.gain = gain
        self.frequency = frequency
        self.energyPerCycle = energyPerCycle
        self.timeenergy_ratio = timeenergy_ratio
        self.cbInput = cbInput
        self.cCycle = cCycle
        self.sDelayMax = sDelayMax

    def _compute_upload_rate(self, effective_bandwidth, N0):
        """Computes the unscaled upload rate using the formula given in eq. (1)"""
        num=self._power*self._gain
        dem=effective_bandwidth*N0
        quotient=num/dem

        #TODO: log_e, I presume?
        return effective_bandwidth * math.log(1+quotient)

    def cost_local(self):
        #eq. (2)
        time   = self.cCycle / self.frequency
        #eq. (3)
        energy = self.cCycle * self.energyPerCycle

        #eq. after (3) but before (4)
        return time*self.timeenergy_ratio \
            + energy*(1-self.timeenergy_ratio)

    def cost_offload(self, effective_bandwidth: float, cOffloaders: int,
                     allocatedClockSpeed: float):
        # eq. (4)
        time_offload = self.cbInput / \
            self._compute_upload_rate(effective_bandwidth)

        # eq. (5)
        energy_offload = self.power * time_offload

        # eq. (6)
        time_processing = self.cCycle / allocatedClockSpeed

        # eq. (7)
        energy_waiting = self.power_waiting * time_processing

        # eq. (9)
        time_total = time_offload + time_processing

        # eq. (10)
        energy_total = energy_offload + energy_waiting
