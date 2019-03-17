from __future__ import annotations

import random
import task

class equipment:
    def __init__(self, power, gain, task_, frequency, energyPerCycle):
        self._power = power
        self._gain = gain
        self._task = task_
        self._frequency = frequency
        self._energyPerCycle = energyPerCycle
        self._N0 = N0
