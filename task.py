# so that we can reference the class by name within the class itself
from __future__ import annotations

import random

class task:
    """Has these attributes:

    cbInput: the byte count of the input size

    cCycle: the count of the number of CPU cycles it will take to compute this
        task

    sDelayMax: maximum tolerable delay of the task, in seconds

    timeenergy_ratio: if one, this task is to be completed as soon as possible
        without regard to energy costs. If zero, this task is to be completed in
        the most energy efficient manner possible (as long as it is still
        completed within sDelayMax seconds)
    """
    def __init__(self, cbInput: int, cCycle: int, sDelayMax: float,
                 timeenergy_ratio: float):
        self.cbInput = cbInput
        self.cCycle = cCycle
        self.sDelayMax = sDelayMax
        self.timeenergy_ratio = timeenergy_ratio
