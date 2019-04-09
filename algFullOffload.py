import fractions
import functools
import math
import numpy
import random
import typing

import simulation
import algConst_world

"""This trivial algorithm always runs all tasks locally.
"""
class algFullOffload_world(algConst_world.algConst_world):
    def __init__(self, simulation: simulation.simulation):
        super().__init__(simulation, action=(1,) * simulation.cEquipment)
