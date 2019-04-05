import random
import numpy
import math

import equipment
import simulation

def newEquipment() -> equipment.equipment:
    # values given in the paper
    power=500*1e-3
    power_waiting=100*1e-3
    freq=1*1e9
    energyPerCycle=1e-27 * (freq**2)
    timeenergy_ratio = 0.5
    cbInput = random.randint(300, 500) * 1000
    cCycle = random.randint(900, 1100) * 1000000

    maxDistance = 200
    distance = maxDistance * math.sqrt(random.random())
    posTheta = random.random() * 2 * math.pi

    #TODO: the paper doesn't specify how to initalize these?
    gain=numpy.random.rayleigh(distance)
    sDelayMax = random.random() * 30 + 15

    return equipment.equipment(power=power, power_waiting=power_waiting,
                               gain=gain, frequency=freq,
                               energyPerCycle=energyPerCycle,
                               timeenergy_ratio=timeenergy_ratio,
                               cbInput=cbInput, cCycle=cCycle,
                               sDelayMax=sDelayMax)


def newSimulation() -> simulation.simulation:
    #given
    bandwidth=10e6
    mec_clockspeed=5e9

    #idk
    N0 = 123

    #TMP
    cEquipment = 7

    return simulation.simulation(bandwidth=bandwidth, cEquipment=cEquipment,
                                 mec_clockspeed=mec_clockspeed, N0=N0,
                                 consEquipment=newEquipment)

qtableConfig={"learning_rate": 0.3}

future_discount=0.99

world_config_num_cost_buckets=10
