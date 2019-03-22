import random

import task
import equipment
import world

def newWorld() -> world.world:
    #given
    bandwidth=10e6
    mec_clockspeed=5e9

    #idk
    cEquipment = 7
    N0 = 123

    return world.world(bandwidth=10e6, cEquipment=cEquipment, mec_clockspeed=mec_clockspeed, N0=N0)

def newTask() -> task.task:
    # values given in the paper
    cbInput = random.randint(300, 500) * 1000
    cCycle = random.randint(900, 1100) * 1000000

    # TODO: the paper doesn't specify how to initialize this?
    sDelayMax = 10000

    return task.task(cbInput=cbInput, cCycle=cCycle,
                     sDelayMax=sDelayMax)

def newEquipment(world) -> equipment.equipment:
    # values given in the paper
    power=500*1e-3
    power_waiting=100*1e-3
    task_=newTask()
    freq=1*1e9
    energyPerCycle=1e-27 * (freq**2)
    timeenergy_ratio = 0.5

    #unused?
    #pos=randomPosIn(radius=200m)

    #TODO: the paper doesn't specify how to initalize gain?
    gain=10000

    return equipment.equipment(world_=world, power=power,
                               power_waiting=power_waiting, gain=gain,
                               task_=task_, frequency=freq,
                               energyPerCycle=energyPerCycle,
                               timeenergy_ratio=timeenergy_ratio)

qtableConfig={"learning_rate": 0.3}

future_discount=0.99

world_config_num_cost_buckets=10
