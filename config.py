import random

import task
import equipment
import world

def newWorld() -> world.world:
    #given
    bandwidth=10e6

    #idk
    cEquipment = 7
    N0 = 123

    return world.world(bandwidth=10e6, cEquipment=cEquipment, N0=N0)

def newTask(world) -> task.task:
    # values given in the paper
    cbInput = random.randint(300, 500) * 1000
    cCycle = random.randint(900, 1100) * 1000000
    timeenergy_ratio = 0.5

    # TODO: the paper doesn't specify how to initialize this?
    sDelayMax = 10000

    return task.task(world_=world, cbInput=cbInput, cCycle=cCycle,
                     sDelayMax=sDelayMax, timeenergy_ratio=timeenergy_ratio)

def newEquipment(world) -> equipment.equipment:
    power=500*1e-3
    task_=newTask(world)
    freq=1*1e9
    energyPerCycle=1e-27 * (freq**2)

    #unused?
    #pos=randomPosIn(radius=200m)

    #TODO: the paper doesn't specify how to initalize gain?
    gain=10000

    return equipment.equipment(power=power, gain=gain, task_=task_,
                               frequency=freq, energyPerCycle=energyPerCycle)
