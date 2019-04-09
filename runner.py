import argparse
import math
import numpy
import random
import statistics
import sys
import time

import algSmart_world
import alg1_world
import algConst_world
import config
import equipment
import qlearning
import qtable

parser = argparse.ArgumentParser(description='Run edge computing simulations')
parser.add_argument('algorithm', choices=['one', 'smart', 'const'])
parser.add_argument('--train-episodes', type=int, default=10000, help="How many episodes to perform during training")
parser.add_argument('--eval-episodes', type=int, default=1000, help="How many episodes to perform during evaluation")
parser.add_argument('--future-discount', type=float, default=0.5, help="How much the Q-Table will value the next turn's reward")
parser.add_argument('--equipment-count', type=int, default=7, help="How much UE's the simulation will have")
parser.add_argument('--learning-rate', type=float, default=0.3, help="The Q-Table updates with a decaying moving average. This is the weight of the most recent observation")
parser.add_argument('--bandwidth', type=float, default=10e6, help="The total bandwidth that is shared by all the transmitters (Hz)")
parser.add_argument('--mec-clockspeed', type=float, default=5e9, help="The clockspeed of the MEC server's CPU (Hz)")
parser.add_argument('--n0', type=float, default=1e-4, help="???")

#equipment config
parser.add_argument('--equipment-power', type=float, default=500e-3, help="The transmission power of equipment transmitters")
parser.add_argument('--equipment-power-idle', type=float, default=100e-3, help="The power of equipment transmitters when idle")
parser.add_argument('--equipment-clockspeed', type=float, default=1e9, help="The equipment's CPU's clockspeed (Hz)")
parser.add_argument('--equipment-timeenergy-ratio', type=float, default=0.5, help="1 to optimize for time, 0 to optimize for energy, intermediate values are a linear mix")

args = parser.parse_args()



def newEquipment(arg=None) -> equipment.equipment:
    # values given in the paper
    power=args.equipment_power
    power_waiting=args.equipment_power_idle
    freq=args.equipment_clockspeed
    energyPerCycle=1e-27 * (freq**2)
    timeenergy_ratio = args.equipment_timeenergy_ratio
    cbInput = random.uniform(300, 500) * 1000
    cCycle = random.uniform(900, 1100) * 1000000

    maxDistance = 200
    distance = maxDistance * math.sqrt(random.random())
    posTheta = random.random() * 2 * math.pi

    #TODO: the paper doesn't specify how to initalize these?
    gain=numpy.random.rayleigh(distance)
    sDelayMax = 1000 #when offloading, just uploading can take 25+ seconds
                     #depending on randomness... (All tasks can be processed
                     #locally in < 1.5 sec...)

    return equipment.equipment(power=power, power_waiting=power_waiting,
                               gain=gain, frequency=freq,
                               energyPerCycle=energyPerCycle,
                               timeenergy_ratio=timeenergy_ratio,
                               cbInput=cbInput, cCycle=cCycle,
                               sDelayMax=sDelayMax, distance=distance)

equipmentStateMetadata = (3,3,3)
def equipmentToState(equipment):
    #NOTE: you might think that this function should go in equipment.py, but it
    #actually belongs here in config.py because it is dependent on the
    #implementation of config.newEquipment

    distributions = [#"percentiler" takes the linearOffset (see below) of the
                     #actual value and returns the percentile of that value
                     #relative to the distribution that generated it
        {"min": 300*1000, "max": 500*1000, "actual": equipment.cbInput, "granularity": 3, "percentiler": lambda x: x},
        {"min": 900*1000000, "max": 1100*1000000, "actual": equipment.cCycle, "granularity": 3, "percentiler": lambda x: x},
        {"min": 0, "max": 200, "actual": equipment.distance, "granularity": 3, "percentiler": lambda x: x**2},
    ]
    states = []
    for distribution in distributions:
        assert(distribution["min"] <= distribution["actual"] and distribution["actual"] <= distribution["max"])

        #0 = min; 1 = max
        linearOffset = (distribution["actual"] - distribution["min"]) / (distribution["max"] - distribution["min"])

        state = math.floor(distribution["granularity"] * distribution["percentiler"](linearOffset))
        state = min(state, distribution["granularity"] - 1) # just in case actual == max
        states.append(state)
    return tuple(states)



s = config.SmartSimulation(bandwidth=args.bandwidth, cEquipment=args.equipment_count,
                                 mec_clockspeed=args.mec_clockspeed, N0=args.n0,
                                 consEquipment=newEquipment)

if args.algorithm == 'smart':
    w = algSmart_world.algSmart_world(s, config.equipmentToState, config.equipmentStateMetadata, maxIter=5*args.equipment_count)
elif args.algorithm == 'one':
    w = alg1_world.alg1_world(s)
elif args.algorithm == 'const':
    w = algConst_world.algConst_world(s)
else:
    assert(False)

x1 = 0
y1 = 1

x2 = 5000
y2 = 0.01
assert(x1 <= x2)
assert(0 <= y2 and y2 <= y1 and y1 <= 1)
def computeRandAct(episode: int) -> int:
    prob = ((y2-y1)/(x2-x1))*(episode - x1) + y1

    prob = min(prob, y1)
    prob = max(prob, y2)

    return prob

ql = qlearning.qlearning(env=w, compute_randact=computeRandAct,
                         consPlayer=qtable.qtable,
                         player_config={"learning_rate": args.learning_rate},
                         future_discount=args.future_discount)

t1 = time.time()
print("Training Q-Table")
ql.train(args.train_episodes)
t2 = time.time()

print("Evaluating Q-Table")
results = ql.evaluate(args.eval_episodes)

medianmin    = statistics.median(result["min"]      for result in results)
medianmax    = statistics.median(result["max"]      for result in results)
medianlocal  = statistics.median(result["local"]    for result in results)
medianactual = statistics.median(result["actual"]   for result in results)
medianquant  = statistics.median(result["quantile"] for result in results)

print("TRAINING:")
print(f"Training duration: {(t2-t1):.2f}")
print(f"Number of updates: {ql.getTrainUpdateCount()}")
print(f"Number of entries in Q-Table: {ql.player._table.size}")
print("")
print("EVALUATION:")
print(f"min: {medianmin:.2f}")
print(f"max: {medianmax:.2f}")
print(f"local: {medianlocal:.2f}")
print(f"actual: {medianactual:.2f}")
print(f"quantile: {medianquant:.2f}")
