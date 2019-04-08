import time

import algSmart_world
import config
import equipment
import qlearning
import qtable
import statistics

s = config.newSimulation()

w = algSmart_world.algSmart_world(s, config.equipmentToState, config.equipmentStateMetadata)

ql = qlearning.qlearning(env=w,
                         compute_randact=lambda episodeNum: 0.1,
                         consPlayer=qtable.qtable,
                         player_config=config.qtableConfig,
                         future_discount=config.future_discount)

t1 = time.time()

#alg 1: 3**6 states, 4 actions => 2916 states
ql.train(1000000) #approx. 1k / 25min

t2 = time.time()

results = ql.evaluate(1000)

avgmin    = statistics.median(result["min"]    for result in results)
avgmax    = statistics.median(result["max"]    for result in results)
avglocal  = statistics.median(result["local"]  for result in results)
avgactual = statistics.median(result["actual"] for result in results)

print("TRAINING:")
print(f"Training duration: {(t2-t1):.2f}")
print(f"Number of updates: {ql.getTrainUpdateCount()}")
print(f"Numver of entries in Q-Table: {ql.player._table.size}")

print("EVALUATION:")
print(f"min: {avgmin:.2f}; max: {avgmax:.2f}; local: {avglocal:.2f}; actual: {avgactual:.2f}")
