import time

import alg1_world
import config
import equipment
import qlearning
import qtable
import statistics

s = config.newSimulation()

w = alg1_world.alg1_world(s)

ql = qlearning.qlearning(env=w,
                         compute_randact=lambda episodeNum: 0.1,
                         consPlayer=qtable.qtable,
                         player_config=config.qtableConfig,
                         future_discount=config.future_discount)

t1 = time.time()

ql.train(1000)

t2 = time.time()

print(f"Training duration: {t2-t1}") # approx 25 seconds
print(f"Number of updates: {ql.getTrainUpdateCount()}") # approx 5k
#5333 updates / 25 sec == 213 updates / sec

results = ql.evaluate(100)

avgmin    = statistics.mean(result["min"]    for result in results)
avgmax    = statistics.mean(result["max"]    for result in results)
avglocal  = statistics.mean(result["local"]  for result in results)
avgactual = statistics.mean(result["actual"] for result in results)

print(f"min: {avgmin}; max: {avgmax}; local: {avglocal}; actual: {avgactual}")
