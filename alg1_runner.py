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

#alg 1: 3432 possible actions, 8x8 state => 220k entries in Q-Table

ql.train(1000) #approx. 1k / 5min

t2 = time.time()

print(f"Training duration: {t2-t1}")
print(f"Number of updates: {ql.getTrainUpdateCount()}")
#73343 updates / 267 sec == 275 updates/sec

results = ql.evaluate(100)

avgmin    = statistics.mean(result["min"]    for result in results)
avgmax    = statistics.mean(result["max"]    for result in results)
avglocal  = statistics.mean(result["local"]  for result in results)
avgactual = statistics.mean(result["actual"] for result in results)

#with 1000 train, 100 test:
#min: 5.860929388809343; max: 11.578261641360378; local: 7.026610000000001; actual: 8.483781943784118

print(f"min: {avgmin}; max: {avgmax}; local: {avglocal}; actual: {avgactual}")
