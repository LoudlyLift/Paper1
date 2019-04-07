import time

import alg1_world
import config
import equipment
import qlearning
import qtable

s = config.newSimulation()

w = alg1_world.alg1_world(s)

ql = qlearning.qlearning(env=w,
                         compute_randact=lambda episodeNum: 0.1,
                         consPlayer=qtable.qtable,
                         player_config=config.qtableConfig,
                         future_discount=config.future_discount)

t1 = time.time()

ql.train(100)

t2 = time.time()

print(t2-t1)

print(ql.getTrainUpdateCount())

#5333 updates / 25 sec == 213 updates / sec
