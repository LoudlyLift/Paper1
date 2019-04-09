import time
import sys
import argparse

import algSmart_world
import alg1_world
import config
import equipment
import qlearning
import qtable
import statistics

parser = argparse.ArgumentParser(description='Run edge computing simulations')
parser.add_argument('algorithm', choices=['one', 'smart'])
parser.add_argument('--train-episodes', type=int, default=10000, help="How many episodes to perform during training")
parser.add_argument('--eval-episodes', type=int, default=1000, help="How many episodes to perform during evaluation")
args = parser.parse_args()

s = config.newSimulation()

if args.algorithm == 'smart':
    w = algSmart_world.algSmart_world(s, config.equipmentToState, config.equipmentStateMetadata, maxIter=5*config.cEQUIPMENT)
elif args.algorithm == 'one':
    w = alg1_world.alg1_world(s)
else:
    assert(False)

ql = qlearning.qlearning(env=w,
                         compute_randact=lambda episodeNum: 0.1,
                         consPlayer=qtable.qtable,
                         player_config=config.qtableConfig,
                         future_discount=config.future_discount)

t1 = time.time()

ql.train(args.train_episodes)

t2 = time.time()

results = ql.evaluate(args.eval_episodes)

avgmin    = statistics.median(result["min"]    for result in results)
avgmax    = statistics.median(result["max"]    for result in results)
avglocal  = statistics.median(result["local"]  for result in results)
avgactual = statistics.median(result["actual"] for result in results)

print("TRAINING:")
print(f"Training duration: {(t2-t1):.2f}")
print(f"Number of updates: {ql.getTrainUpdateCount()}")
print(f"Numver of entries in Q-Table: {ql.player._table.size}")
print("")
print("EVALUATION:")
print(f"min: {avgmin:.2f}\nmax: {avgmax:.2f}\nlocal: {avglocal:.2f}\nactual: {avgactual:.2f}")
