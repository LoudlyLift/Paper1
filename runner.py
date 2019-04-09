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
                         player_config=config.qtableConfig,
                         future_discount=config.future_discount)

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
print(f"Numver of entries in Q-Table: {ql.player._table.size}")
print("")
print("EVALUATION:")
print(f"min: {medianmin:.2f}")
print(f"max: {medianmax:.2f}")
print(f"local: {medianlocal:.2f}")
print(f"actual: {medianactual:.2f}")
print(f"quantile: {medianquant:.2f}")
