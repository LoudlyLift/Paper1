import math
import simulation

import algConst_world

"""This trivial algorithm always chooses the best possible action
"""
class algOptimal_world(algConst_world.algConst_world):
    def __init__(self, simulation: simulation.simulation):
        super().__init__(simulation, action="Yolo")

    def step(self, _):
        optimal_cost = math.inf
        for action in self.possibleActions:
            optimal_cost = min(optimal_cost, self.simulation.computeCost(action))

        self._prior_cost = optimal_cost
        reward = 0
        done = True
        state = self.getState()

        return (state, reward, done)
