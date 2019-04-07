import typing
import numpy
import math

class qtable:
    """See qlearning.py for the definitions of state_metadata, state, and qValues
    """

    def __init__(self, state_metadata: typing.Tuple[int, ...], num_actions: int, config=None):
        self._table = numpy.zeros(state_metadata + (num_actions,))
        self._learning_rate = config['learning_rate']

    def computeQState(self, state):
        return self._table[state]

    def updateQState(self, _, state, qValues):
        lr = self._learning_rate
        val = self._table[state]

        val = (1 - lr) * val + lr * qValues

        self._table[state] = val
