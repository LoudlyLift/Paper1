import numpy
import math

def bestLegalMove(qvals, legality):
    actV = -math.inf
    actI = -1
    for index, (val, legal) in enumerate(zip(qvals, legality)):
        if legal and val > actV:
            actI = index
            actV = val
    #assert(actI != -1)
    if actI == -1:
        import pdb; pdb.set_trace()
        foo=1+1
    return actI

class qlearning:
    """env state must be a tuple; no lists, no matricies, etc. A simple tuple of
    integers, and all integers in the range [0, max) must be valid, where max is
    the corresponding value from getStateMetadata.

    env must define these methods:

        reset(self): resets env for a new game (including the first), and
        returns the starting state.

        getRandomMove(self): get a random move

        step(self, int): perform the specified move and return the tuple
            (state_new,reward,done).

        getStateMetadata(): returns a tuple of the same length as the state
        vector. Each entry is an integer specifying the number of values that
        entry can take in the actual state vector.

        getNumActions(): returns the number of actions that can be made at
        any given time

        getLegalMoves(): returns a list of length == getNumActions(), whose
        entries are False if the move is illegal, and true if
        legal. (Equivalently, could be a list of zero/one)

    compute_randact(episode_num): given the episode number, this computes
    probability with which a random move should be made instead of action
    chosen.

    cls_player must make an instance using
    cls_player(state_shape, num_actions). That instance must have these
    methods:

        computeQState(self, state): returns a list of the estimated value of
        taking each enumerated action. (i.e. the row of the QTable
        corresponding to state)

        updateQState(self, state, qValues): do the player's equivalent of
        updating state's row in the Q-Table to match it's new estimated
        values.

    """
    def __init__(self, env, compute_randact, consPlayer, player_config=None, future_discount=.99):
        self._env = env
        self._compute_randact = compute_randact
        self._future_discount = future_discount

        state_metadata = env.getStateMetadata()
        self._player = consPlayer(state_metadata, self._env.getNumActions(), config=player_config)

    # runs count episodes.
    #
    # Returns (player, [ Σ(episode i's rewards) for i in range(count) ])
    def runEpisodes(self, count=1):
        reward_sums = []
        cStep = 0
        for ep_num in range(1, count+1):
            try:
                state_old = self._env.reset()
                reward_sum = 0
                done = False

                while not done:
                    allActQs = self._player.computeQState(state_old)
                    doRandom = numpy.random.rand(1) < self._compute_randact(ep_num)
                    if doRandom:
                        act = self._env.getRandomMove()
                    else:
                        legalMoves = self._env.getLegalMoves()
                        act = bestLegalMove(allActQs, legalMoves)

                    state_new,reward,done = self._env.step(act)
                    if done:
                        maxHypotheticalQ = 0
                    else:
                        qvals = self._player.computeQState(state_new)
                        legalMoves = self._env.getLegalMoves()
                        maxHypotheticalQ = bestLegalMove(qvals, legalMoves)
                    allActQs[act] = reward + self._future_discount * maxHypotheticalQ
                    self._player.updateQState(cStep, state_old, allActQs)

                    reward_sum += reward
                    state_old = state_new
                    cStep += 1
                reward_sums.append(reward_sum)
            except KeyboardInterrupt as e:
                print("Keyboard Interrupt")
                break
        return (self._player, reward_sums)
