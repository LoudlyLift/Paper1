import config
import numpy

#each tuple is (additional, buckets, weights, expected)
testCases = [
    (0.3,  [0.5, 0.1, 0.1], [1.5, 1.5, 1.5], [0.5, 0.25, 0.25]),
    (0.1,  [0.5, 0.3, 0.1], [8,   8,   8],   [0.5, 0.3, 0.2]),
    (0,    [0.5, 0.3, 0.2], [8,   8,   8],   [0.5, 0.3, 0.2]),
    (1,    [0.0, 0.0, 0.0], [10,  5,   5],   [0.5, 0.25, 0.25]),
    (1,    [0,   0,   0],   [10,  5,   5],   [0.5, 0.25, 0.25]),
    (0.7,  [0.3],           [1],             [1]),
    (0.6,  [0.2, 0.2],      [9, 1],          [0.8, 0.2]),
    (0.4,  [0.3, 0.2, 0.1], [3,   2,   1],   [0.5, 1/3, 1/6]),
    (0.4,  [0.3, 0.2, 0.1], [4,   2,   2],   [0.5, 0.25, 0.25]),
    (0.8,  [0.1, 0.1],      [1,   1],        [0.5, 0.5]),
]

THRESHOLD=1e-9

fail_count = 0
for test in testCases:
    additional = test[0]
    buckets = test[1]
    weights = test[2]
    expected = test[3]
    assert(abs(sum(expected) - (additional + sum(buckets))) <= THRESHOLD)

    actual = config.SmartSimulation.weightedDistribution(additional, buckets, weights)

    difference = numpy.array(actual) - numpy.array(expected)

    if max(abs(difference)) > THRESHOLD:
        fail_count += 1
        print(f"TEST FAILED:\nAdditional: {additional}\nBuckets: {buckets}\nWeights: {weights}\nExpected: {expected}\nActual: {actual}\n\n")

exit(fail_count)
