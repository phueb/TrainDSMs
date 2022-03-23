"""
the purpose of this script is to compute accuracy levels of a model that guesses randomly,
and has different kinds of information about the target ranking or none.

"""

import numpy as np

num_trials = 1_000_000
num_labels = 32
num_confusable = 2

a = np.arange(num_labels)

# simulate random guessing in exp 2b 1 and 3b1
hits = 0
for i in range(num_trials):
    b = a.copy()
    b[:num_confusable] = np.random.permutation(a[:num_confusable])
    if b[0] == a[0] and b[1] == a[1]:
        hits += 1

print(hits / num_trials)
