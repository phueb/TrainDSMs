from itertools import product
import numpy as np

from two_stage_nlp.aggregator import Aggregator

# factors to include or exclude
CORPUS_NAME = 'childes-20180319'  # childes-20180319 or tasa-20181213
NUM_VOCAB = 4096
EVAL_NAME = 'matching'
EMBED_SIZE = 200

DF_FROM_FILE = True
SAVE = False
MIN_NUM_REPS = 1


# get all data
ag = Aggregator()

for arch, task in product(
        ['comparator', 'classifier'],
        [
            # 'hypernyms',
            # 'cohyponyms_semantic',
            # 'cohyponyms_syntactic',
            # 'events',
            # 'features_has',
            'features_is',
            # 'nyms_syn_jw',
            # 'nyms_ant_jw'
        ]):

    #
    if arch == 'classifier':
        prop_negatives = [np.nan]
    elif arch == 'comparator':
        prop_negatives = [0.5, 0.0]
    else:
        raise  AttributeError('Invalid arg to "architecture".')
    for prop_negative in prop_negatives:
        print(arch, task, prop_negative)
        ag.make_task_plot(CORPUS_NAME, NUM_VOCAB, arch, EVAL_NAME, task, EMBED_SIZE, prop_negative,
                          load_from_file=DF_FROM_FILE,
                          width=20,
                          dpi=200,
                          save=SAVE,
                          min_num_reps=MIN_NUM_REPS)