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


ag = Aggregator()

for arch, task in product(
        ['comparator', 'classifier'],
        [
            'hypernyms',
            'cohyponyms_semantic',
            'cohyponyms_syntactic',
            'events',
            'features_has',
            'features_is',
            # 'nyms_syn_jw',
            # 'nyms_ant_jw'
        ]):

    #
    if arch == 'classifier':
        neg_pos_ratios = [np.nan]
        num_epochs_per_row_word_list = [20]
    elif arch == 'comparator':
        neg_pos_ratios = [1.0]
        num_epochs_per_row_word_list = [0.2]
    else:
        raise  AttributeError('Invalid arg to "architecture".')
    for neg_pos_ratio in neg_pos_ratios:
        for num_epochs_per_row_word in num_epochs_per_row_word_list:
            print(arch, task, neg_pos_ratio, num_epochs_per_row_word)
            ag.make_task_plot(CORPUS_NAME, NUM_VOCAB, arch, EVAL_NAME, task, EMBED_SIZE,
                              neg_pos_ratio,
                              num_epochs_per_row_word,
                              load_from_file=DF_FROM_FILE,
                              width=20,
                              dpi=200,
                              save=SAVE,
                              min_num_reps=MIN_NUM_REPS)