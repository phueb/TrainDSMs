from itertools import product

from two_stage_nlp.aggregator import Aggregator

# factors to include or exclude
CORPUS_NAME = 'childes-20180319'  # childes-20180319 or tasa-20181213
NUM_VOCAB = 4096
EVAL_NAME = 'matching'
EMBED_SIZE = 200
PROP_NEGATIVE = 0.5  # TODO

DF_FROM_FILE = True
SAVE = True
MIN_NUM_REPS = 1


# get all data
ag = Aggregator()

for arch, task_name in product(
        ['classifier', 'comparator'],
        [
            # 'hypernyms',
            # 'cohyponyms_semantic',
            # 'cohyponyms_syntactic',
            # 'events',
            # 'features_has',
            # 'features_is',
            'nyms_syn_jw',
            # 'nyms_ant_jw'
        ]):
    ag.make_task_plot(CORPUS_NAME, NUM_VOCAB, arch, EVAL_NAME, task_name, EMBED_SIZE, PROP_NEGATIVE,
                      load_from_file=DF_FROM_FILE,
                      width=20,
                      dpi=300,
                      save=False,
                      min_num_reps=MIN_NUM_REPS)