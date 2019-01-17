from itertools import product

from src.aggregator import Aggregator

# factors to include or exclude
CORPUS_NAME = 'childes-20180319'  # childes-20180319 or tasa-20181213
NUM_VOCAB = 4096
ARCHITECTURE_NAME = 'comparator'
EVALUATOR_NAME = 'matching'
TASK_NAME = 'hypernyms'  # TODO use _jw for paper
EMBED_SIZE = 500

DF_FROM_FILE = True
SAVE = False


# get all data
ag_matching = Aggregator(EVALUATOR_NAME)

# plot scores
for embed_size, task_name in product(
        [500],
        [
            'hypernyms',
            # 'cohyponyms_semantic',
            # 'cohyponyms_syntactic',
            # 'events',
            # 'features_has',
            # 'features_is',
            # 'nyms_syn_jw',
            # 'nyms_ant_jw'
        ]):
    ag_matching.make_task_plot(CORPUS_NAME, NUM_VOCAB, ARCHITECTURE_NAME, task_name, embed_size,
                               load_from_file=DF_FROM_FILE, width=20, dpi=300, save=SAVE)