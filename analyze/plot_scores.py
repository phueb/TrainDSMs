from itertools import product

from src.aggregator import Aggregator

# factors to include or exclude
CORPUS_NAME = 'childes-20180319'  # childes-20180319 or tasa-20181213
NUM_VOCAB = 4096
ARCHITECTURE_NAME = 'comparator'
EVAL_NAME = 'matching'
TASK_NAME = 'cohyponyms_semantic'  # TODO use _jw for paper
EMBED_SIZE = 200

DF_FROM_FILE = True
SAVE = False


# get all data
ag = Aggregator()

if not SAVE:
    ag.make_task_plot(CORPUS_NAME, NUM_VOCAB, ARCHITECTURE_NAME, EVAL_NAME, TASK_NAME, EMBED_SIZE,
                      load_from_file=DF_FROM_FILE, verbose=True, width=20, dpi=500, save=SAVE)

else:
    for embed_size, task_name in product(
            [500],
            [
                # 'hypernyms',
                'cohyponyms_semantic',
                # 'cohyponyms_syntactic',
                # 'events',
                # 'features_has',
                # 'features_is',
                # 'nyms_syn_jw',
                # 'nyms_ant_jw'
            ]):
        ag.make_task_plot(CORPUS_NAME, NUM_VOCAB, ARCHITECTURE_NAME, EVAL_NAME, task_name, embed_size,
                          load_from_file=DF_FROM_FILE, width=20, dpi=300, save=SAVE)