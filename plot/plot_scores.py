from itertools import product

from two_process_nlp.aggregator import Aggregator

# factors to include or exclude
CORPUS_NAME = 'childes-20180319'  # childes-20180319 or tasa-20181213
NUM_EPOCHS = 1000
NUM_VOCAB = 4096
EVAL_NAME = 'identification'
EMBED_SIZE = 200

DF_FROM_FILE = True
SAVE = False
MIN_NUM_REPS = 1


ag = Aggregator()


for task, arch in product(
        [
            # 'cohyponyms_semantic',
            # 'hypernyms',
            # 'cohyponyms_syntactic',
            # 'events',
            # 'features_has',
            # 'features_is',
            # 'nyms_syn_jw',
            'nyms_ant_jw'
        ],
        ['classifier', 'comparator'],):

    #
    if arch == 'classifier':
        neg_pos_ratios = [1.0]
    elif arch == 'comparator':
        neg_pos_ratios = [1.0]
    else:
        raise  AttributeError('Invalid arg to "architecture".')
    for neg_pos_ratio in neg_pos_ratios:
        print(arch, task, neg_pos_ratio, NUM_EPOCHS)
        ag.make_task_plot(CORPUS_NAME, NUM_VOCAB, arch, EVAL_NAME, task, EMBED_SIZE,
                          neg_pos_ratio,
                          NUM_EPOCHS,
                          load_from_file=DF_FROM_FILE,
                          height=10,
                          width=16,
                          dpi=200,
                          save=SAVE,
                          min_num_reps=MIN_NUM_REPS)