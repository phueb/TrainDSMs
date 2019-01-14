
from src.aggregator import Aggregator

# factors to include or exclude
CORPUS_NAME = 'childes-20180319'  # childes-20180319 or tasa-20181213
NUM_VOCAB = 4096
ARCHITECTURE_NAME = 'comparator'
EVALUATOR_NAME = 'matching'
TASK_NAME = 'cohyponyms_syntactic'
EMBED_SIZE = 500

DF_FROM_FILE = True
SHOW = False  # TODO just for making tables


# get all data
ag_matching = Aggregator(EVALUATOR_NAME)

# plot scores
ag_matching.show_task_plot(CORPUS_NAME, NUM_VOCAB, ARCHITECTURE_NAME, TASK_NAME, EMBED_SIZE,
                           load_from_file=DF_FROM_FILE, width=14, dpi=300, show=SHOW)