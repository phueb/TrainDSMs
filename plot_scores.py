
from src.aggregator import Aggregator

# factors to include or exclude
CORPUS_NAME = 'childes-20180319'  # childes-20180319 or tasa-20181213
NUM_VOCAB = 4096
ARCHITECTURE_NAME = 'comparator'
EVALUATOR_NAME = 'identification'
TASK_NAME = 'cohyponyms_semantic'  # can be cohyponyms_semantic, cohyponyms_syntactic, hypernyms, nyms_syn, nyms_ant
EMBED_SIZE = 30

DF_FROM_FILE = True


# get all data
ag_matching = Aggregator('matching')

# plot scores
ag_matching.show_task_plot(CORPUS_NAME, NUM_VOCAB, ARCHITECTURE_NAME, TASK_NAME, EMBED_SIZE,
                           load_from_file=DF_FROM_FILE, width=22)