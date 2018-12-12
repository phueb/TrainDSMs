
from src.aggregator import Aggregator

# factors to include or exclude
INCLUDE_DICT = {'num_vocab': 4096, 'corpus_name': 'childes-20180319'}
ARCHITECTURE_NAME = 'comparator'
EVALUATOR_NAME = 'identification'
TASK_NAME = 'cohyponyms_semantic'  # can be cohyponyms_semantic, cohyponyms_syntactic, hypernyms, nyms_syn, nyms_ant
EMBED_SIZE = 500

DF_FROM_FILE = True


# get all data
ag_matching = Aggregator('matching')

# plot scores for task + arch +  evaluation + embed_size across all embedders
ag_matching.show_task_plot(ARCHITECTURE_NAME, TASK_NAME, EMBED_SIZE, load_from_file=DF_FROM_FILE)




