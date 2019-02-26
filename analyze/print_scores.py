from two_process_nlp.aggregator import Aggregator

DF_FROM_FILE = True

CORPUS = 'childes-20180319'
NUM_VOCAB = 4096
EMBEDDER = 'srn'  # rnd_normal
EMBED_SIZE = 500
TASK = 'cohyponyms_semantic'
PROCESS = 'novice'

ag_matching = Aggregator()
matching_df = ag_matching.make_df(load_from_file=True, verbose=True)
for col in matching_df.columns:
    print(col)

# filter
filtered_df = matching_df[(matching_df['embedder'] == EMBEDDER) &
                          (matching_df['task'] == TASK) &
                          (matching_df['corpus'] == CORPUS) &
                          (matching_df['num_vocab'] == NUM_VOCAB) &
                          (matching_df['process'] == PROCESS) &
                          (matching_df['embed_size'] == EMBED_SIZE)]

print(filtered_df[['location', 'score']])