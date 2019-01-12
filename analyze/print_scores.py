from src.aggregator import Aggregator

DF_FROM_FILE = True

CORPUS = 'childes-20180319'
NUM_VOCAB = 4096
EMBEDDER = 'lstm'  # rnd_normal
EMBED_SIZE = 30
TASK = 'cohyponyms_semantic'
STAGE = 'novice'

ag_matching = Aggregator('matching')
matching_df = ag_matching.make_df(load_from_file=True)
for col in matching_df.columns:
    print(col)

# filter
filtered_df = matching_df[(matching_df['embedder'] == EMBEDDER) &
                          (matching_df['task'] == TASK) &
                          (matching_df['corpus'] == CORPUS) &
                          (matching_df['num_vocab'] == NUM_VOCAB) &
                          (matching_df['stage'] == STAGE) &
                          (matching_df['embed_size'] == EMBED_SIZE)]

print(filtered_df[['location', 'score']])