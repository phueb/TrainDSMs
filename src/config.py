import os
from pathlib import Path


class Global:
    sim_method = 'cosine'
    embed_size = 128


class Categorization:
    # novice
    NUM_BAYES_STEPS = 10
    # expert
    num_epochs = 1
    mb_size = 1
    num_acts_samples = 50
    num_steps_to_eval = 200
    learning_rate = 0.005
    num_hiddens = 100
    shuffle_cats = False
    mode = 'semantic'


class Corpus:
    name = 'childes-20180319'


class Embedder:
    dir = Path(os.getenv('EMBEDDINGS_PATH', Path(__file__).parent.parent / 'embeddings'))
    retrain = False
    save = False


class Fig:
    width = 7
    dpi = 196
    AXLABEL_FONT_SIZE = 12