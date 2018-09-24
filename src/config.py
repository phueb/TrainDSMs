import os
from pathlib import Path


class Global:
    results_dir = Path(os.getenv('RESULTS_DIR', Path(__file__).parent.parent / 'results'))
    sim_method = 'cosine'
    embed_size = 128


class Categorization:
    dir = Path(os.getenv('TASK_DIR', Path(__file__).parent.parent / 'task_data'))
    # novice
    num_opt_steps = 10
    # expert
    test_size = 0.3
    num_epochs = 1
    mb_size = 1
    num_acts_samples = 50
    num_steps_to_eval = 200
    learning_rate = 0.005
    num_hiddens = 64
    shuffle_cats = False


class Corpus:
    name = 'childes-20180319'


class Embeddings:
    dir = Path(os.getenv('EMBEDDINGS_DIR', Path(__file__).parent.parent / 'embeddings'))
    retrain = False
    save = False


class Fig:
    width = 7
    dpi = 196
    AXLABEL_FONT_SIZE = 12