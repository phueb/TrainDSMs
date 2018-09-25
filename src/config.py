import os
from pathlib import Path


class Global:
    sim_method = 'cosine'
    embed_size = 512


class Categorization:
    dir = Path(os.getenv('TASK_DIR', Path(__file__).parent.parent / 'task_data'))
    # novice
    num_opt_steps = 1
    # expert
    test_size = 0.3
    num_epochs = 1
    mb_size = 1
    num_acts_samples = 50
    num_steps_to_eval = 200
    learning_rate = 0.005
    num_hiddens = 64
    shuffle_cats = False


class Corpora:
    dir = Path(os.getenv('CORPORA_DIR', Path(__file__).parent.parent / 'corpora'))
    name = 'childes-20180319'


class Embeddings:
    dir = Path(os.getenv('EMBEDDINGS_DIR', Path(__file__).parent.parent / 'embeddings'))
    retrain = False
    save = False


class Figs:
    dir = Path(os.getenv('FIGS_DIR', Path(__file__).parent.parent / 'figs'))
    width = 7
    dpi = 196
    AXLABEL_FONT_SIZE = 12