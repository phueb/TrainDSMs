import os
from pathlib import Path


class Global:
    task_dir = Path(os.getenv('TASK_DIR', Path(__file__).parent.parent / 'task_data'))
    corpora_dir = Path(os.getenv('CORPORA_DIR', Path(__file__).parent.parent / 'corpora'))
    figs_dir = Path(os.getenv('FIGS_DIR', Path(__file__).parent.parent / 'figs'))
    embeddings_dir = Path(os.getenv('EMBEDDINGS_DIR', Path(__file__).parent.parent / 'embeddings'))
    sim_method = 'cosine'


class RandomControl:
    distribution = 'uniform'
    embed_size = 512

class WW:
    window_size = 5
    window_weight = 'flat'


class Categorization:  # TODO make unique for each embedder
    # novice
    num_opt_steps = 5
    # expert
    test_size = 0.3
    num_epochs = 10
    mb_size = 8
    max_freq = 50  # TODO log transform
    num_evals = 10
    learning_rate = 0.01
    num_hiddens = 64  # learning reaches 100% acc without hiddens but takes longer


class Corpora:
    UNK = 'UNKNOWN'
    # name = 'childes-20180319'
    name = 'childes-20171212'
    num_vocab = 16384


class Embeddings:
    retrain = False
    save = False


class Figs:
    width = 7
    dpi = 196
    axlabel_fontsize = 12
    line_width = 2