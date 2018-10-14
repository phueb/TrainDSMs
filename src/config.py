import os
from pathlib import Path


class Dirs:
    tasks = Path(os.getenv('TASK_DIR', Path(__file__).parent.parent / 'task_data'))
    corpora = Path(os.getenv('CORPORA_DIR', Path(__file__).parent.parent / 'corpora'))
    figs = Path(os.getenv('FIGS_DIR', Path(__file__).parent.parent / 'figs'))
    embeddings = Path(os.getenv('EMBEDDINGS_DIR', Path(__file__).parent.parent / 'embeddings'))
    params = Path(os.getenv('PARAMS_DIR', Path(__file__).parent.parent / 'params'))


class TaskData:
    vocab_sizes = [4096, 8192, 16384]


class Categorization:  # TODO make unique for each embedder - separate models from task classes?
    # novice
    num_opt_steps = 5
    # expert
    device = 'cpu'
    test_size = 0.3
    num_epochs = 1000
    mb_size = 4
    max_freq = 1  # TODO log transform
    num_evals = 10
    learning_rate = 0.005
    num_hiddens = 128  # learning reaches 100% acc without hidden units but takes longer


class NymMatching:
    num_distractors = 5


class Embeddings:
    precision = 5
    retrain = True
    save = True
    sim_method = 'cosine'

class Corpus:
    spacy_batch_size = 50  # doesn't seem to affect speed loading childes-20180319
    UNK = 'UNKNOWN'
    name = 'childes-20180319'
    num_vocab = 4096


class Figs:
    width = 7
    dpi = 196
    axlabel_fontsize = 12
    line_width = 2