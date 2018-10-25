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


class Categorization:  # TODO make unique to each embedder
    """
    Adadelta
    lstm: lr=0.1 + mb_size=8 + num_hiddens=64 + beta=0.0
    ww ppmi svd-200: lr=0.1 + mb_size=8 + num_hiddens=64 + beta=0.0 - CANNOT LEARN WITH REGULARIZATION
    """
    # novice
    num_opt_steps = 3
    # expert
    beta = 0.0
    run_shuffled = False
    device = 'cpu'
    num_folds = 6  # also determines number of examples in test vs. train splits
    num_epochs = 400
    num_evals = 10
    mb_size = 8
    log_freq = False
    learning_rate = 0.1
    num_hiddens = 64
    # figs
    softmax_criterion = 0.5
    num_bins = 10


class NymMatching:
    num_distractors = 5


class Embeddings:
    precision = 5
    retrain = False
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