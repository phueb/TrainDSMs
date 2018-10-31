import os
from pathlib import Path


class Dirs:
    tasks = Path(os.getenv('TASK_DIR', Path(__file__).parent.parent / 'task_data'))
    corpora = Path(os.getenv('CORPORA_DIR', Path(__file__).parent.parent / 'corpora'))
    runs = Path(os.getenv('RUNS_DIR', Path(__file__).parent.parent / 'runs'))


class Task:
    vocab_sizes = [4096, 8192, 16384]
    retrain = True
    clear_scores = False
    append_scores = False
    save_figs = False
    num_opt_steps = 1
    device = 'cpu'
    num_evals = 10


class Categorization:  # TODO make unique to each embedder
    """
    Adadelta
    lstm: lr=0.1 + mb_size=8 + num_hiddens=64 + beta=0.0
    ww ppmi svd-200: lr=0.1 + mb_size=8 + num_hiddens=64 + beta=0.0 - CANNOT LEARN WITH REGULARIZATION
    """
    beta = 0.0
    num_folds = 6  # also determines number of examples in test vs. train splits
    num_epochs = 500  # 500
    mb_size = 8
    log_freq = False
    learning_rate = 0.1
    num_hiddens = 64
    # figs
    softmax_criterion = 0.5
    num_bins = 10
    num_diagnosticity_steps = 50


class NymMatching:  # TODO embedder-dependent
    """
    SGD
    cbow: lr=0.00001 + mb_size=2 + num_output=128 + margin=100.0
    AdaDelta
    srn: lr=0.00001 + mb_size=2 + num_output=128 + margin=100.0 + beta=0.2
    """
    margin = 100.0  # must be float and MUST be at least 40 or so
    remove_duplicate_nyms = False  # needs to be False to get above chance
    train_on_second_neighbors = True  # performance is much better with additional training
    beta = 0.3
    num_output = 128
    mb_size = 2
    num_epochs = 500
    learning_rate = 0.1
    num_folds = 4


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