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
    window_size = 1
    window_weight = 'flat'
    matrix_type = 'summed'


class LSTM:
    train_percent = 0.9
    num_eval_steps = 1000
    shuffle_per_epoch = True
    embed_init_range = 0.1
    dropout_prob = 0
    num_layers = 1
    num_steps = 7
    embed_size = 512
    batch_size = 64  # TODO was 20
    num_epochs = 20
    lr_decay_base = 1 / 1.15
    initital_lr = 0.1  # TODO was 20
    num_epochs_with_flat_lr = 10  # TODO was 5
    grad_clip = None  # TODO was 0.25


class Categorization:  # TODO make unique for each embedder
    # novice
    num_opt_steps = 1
    # expert
    test_size = 0.3
    num_epochs = 1000
    mb_size = 4
    max_freq = 1  # TODO log transform
    num_evals = 10
    learning_rate = 0.005
    num_hiddens = 128  # learning reaches 100% acc without hiddens but takes longer


class Reduce:
    dimensions = 200

    # parameters for random vector accumulation
    rv_mean = 0
    rv_stdev = 1


class Corpora:
    UNK = 'UNKNOWN'
    name = 'childes-20180319'
    # name = 'childes-20171212'
    num_vocab = 4096


class Embeddings:
    retrain = False
    save = True


class Figs:
    width = 7
    dpi = 196
    axlabel_fontsize = 12
    line_width = 2