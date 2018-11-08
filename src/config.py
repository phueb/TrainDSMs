import os
from pathlib import Path


class Dirs:
    tasks = Path(os.getenv('TASK_DIR', Path(__file__).parent.parent / 'task_data'))
    corpora = Path(os.getenv('CORPORA_DIR', Path(__file__).parent.parent / 'corpora'))
    runs = Path(os.getenv('RUNS_DIR', Path(__file__).parent.parent / 'runs'))


class Task:
    debug = False
    num_processes = 4  # too high (e.g. 8) doesn't result in speed-up (4 is sweet spot, 3x speedup) on 8-core machine
    num_reps = 3
    num_folds = 4
    vocab_sizes = [4096, 8192, 16384]
    retrain = False
    save_scores = True
    save_figs = False
    num_opt_steps = 3
    device = 'cpu'
    num_evals = 10
    metric = 'ba'
    remove_duplicate_nyms = False  # needs to be False to get above chance


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
    # hypernym_identification
    softmax_criterion = 0.5
    num_bins = 10
    num_diagnosticity_steps = 50