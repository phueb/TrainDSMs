from pathlib import Path

from ludwigcluster import config as ludwig_config

class Dirs:
    root = Path(__file__).parent.parent
    src = root / 'src'
    tasks = root / 'tasks'
    corpora = root / 'corpora'
    runs = root / 'runs'
    categories = root / 'categories'


class Eval:
    debug = False   # catches tensorflow errors properly
    num_processes = 4  # if too high (e.g. 8) doesn't result in speed-up (4 is sweet spot, 3x speedup) on 8-core machine
    max_num_eval_rows = 600  # 1200x1200 uses over 32GB RAM
    max_num_eval_cols = 600  # 600  # should be as large as num_rows for full matching evaluation
    standardize_num_relata = False  # don't do this - it reduces performance dramatically
    only_negative_examples = False
    num_reps = 2
    num_folds = 4
    vocab_sizes = [4096]  # also: 8192, 16384
    retrain = False
    save_scores = True
    save_figs = False
    num_opt_steps = 5
    num_evals = 10
    matching_metric = 'BalAcc'
    remove_duplicates_for_identification = False  # needs to be False to get above chance


class Embeddings:
    precision = 5
    retrain = False
    save = True
    sim_method = 'cosine'


class Corpus:
    UNK = 'UNKNOWN'
    name = 'childes-20180319'
    # name = 'tasa-20181213'
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


class Ludwig:
    project_name = '2StageNLP'
    runs_dir = ludwig_config.Dirs.lab / project_name / 'runs'
    exit_on_error = True
    default_to_local = False

class Glove:
    num_threads = 8