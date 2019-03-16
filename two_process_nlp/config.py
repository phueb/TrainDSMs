from pathlib import Path


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'two_process_nlp'
    create = root / 'create'
    tasks = root / 'tasks'
    corpora = root / 'corpora'
    runs = root / '{}_runs'.format(src.name)


class RemoteDirs:
    root = Path('/') / 'media' / 'lab' / '2ProcessNLP'
    runs = root / 'runs'


class Eval:
    debug = False   # catches tensorflow errors properly
    only_process1 = False
    shuffled_control = False
    resample = False
    verbose = True
    num_relata = 3  # identification
    num_lures = 3  # identification
    num_epochs_matching = 100
    num_epochs_identification = 2000  # 2000 is good for comparator but much less is needed for classifier
    num_processes = 4  # if too high (e.g. 8) doesn't result in speed-up (4 is sweet spot, 3x speedup) on 8-core machine
    max_num_eval_rows = 600  # 1200x1200 uses over 32GB RAM
    max_num_eval_cols = 600  # 600  # should be as large as num_rows for full matching evaluation
    standardize_num_relata = False  # don't do this - it reduces performance dramatically
    num_folds = 4
    num_opt_steps = 5
    num_evals = 20
    matching_metric = 'BalAcc'

    tertiary_probes = ['badger', 'bear', 'beaver', 'buffalo', 'bull', 'bunny', 'camel', 'cat',
                       'chimpanzee', 'chipmunk', 'cow', 'deer', 'donkey',
                       'elephant', 'giraffe', 'gorilla', 'groundhog', 'hamster', 'hare', 'hippo',
                       'horse', 'kitten', 'mammoth', 'moose', 'mouse', 'opossum', 'panda',
                       'pony', 'pup', 'rabbit', 'rat', 'rhino', 'squirrel', 'walrus', 'whale']

    # tertiary_probes = ['one', 'two', 'three', 'four', 'five', 'ten', 'eleven', 'twelve', 'thirteen',
    #                    'twenty', 'thirty', 'fifty', 'hundred', 'thousand', 'million']

    #
    assert num_epochs_matching % num_evals == 0
    assert num_epochs_identification % num_evals == 0


class Embeddings:
    save_w2e = True
    verbose = True
    precision = 5


class Corpus:
    UNK = 'UNKNOWN'
    name = 'childes-20180319'
    # name = 'tasa-20181213'
    num_vocab = 16384  # TODO test
    vocab_sizes = [4096, 8192, 16384]  # also: 4096, 8192, 16384


class Glove:
    num_threads = 8