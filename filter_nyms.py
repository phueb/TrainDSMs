from cytoolz import itertoolz
import numpy as np
import pyprind
import sys
import multiprocessing as mp

from src.embedders.random_control import RandomControlEmbedder
from src.params import RandomControlParams
from src.architectures import comparator
from src.evaluators.matching import Matching
from src.embedders.base import w2e_to_sims
from src.params import gen_all_param_combinations
from src.embedders.base import EmbedderBase
from src import config

DEBUG = False  # reduces computation time

NYM_TYPE = 'ant'

CORPUS_NAME = config.Corpus.name
VOCAB_SIZE = config.Corpus.num_vocab
DISTANCE_THR = 7 * 2  # distance in number of words
NUM_PROCESSES = 8

config.Eval.max_num_eval_rows = 10000
config.Eval.max_num_eval_cols = 10000


def filter_nyms(doc, pid):
    if DEBUG:
        doc = doc[:1000]
    res = []  # pairs
    if len(doc) < DISTANCE_THR:
        return res
    else:
        print('Starting process {}: Num words in doc={}'.format(pid, len(doc)))
    pbar = pyprind.ProgBar(len(doc), stream=sys.stdout)
    for window_id, window in enumerate(itertoolz.sliding_window(DISTANCE_THR, doc)):
        probes_in_window = [w for w in window if w in row_and_col_words]
        for p1 in ev.row_words:
            if p1 in probes_in_window:
                for p2 in ev.probe2relata[p1]:
                    if p2 in probes_in_window:
                        res.append((p1, p2))
        pbar.update()
    return res

# embedder
embedders = (RandomControlEmbedder(param2id, param2val)
             for param2id, param2val in gen_all_param_combinations(RandomControlParams))
embedder = next(embedders)
embedder.train()  # populates w2e
# evaluator
ev = Matching(comparator, 'nyms', NYM_TYPE)
vocab_sims_mat = w2e_to_sims(embedder.w2e, embedder.vocab, embedder.vocab)
all_eval_probes, all_eval_candidates_mat = ev.make_all_eval_data(
    vocab_sims_mat, embedder.vocab, suffix='_unfiltered')  # populates probe2relata
ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
                        all_eval_probes, all_eval_candidates_mat, rep_id=0)
row_and_col_words = set(ev.row_words + ev.col_words)
print('Num probes: {}'.format(len(ev.probe2relata)))
print('Num unique row + col words : {}'.format(len(row_and_col_words)))

# filter nyms
docs = EmbedderBase.load_corpus_data(num_vocab=VOCAB_SIZE)[3]
pool = mp.Pool(processes=NUM_PROCESSES)
results = [pool.apply_async(filter_nyms, kwds={'doc': doc, 'pid': n})
           for n, doc in enumerate(np.array_split(np.concatenate(docs), NUM_PROCESSES))]
new_probe2relata = {p: set() for p in ev.probe2relata.keys()}
try:
    num_kept = 0
    for n, res in enumerate(results):
        pairs = res.get()
        for p1, p2 in pairs:
            if p2 not in new_probe2relata[p1]:
                new_probe2relata[p1].add(p2)
                num_kept += 1
except KeyboardInterrupt:
    pool.close()
    raise SystemExit('Interrupt occurred during multiprocessing. Closed worker pool.')
else:
    pool.close()

# print results
for k, v in new_probe2relata.items():
    print(k, v)
print('Num kept pairs={}'.format(num_kept))

# write to file
out_path = config.Dirs.tasks / 'nyms' / NYM_TYPE / '{}_{}.txt'.format(CORPUS_NAME, VOCAB_SIZE)
if not out_path.parent.exists():
    out_path.parent.mkdir()
with out_path.open('w') as f:
    print('Writing {}'.format(out_path))
    for probe, nyms in new_probe2relata.items():
        if not nyms:
            continue
        nyms = ' '.join(nyms)
        line = '{} {}\n'.format(probe, nyms)
        print(line.strip('\n'))
        f.write(line)