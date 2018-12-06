import numpy as np

from src import config
from src.architectures import comparator
from src.evaluators.matching import Matching
from src.embedders.base import w2e_to_sims

config.Eval.save_scores = False
config.Eval.only_negative_examples = False  # TODO

EMBED_SIZE = 30
NUM_PROBES = 500  # needs to be more than mb_size
NUM_PROBE_RELATA = 40  # this also changes prob of pos examples


class MatchingParams:
    prop_negative = [None]  # 0.3 is better than 0.1 or 0.2 but not 0.5
    # arch-evaluator interaction
    num_epochs = [100]  # 100 is better than 10 or 20 or 50


class RandomControlEmbedderDummy():
    def __init__(self, embed_size, vocab):
        self.embed_size = embed_size
        self.dim1 = embed_size
        self.vocab = vocab
        self.w2e = self.make_w2e()
        #
        self.name = 'random_control'
        self.time_of_init = 'test'

    def make_w2e(self):
        return {w: np.random.uniform(-1.0, 1.0, self.embed_size) for w in self.vocab}


# embedder
p = config.Dirs.corpora / '{}_vocab.txt'.format(config.Corpus.name)
vocab = np.loadtxt(p, 'str')
embedder = RandomControlEmbedderDummy(EMBED_SIZE, vocab)

# probes + relata
probes = np.random.choice(vocab, size=NUM_PROBES, replace=False)
probe_relata = [np.random.choice(vocab,
                                 size=NUM_PROBE_RELATA,
                                 replace=False) for _ in range(NUM_PROBES)]

# all_eval_probes + all_eval_candidates_mat
relata = sorted(np.unique(np.concatenate(probe_relata)))
all_eval_probes = probes
num_eval_probes = len(all_eval_probes)
all_eval_candidates_mat = np.asarray([relata for _ in range(num_eval_probes)])

# ev
ev = Matching(comparator, 'cohyponyms', 'semantic', matching_params=MatchingParams)
ev.probe2relata = {p: r for p, r in zip(probes, probe_relata)}

# prepare data
ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
    all_eval_probes, all_eval_candidates_mat, rep_id=0)
print('Shape of all eval data={}'.format(all_eval_candidates_mat.shape))
print('Shape of down-sampled eval data={}'.format(ev.eval_candidates_mat.shape))
ev.pos_prob = ev.calc_pos_prob()

# score
sims_mat = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words)  # sims can have duplicate rows
ev.score_novice(sims_mat)
ev.train_and_score_expert(embedder, rep_id=0)