import pandas as pd
import sys
from collections import Counter, OrderedDict
from itertools import islice
import pyprind
from spacy.lang.en import English

from src import config
from src.aggregator import Aggregator
from src.architectures import comparator
from src.evaluators.matching import Matching
from src.embedders.base import w2e_to_sims
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder


nlp = English()


def preprocessing_job(num_vocab=config.Corpus.num_vocab):  # TODO test do this once before embedder jobs
    docs = []
    w2freq = Counter()
    # tokenize + count words
    p = config.Dirs.corpora / '{}.txt'.format(config.Corpus.name)
    with p.open('r') as f:
        texts = f.read().splitlines()  # removes '\n' newline character
    num_texts = len(texts)
    print('\nTokenizing {} docs...'.format(num_texts))
    pbar = pyprind.ProgBar(num_texts, stream=sys.stdout)
    # TODO tokenization could benefit from multiprocessing
    for text in texts:
        spacy_doc = nlp(text)
        doc = [w.text for w in spacy_doc]
        docs.append(doc)
        c = Counter(doc)
        w2freq.update(c)
        pbar.update()
    # vocab
    deterministic_w2f = OrderedDict(sorted(w2freq.items(), key=lambda item: (item[1], item[0]), reverse=True))
    if num_vocab is None:
        vocab = list(islice(deterministic_w2f.keys(), 0, num_vocab))
    else:
        vocab = list(islice(deterministic_w2f.keys(), 0, num_vocab - 1))
        vocab.append(config.Corpus.UNK)
    vocab = list(sorted(vocab))
    if num_vocab is None:  # if no vocab specified, use the whole corpus
        num_vocab = len(w2freq)
    print('Creating vocab of size {}...'.format(num_vocab))
    print('Least frequent word occurs {} times'.format(deterministic_w2f[vocab[-2]]))
    assert '\n' not in vocab
    assert len(vocab) == num_vocab
    # insert UNK + make numeric
    print('Mapping words to ids...')
    t2id = {t: i for i, t in enumerate(vocab)}
    numeric_docs = []
    for doc in docs:
        numeric_doc = []

        for n, token in enumerate(doc):
            if token in t2id:
                numeric_doc.append(t2id[token])
            else:
                doc[n] = config.Corpus.UNK
                numeric_doc.append(t2id[config.Corpus.UNK])
        numeric_docs.append(numeric_doc)
    return deterministic_w2f, vocab, docs, numeric_docs


def embedder_job(param2val):  # TODO put backup function from rnnlab to ludwigcluster, import it, and put it at end of job
    """
    Train a single embedder once, and evaluate all novice and expert scores for each task once
    """
    # params
    job_name = param2val['job_name']
    print('===================================================')
    print('Starting job {}'.format(job_name))
    print('Param2val:')
    for k, v in param2val.items():
        print(k, v)
    # load embedder
    if 'random_type' in param2val:
        embedder = RandomControlEmbedder(param2val)
    elif 'rnn_type' in param2val:
        embedder = RNNEmbedder(param2val)
    elif 'w2vec_type' in param2val:
        embedder = W2VecEmbedder(param2val)
    elif 'count_type' in param2val:
        embedder = CountEmbedder(param2val)
    elif 'glove_type' in param2val:
        raise NotImplementedError
    else:
        raise RuntimeError('Could not infer embedder name from param2val')
    # stage 1
    print('Training stage 1...')
    sys.stdout.flush()
    embedder.train()
    embedder.save_w2e()
    # stage 2
    for architecture in [comparator]:
        for ev in [
            Matching(architecture, 'cohyponyms', 'semantic'),
            Matching(architecture, 'cohyponyms', 'syntactic'),
            Matching(architecture, 'features', 'is'),
            Matching(architecture, 'features', 'has'),
            Matching(architecture, 'nyms', 'syn'),
            Matching(architecture, 'nyms', 'syn', suffix='_jw'),
            Matching(architecture, 'nyms', 'ant'),
            Matching(architecture, 'nyms', 'ant', suffix='_jw'),
            Matching(architecture, 'hypernyms'),
            Matching(architecture, 'events'),

            # Identification(architecture, 'nyms', 'syn', suffix=''),
            # Identification(architecture, 'nyms', 'ant', suffix=''),
        ]:
            if ev.suffix != '':
                print('WARNING: Using task file suffix "{}".'.format(ev.suffix))
            # check scores_p
            scores_p = ev.make_scores_p(embedder.location)
            try:
                scores_p.parent.exists()
            except OSError:
                raise OSError('{} is not reachable. Check VPN or mount drive.'.format(scores_p))
            if scores_p.exists() and not config.Eval.debug:
                print('WARNING: {} should not exist. This is likely a failure of ludwigcluster to distribute tasks.')
                scores_p.unlink()
            # make eval data - row_words can contain duplicates
            vocab_sims_mat = w2e_to_sims(embedder.w2e, embedder.vocab, embedder.vocab)
            all_eval_probes, all_eval_candidates_mat = ev.make_all_eval_data(vocab_sims_mat, embedder.vocab)
            ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
                all_eval_probes, all_eval_candidates_mat)
            if config.Eval.verbose:
                print('Shape of all eval data={}'.format(all_eval_candidates_mat.shape))
                print('Shape of down-sampled eval data={}'.format(ev.eval_candidates_mat.shape))
            #
            ev.pos_prob = ev.calc_pos_prob()
            # check that required embeddings exist for eval
            for w in set(ev.row_words + ev.col_words):
                if w not in embedder.w2e:
                    raise KeyError('"{}" required for evaluation "{}" is not in w2e.'.format(scores_p, ev.name))
            # score
            sims_mat = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words)  # sims can have duplicate rows
            ev.score_novice(sims_mat)
            df_rows = ev.train_and_score_expert(embedder)
            # save
            for df_row in df_rows:
                if config.Eval.verbose:
                    print('Saving score to {}'.format(scores_p.relative_to(config.Dirs.remote_root)))
                df = pd.DataFrame(data=[df_row], columns=['exp_score', 'nov_score'] + ev.df_header)
                if not scores_p.parent.exists():
                    scores_p.parent.mkdir(parents=True)
                with scores_p.open('a') as f:
                    df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
            # figs
            if config.Eval.save_figs:
                ev.save_figs(embedder)
            print('-')


def aggregation_job(ev_name):
    print('Aggregating runs data for eval={}..'.format(ev_name))
    ag_matching = Aggregator(ev_name)
    df = ag_matching.make_df()
    p = config.Dirs.remote_root / '{}.csv'.format(ag_matching.ev_name)
    df.to_csv(p)
    print('Done. Saved aggregated data to {}'.format(ev_name, p))
    return df