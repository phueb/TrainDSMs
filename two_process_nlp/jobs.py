import pandas as pd
import sys
from collections import Counter, OrderedDict
from itertools import islice
import pyprind
from spacy.lang.en import English

from two_process_nlp import config
from two_process_nlp.aggregator import Aggregator
from two_process_nlp.architectures import comparator, classifier, aligner, extractor
from two_process_nlp.evaluators.matching import Matching
from two_process_nlp.evaluators.identification import Identification
from two_process_nlp.job_utils import init_embedder
from two_process_nlp.job_utils import w2e_to_sims
from two_process_nlp.job_utils import save_corpus_data
from two_process_nlp.job_utils import move_scores_to_server


nlp = English()


def preprocessing_job(num_vocab=None, skip_docs=False, local=False):
    num_vocab = config.Corpus.num_vocab or num_vocab
    #
    docs = []
    w2freq = Counter()
    # tokenize + count words
    p = config.LocalDirs.corpora / '{}.txt'.format(config.Corpus.name)
    with p.open('r') as f:
        texts = f.read().splitlines()  # removes '\n' newline character
    num_texts = len(texts)
    print('\nTokenizing {} docs...'.format(num_texts))
    pbar = pyprind.ProgBar(num_texts, stream=sys.stdout)
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
        vocab = list(deterministic_w2f.keys())
    else:
        vocab = list(islice(deterministic_w2f.keys(), 0, num_vocab - 1))
        vocab.append(config.Corpus.UNK)
    vocab = list(sorted(vocab))
    if num_vocab is None:  # if no vocab specified, use the whole corpus
        num_vocab = len(vocab)
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
    # save to file server
    print('Sending results from corpus preprocessing to file-server...')
    save_corpus_data(deterministic_w2f, vocab, docs, numeric_docs, skip_docs, local)


def main_job(param2val):
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
    # process 1
    embedder = init_embedder(param2val)
    print('Training process 1...')
    sys.stdout.flush()
    if not config.Eval.debug:
        embedder.train()
    else:
        embedder.load_w2e(remote=False)
    embedder.save_w2e() if config.Embeddings.save_w2e else None
    # process 2
    for architecture in [
        comparator,
        classifier,
        extractor,
        aligner,
    ]:
        for ev in [
            # Matching(architecture, 'cohyponyms', 'semantic'),
            # Matching(architecture, 'cohyponyms', 'syntactic'),
            # Matching(architecture, 'features', 'is'),
            # Matching(architecture, 'features', 'has'),
            # Matching(architecture, 'nyms', 'syn', suffix='_jw'),
            # Matching(architecture, 'nyms', 'ant', suffix='_jw'),
            # Matching(architecture, 'hypernyms'),
            # Matching(architecture, 'events'),

            Identification(architecture, 'nyms', 'syn', suffix='_jw'),
            Identification(architecture, 'nyms', 'ant', suffix='_jw'),
        ]:
            if ev.suffix != '':
                print('WARNING: Using task file suffix "{}".'.format(ev.suffix))
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
                    raise KeyError('"{}" required for evaluation "{}" is not in w2e.'.format(w, ev.name))
            # score
            sims_mat = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words)  # sims can have duplicate rows
            novice_scores = ev.score_novice(sims_mat)
            if config.Eval.only_process1:
                expert_scores = []
                control_scores = []
            else:
                expert_scores = ev.train_and_score_expert(embedder, shuffled=False)
                control_scores = ev.train_and_score_expert(embedder, shuffled=True) if \
                    config.Eval.shuffled_control else []  # TODO test
            # save
            for scores, process in [(novice_scores, 'novice'),
                                    (expert_scores, 'expert'),
                                    (control_scores, 'control')]:
                if not scores:
                    print('Did not calculate {} scores'.format(process))
                    continue
                print('process "{}" best score={:2.2f}'.format(process, max([s[0] for s in scores])))
                scores_p = ev.make_scores_p(embedder.location, process)
                df = pd.DataFrame(data=scores,
                                  columns=['score'] + ev.df_header + ['num_epochs'])  # scores is list of lists
                if not scores_p.parent.exists():
                    scores_p.parent.mkdir(parents=True)
                with scores_p.open('w') as f:
                    df.to_csv(f, index=False, na_rep='None')  # otherwise NoneTypes are converted to empty strings
            print('-')
    # move scores to file server
    if not config.Eval.debug and job_name != 'test':
            move_scores_to_server(param2val, embedder.location)


def aggregation_job(verbose=True):
    print('Aggregating runs data...')
    ag = Aggregator()
    df = ag.make_df(load_from_file=False, verbose=verbose)
    p_with_date = config.RemoteDirs.root / ag.df_name_with_date
    p = config.RemoteDirs.root / ag.df_name
    df.to_csv(p_with_date, index=False)
    df.to_csv(p, index=False)
    print('Done. Saved aggregated data to {}'.format(p))
    return df
