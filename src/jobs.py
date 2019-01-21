import pandas as pd
import sys
from collections import Counter, OrderedDict
from itertools import islice
import pyprind
from spacy.lang.en import English
from shutil import copyfile

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


def preprocessing_job(num_vocab=config.Corpus.num_vocab):
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


def backup_job(param_name, job_name, allow_rewrite):
    """
    function is not imported from ludwigcluster because this would require dependency on worker.
    this informs LudwigCluster that training has completed (backup is only called after training completion)
    copies all data created during training to backup_dir.
    Uses custom copytree fxn to avoid permission errors when updating permissions with shutil.copytree.
    Copying permissions can be problematic on smb/cifs type backup drive.
    """
    src = config.Dirs.runs / param_name / job_name
    dst = config.Dirs.backup / param_name / job_name
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True)
    copyfile(str(config.Dirs.runs / param_name / 'param2val.yaml'),
             str(config.Dirs.backup / param_name / 'param2val.yaml'))  # need to copy param2val.yaml

    def copytree(s, d):
        d.mkdir(exist_ok=allow_rewrite)  # set exist_ok=True if dst is partially missing files whcih exist in src
        for i in s.iterdir():
            s_i = s / i.name
            d_i = d / i.name
            if s_i.is_dir():
                copytree(s_i, d_i)
            else:
                copyfile(str(s_i), str(d_i))  # copyfile works because it doesn't update any permissions
    # copy
    print('Backing up data...  DO NOT INTERRUPT!')
    try:
        copytree(src, dst)
    except PermissionError:
        print('Backup failed. Permission denied.')
    except FileExistsError:
        print('Already backed up {}'.format(dst))
    else:
        print('Backed up data to {}'.format(dst))


def embedder_job(param2val):
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
            if config.Eval.only_stage1:
                expert_scores = None
                control_scores = None
            else:
                expert_scores = ev.train_and_score_expert(embedder, shuffled=False)
                control_scores = ev.train_and_score_expert(embedder, shuffled=True)
            # save
            for scores, stage in [(novice_scores, 'novice'), (expert_scores, 'expert'), (control_scores, 'control')]:
                print('stage "{}" best score={:2.2f}'.format(stage, max([s[0] for s in scores])))
                # check
                p = ev.make_scores_p(embedder.location, stage)
                try:
                    p.parent.exists()
                except OSError:
                    raise OSError('{} is not reachable. Check VPN or mount drive.'.format(p))
                if p.exists():
                    print(
                        'WARNING: {} should not exist.'
                        ' This is likely a failure of ludwigcluster to distribute tasks.'.format(p))
                    p.unlink()
                # save
                df = pd.DataFrame(data=scores, columns=['score'] + ev.df_header)  # scores is list of lists
                if not p.parent.exists():
                    p.parent.mkdir(parents=True)
                with p.open('w') as f:
                    df.to_csv(f, index=False, na_rep='None')  # otherwise NoneTypes are converted to empty strings
            print('-')


def aggregation_job(verbose=True):
    print('Aggregating runs data...')
    ag = Aggregator()
    df = ag.make_df(load_from_file=False, verbose=verbose)
    p = config.Dirs.remote_root / ag.df_name
    df.to_csv(p)
    print('Done. Saved aggregated data to {}'.format(p))
    return df