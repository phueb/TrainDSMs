

from src import config

from src.architectures import comparator
from src.architectures import classifier

from src.evaluators.identification import Identification
from src.evaluators.matching import Matching

from src.embedders.base import w2e_to_sims


def embed_and_evaluate(embedder):
    if config.Embeddings.retrain or not embedder.has_embeddings():
        print('Training runs')
        print('==========================================================================')
        embedder.train()
        if config.Embeddings.save:
            embedder.save_params()
            embedder.save_w2freq()
            embedder.save_w2e()
    else:
        print('Found embeddings at {}'.format(config.Dirs.runs / embedder.time_of_init))
        print('==========================================================================')
        embedder.load_w2e()
    print('Embedding size={}'.format(embedder.w2e_to_embeds(embedder.w2e).shape[1]))
    # evaluate
    for architecture in [
        comparator,
        # classifier
    ]:
        for ev in [
            # Matching(architecture, 'cohyponyms', 'semantic'),
            # Matching(architecture, 'cohyponyms', 'syntactic'),
            # Matching(architecture, 'features', 'is'),
            # Matching(architecture, 'features', 'has'),
            # Matching(architecture, 'nyms', 'syn'),
            # Matching(architecture, 'nyms', 'ant'),
            # Matching(architecture, 'hypernyms'),

            Identification(architecture, 'nyms', 'syn', suffix='_unfiltered'),
            # Identification(architecture, 'nyms', 'ant', suffix='_unfiltered'),
            # Identification(architecture, 'cohyponyms', 'semantic'),  # TODO not enough lures - default to random lures below some threshold?
        ]:
            for rep_id in range(config.Eval.num_reps):
                if config.Eval.retrain or config.Eval.debug or not embedder.completed_eval(ev, rep_id):
                    print('Starting evaluation "{}" replication {}'.format(ev.full_name, rep_id))
                    print('---------------------------------------------')
                    # make eval data - row_words can contain duplicates
                    vocab_sims_mat = w2e_to_sims(embedder.w2e, embedder.vocab, embedder.vocab)
                    all_eval_probes, all_eval_candidates_mat = ev.make_all_eval_data(vocab_sims_mat, embedder.vocab)
                    ev.row_words, ev.col_words, ev.eval_candidates_mat = ev.downsample(
                        all_eval_probes, all_eval_candidates_mat, rep_id)
                    print('Shape of all eval data={}'.format(all_eval_candidates_mat.shape))
                    print('Shape of down-sampled eval data={}'.format(ev.eval_candidates_mat.shape))
                    #
                    ev.pos_prob = ev.calc_pos_prob()
                    # check embeddings for words
                    for p in set(ev.row_words + ev.col_words):
                        if p not in embedder.w2e:
                            raise KeyError('"{}" required for evaluation "{}" is not in w2e.'.format(p, ev.name))
                    # score
                    sims_mat = w2e_to_sims(embedder.w2e, ev.row_words, ev.col_words)  # sims can have duplicate rows
                    ev.score_novice(sims_mat)
                    ev.train_and_score_expert(embedder, rep_id)
                    # figs
                    if config.Eval.save_figs:
                        ev.save_figs(embedder)
                else:
                    print('Embedder completed "{}" replication {}'.format(ev.full_name, rep_id))
                    print('---------------------------------------------')



