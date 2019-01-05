from itertools import chain

from src import config
from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams, GloveParams
from src.params import gen_all_param_combinations
from src.embedders.glove import GloveEmbedder
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.experiment import embed_and_evaluate


# todo:
# TODO nym task doesn't get above chance - make option to filter by POS - otherwise can't get above chance
# TODO neighbors: make sure they don't reappear often to bias correct no-answers durign testing
# TODO neighbors: manually exclude neighbors which are actually synonyms or antonyms but not excluded automatically
# TODO adagrad: sgd underperforms adagrad - implement adagrad
# TODO cuda error unspecified launch failure - only occurs after completion of 1 training on GPU - flush memory? - tensorflow grabs all gpu memory
# TODO confusion matrix for syntactic cohyponym task - balance syntactic category sizes?
# TODO there are only 200 semantic cohyponyms for tasa - make more
# TODO use all vocab items for syntactic cohyponym task
# TODO show that large embed_size leads to overfitting and wrose expert performance


embedders = chain(
    (CountEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(CountParams)),
    (RNNEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RNNParams)),
    # (GloveEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(GloveParams)),
    (RandomControlEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RandomControlParams)),
    (W2VecEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(Word2VecParams)),
)

# run full experiment
runtime_errors = []
while True:
    # get embedder
    try:
        embedder = next(embedders)
    except RuntimeError as e:
        print('WARNING: embedder raised RuntimeError:')
        print(e)
        runtime_errors.append(e)
        continue
    except StopIteration:
        print('Finished experiment')
        for e in runtime_errors:
            print('with RunTimeError:')
            print(e)
        break
    # embed and evaluate embedder
    embed_and_evaluate(embedder)