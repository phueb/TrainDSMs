from itertools import chain

from src.params import CountParams, RNNParams, Word2VecParams, RandomControlParams, GloveParams
from src.params import gen_all_param_combinations
from src.embedders.glove import GloveEmbedder
from src.embedders.rnn import RNNEmbedder
from src.embedders.count import CountEmbedder
from src.embedders.random_control import RandomControlEmbedder
from src.embedders.w2vec import W2VecEmbedder

from src.experiment import embed_and_evaluate


# TODO BUGS:
# TODO nym task doesn't get above chance - make option to filter by POS - otherwise can't get above chance
# TODO neighbors reappear  to bias correct no-answers during testing
# TODO neighbors: neighbors which are actually synonyms or antonyms are not excluded automatically
# TODO adagrad: sgd underperforms adagrad - implement adagrad


embedders = chain(
    (RNNEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(RNNParams)),
    (CountEmbedder(param2id, param2val) for param2id, param2val in gen_all_param_combinations(CountParams)),
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