import math
import networkx as nx
import time

from missingadjunct.corpus import Corpus
from missingadjunct.utils import make_blank_sr_df

from traindsms.dsms.lon import LON

test_corpus = Corpus(include_location=False,
                     include_location_specific_agents=False,
                     num_epochs=1000,
                     seed=1,
                     )
sentences = []
for s in test_corpus.get_sentences():  # a sentence is a string
    tokens = s.split()
    sentences.append(tokens)

dsm = LON(sentences)
dsm.train()


def test_model(dsm):

    df_blank = make_blank_sr_df()
    df_results = df_blank.copy()
    instruments = df_blank.columns[3:]
    assert set(instruments).issubset(test_corpus.vocab)

    # fill in blank data frame with semantic-relatedness scores
    sr_bank = {}
    for verb_phrase, row in df_blank.iterrows():
        verb, theme = verb_phrase.split()
        scores = []

        if verb not in sr_bank:
            sr_bank[verb] = dsm.activation_spreading_analysis(verb, instruments, [])
        if theme not in sr_bank:
            sr_bank[theme] = dsm.activation_spreading_analysis(theme, instruments, [])

        for instrument in instruments:  # instrument columns start after the 3rd column

            sr = math.log(sr_bank[verb][instrument] * sr_bank[theme][instrument])
            scores.append(sr)

        # collect sr scores in new df
        df_results.loc[verb_phrase] = [row['verb-type'], row['theme-type'], row['phrase-type']] + scores

    print(df_results.loc['preserve pepper'].round(3))
    df_results.to_csv('df_test_lon.csv')


test_model(dsm)


def test_activation():
    """
    This is an attempt to get access to activation-spreading distance for all paths. It failed as number of possible paths
    could be very large basing on the graph structure, and time complexity could be exponential
    """
    diameter = nx.algorithms.distance_measures.diameter(dsm.undirected_network)
    print(diameter)

    for node in dsm.node_list:
        start_time = time.time()
        dsm.get_path_distance(node, 1, [])
        end_time = time.time()
        time_getting_paths = end_time - start_time
        print()
        print('{} used to get paths from {}'.format(time_getting_paths, node))
        print()
        node_paths = dsm.path_distance_dict[node]
        for target_node in node_paths:
            print(node, target_node)
            print(node_paths[target_node])

