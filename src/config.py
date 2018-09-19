
class CategoryClassifier:
    num_epochs = 1
    mb_size = 1
    num_acts_samples = 50
    num_steps_to_eval = 200
    learning_rate = 0.005
    num_hiddens = 100
    rep_type = 'proto'
    shuffle_cats = False
    mode = 'semantic'


class Corpus:
    name = 'childes-20180319_terms'
