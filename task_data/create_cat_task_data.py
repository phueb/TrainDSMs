from operator import itemgetter

from src import config
from src.utils import make_w2freq


CORPUS_NAME = 'childes-20180319'

w2freq = make_w2freq(CORPUS_NAME)
sorted_freq_tuples = w2freq.most_common()


for vocab_size in config.Tasks.vocab_sizes:
    vocab_dict = {}
    for i in range(vocab_size-1):
        freq_tuple = sorted_freq_tuples[i]
        vocab_dict[freq_tuple[0]] = freq_tuple[1]
    for task in ['semantic_categorization', 'syntactic_categorization']:
        task_input_file = open(task+".txt")
        task_output_file = open(task+"_"+str(vocab_size)+".txt", 'w')
        for line in task_input_file:
            data = (line.strip().strip('\n').strip()).split()
            if data[0] in vocab_dict:
                task_output_file.write(line)
        task_input_file.close()
        task_output_file.close()