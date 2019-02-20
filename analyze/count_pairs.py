
from two_stage_nlp import config
from analyze.utils import make_task_name2_probe_data


CORPUS_NAME = config.Corpus.name
NUM_VOCAB = config.Corpus.num_vocab

# data
task_name2_probe_data = make_task_name2_probe_data(
    corpus_name=config.Corpus.name, num_vocab=config.Corpus.num_vocab)

# print
num_row_words_sum = 0
num_total_possible_sum = 0
pos_prob_sum = 0
neg_prob_sum = 0
num_tasks = 0
for task_name, probe_data in task_name2_probe_data.items():
    num_row_words, num_unique_probes, num_total_possible, num_pos, num_neg, num_pos_possible, diff = probe_data
    #
    print(task_name)
    print('Num unique probes={}'.format(num_unique_probes))
    print('Average num relata per probe={}'.format(num_pos_possible / num_row_words))
    print('Num unique total possible={:,}'.format(num_total_possible))
    print('Num unique positive pairs={:,}'.format(num_pos))
    print('Num unique positive possible={:,}'.format(num_pos_possible))
    print('Positive prob={:.3f}'.format(num_pos / num_total_possible))
    print('Num unique negative possible={:,}'.format(num_neg))
    print('Negative prob={:.3f}'.format(num_neg / num_total_possible))
    #
    if diff > 0:
        print('WARNING: Difference={}. Duplicates pairs exist'.format(diff))
    print()
    # collect
    num_row_words_sum += num_row_words
    num_total_possible_sum += num_total_possible
    pos_prob_sum += num_pos / num_total_possible
    neg_prob_sum += num_neg / num_total_possible
    num_tasks += 1

print('Average num row_words per task={}'.format(num_row_words_sum / num_tasks))
print('Average num_total_possible per task={}'.format(num_total_possible_sum / num_tasks))
print('Average pos_prob per task={}'.format(pos_prob_sum / num_tasks))
print('Average neg_prob per task={}'.format(neg_prob_sum / num_tasks))
print('Average neg_pos_ratio={}'.format((neg_prob_sum / num_tasks) / (pos_prob_sum / num_tasks)))