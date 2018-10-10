from operator import itemgetter

corpus = '../corpora/childes-20180319.txt'
task_list = ['semantic_categorization', 'syntactic_categorization']
size_list = [4096, 8192, 16384]

freq_dict = {}
f = open(corpus)
for line in f:
    data = (line.strip().strip('\n').strip()).split()
    for token in data:
        if token in freq_dict:
            freq_dict[token] += 1
        else:
            freq_dict[token] = 1
f.close()
sorted_freq_tuples = sorted(freq_dict.items(), key=itemgetter(1), reverse=True)

for vocab_size in size_list:
    vocab_dict = {}
    for i in range(vocab_size-1):
        freq_tuple = sorted_freq_tuples[i]
        vocab_dict[freq_tuple[0]] = freq_tuple[1]
    for task in task_list:
        task_input_file = open(task+".txt")
        task_output_file = open(task+"_"+str(vocab_size)+".txt", 'w')
        for line in task_input_file:
            data = (line.strip().strip('\n').strip()).split()
            if data[0] in vocab_dict:
                task_output_file.write(line)
        task_input_file.close()
        task_output_file.close()