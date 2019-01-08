import sys, os
from random import shuffle

cat_dict = {}

nym_filename = sys.argv[1]
vocab_filename = sys.argv[2]

# this is used to make the output file named correctly
CORPUS_NAME = 'childes-20180319_4096'

# read in the nym lists into a dict, with each nym catagory pointing to a tuple of two lists, one for each polar side of the nym category
# example cat_dict['happiness'] = ([happiness, joy, elation], [sadness, grief])
f = open(nym_filename)
for line in f:

	data = (line.strip().strip('\n').strip()).split()
	word = data[0]
	gram = data[1]
	cat = data[2]
	group = int(data[3])

	if cat not in cat_dict:
		cat_dict[cat] = ([],[])

	if group == 1:
		if word not in cat_dict[cat][0]:
			cat_dict[cat][0].append(word)

	elif group == -1:
		if word not in cat_dict[cat][1]:
			cat_dict[cat][1].append(word)
f.close()

# read in the vocab list
vocab_dict = {}
i = 0
f = open(vocab_filename)
for line in f:
	data = line.strip().strip('\n').strip()
	vocab_dict[data] = i
	i += 1
f.close()

# turn nym dict into an order-shuffled tuple list (key, value)
cat_list = list(cat_dict.items())
shuffle(cat_list)


# generate the final nym lists
os.mkdir('nyms')
os.mkdir('nyms/syn')
os.mkdir('nyms/ant')

syn_outfile = open('nyms/syn/' + CORPUS_NAME + '.txt', 'w')
ant_outfile = open('nyms/ant/' + CORPUS_NAME + '.txt', 'w')

syn_dict = {}
ant_dict = {}

for cat_data in cat_list:
	cat = cat_data[0]
	group1 = cat_data[1][0]
	group2 = cat_data[1][1]

	for probe in group1:
		if probe in vocab_dict:
			if probe not in syn_dict:
				output_list = [probe]
				for nym in group1:
					if probe != nym:
						if nym in vocab_dict:
							output_list.append(nym)

				if len(output_list) > 1:
					output_string = ' '.join(output_list)
					syn_outfile.write(output_string + '\n')
					syn_dict[word] = 1

	for probe in group2:
		if probe in vocab_dict:
			if probe not in ant_dict:
				output_list = [probe]
				for nym in group2:
					if probe != nym:
						if nym in vocab_dict:
							output_list.append(nym)
				if len(output_list) > 1:
					output_string = ' '.join(output_list)
					ant_outfile.write(output_string + '\n')
					ant_dict[word] = 1
syn_outfile.close()
ant_outfile.close()



