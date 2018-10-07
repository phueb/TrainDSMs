import pyprind
import json
import operator

REPLACE_DICT = {}
REPLACE_DICT['\n'] = ''
REPLACE_DICT['\r'] = ''
REPLACE_DICT['"'] = ''
REPLACE_DICT['['] = ''
REPLACE_DICT[']'] = ''
REPLACE_DICT['('] = '( '
REPLACE_DICT[')'] = ' )'
REPLACE_DICT['{'] = '{ '
REPLACE_DICT['}'] = ' }'
REPLACE_DICT[','] = ' ,'
REPLACE_DICT['!'] = ' !'
REPLACE_DICT['?'] = ' ?'
REPLACE_DICT['.'] = ' .'

class Corpus():
    def __init__(self):
        self.name = 0
        self.path = 0
        self.output_filename = 0

    def generate_corpus_from_doc_dir(self, path, name):
        self.name = name
        self.path = path

    def generate_wiki_corpus(self, path, name, num_docs=None):
        self.name = name
        self.path = path

        infile = open(self.path)
        text_outfile = open(name+"_text.txt", 'w')
        titles_outfile = open(name + "_titles.txt", 'w')
        freqs_outfile = open(name + "_freqs.txt", "w")

        freq_dict = {}

        linecounter = 0
        for line in infile:
            json1_data = json.loads(line)
            title = json1_data['title'].lower()
            text = json1_data['text']
            text = text.lower()
            tokens = text.split()
            final_tokens = []

            for token in tokens:
                if token in REPLACE_DICT:
                    replacement = REPLACE_DICT[token]
                    if len(replacement) > 0:
                        final_tokens.append(replacement)
                        if not replacement in freq_dict:
                            freq_dict[replacement] = 1
                        freq_dict[replacement] += 1
                else:
                    final_tokens.append(token)
                    if not token in freq_dict:
                        freq_dict[token] = 1
                    freq_dict[token] += 1

                output_string = ' '.join(final_tokens) + '\n'

            text_outfile.write(output_string)
            titles_outfile.write(title + '\n')

            linecounter += 1
            if linecounter % 1000 == 0:
                print("Finished {} lines".format(linecounter))

        text_outfile.close()
        titles_outfile.close()

        sorted_list = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
        for item in sorted_list:
            freqs_outfile.write('{} {}\n'.format(item[0], item[1]))
        freqs_outfile.close()









