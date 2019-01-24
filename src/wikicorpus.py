import json
import spacy
import time


class Corpus():
    def __init__(self):
        self.name = 0
        self.path = 0
        self.output_filename = 0

    def generate_corpus_from_doc_dir(self, path, name):
        self.name = name
        self.path = path

    def generate_wiki_corpus(self, path, name):
        self.name = name
        self.path = path
        nlp = spacy.load('en')

        infile = open(self.path)
        num_docs = 0
        print('Counting documents')
        for line in infile:
            num_docs += 1
            if num_docs % 10000 == 0:
                print('     {} docs'.format(num_docs))
        infile.close()

        infile = open(self.path)
        text_outfile = open(name+"_text.txt", 'w')
        titles_outfile = open(name + "_titles.txt", 'a')
        freqs_outfile = open(name + "_freqs.txt", "w")

        freq_dict = {}
        linecounter = 0
        token_counter = 0
        for line in infile:
            start = time.time()
            json1_data = json.loads(line)

            title = json1_data['title'].lower()
            text = json1_data['text']

            spacy_doc = nlp(text)  # TODO use Tokenizer only to improve performance
            doc_size = len(spacy_doc)
            token_counter += doc_size
            titles_outfile.write('{} {}\n'.format(title, doc_size))

            output_list = []
            for token in spacy_doc:
                string_token = token.text.lower()
                string_token = string_token.strip().strip('\n').strip()
                if len(string_token) > 0:
                    if string_token not in freq_dict:
                        freq_dict[string_token] = 1
                    freq_dict[string_token] += 1
                    output_list.append(string_token)
            output_string = ' '.join(output_list) + '\n'
            text_outfile.write(output_string)

            if linecounter % 10 == 0:
                took = time.time() - start
                start = time.time()
                perc = float(linecounter) / num_docs
                print('{}/{}  {:0.4f}  {:0.3f} sec.'.format(linecounter, num_docs, perc, took))
            linecounter += 1

        text_outfile.close()
        titles_outfile.close()

        for item in freq_dict:
            freqs_outfile.write('{} {}\n'.format(item, freq_dict[item]))
        freqs_outfile.close()









