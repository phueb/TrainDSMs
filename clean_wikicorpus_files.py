import re
from pathlib import Path

'''
clean text files from wikicorpus project (http://www.cs.upc.edu/~nlp/wikicorpus/
then, save to single corpus file
'''


NUM_ARTICLES = 100  # 4 -> 200k words
EXCLUDE = ['\n', 'endofarticle.']

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_special_chars(text, char_list):
    for char in char_list:
        text=text.replace(char,'')
    return text.replace(u'\xa0', u' ')


def remove_non_printed_chars(string):
    reg = re.compile('[^a-zA-Zа-яА-ЯёЁ,.;\s\'-]')
    return reg.sub(' <NOTPR> ', string)


def remove_trailing(string):
    # remove extra spaces, remove trailing spaces, lower the case
    return re.sub('\s+',' ',string).strip()

num_lines = 0
num_words = 0
with open('wikicorpus_cleaned.txt', 'w') as out_f:
    wikicorpus_dir = Path('/home/ph/Downloads/raw.en')
    for p in wikicorpus_dir.glob('englishText*'):
        # load
        with p.open('rb') as in_f:
            content = in_f.read().lower().decode("utf-8", 'replace')
        articles = re.split('<doc.*>', content)
        # clean
        for article in articles[:NUM_ARTICLES]:
            if article != '':
                cleaned = remove_trailing(
                    remove_non_printed_chars(
                        remove_special_chars(
                            remove_html_tags(article), EXCLUDE)))
                # save
                out_f.write(cleaned)
                out_f.write('\n')
                #
                num_words += len(cleaned.split())
                num_lines += 1
        print(num_lines, '{:,}'.format(num_words))