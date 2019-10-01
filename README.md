# Two Process NLP

## Background

Research code used to test the two-process theory of semantic development.
The theory is outlined in a CogSci 2019 submission, available [here](https://osf.io/6jfkx/).

The code herein can be used to compare the quality of word various word embedding models trained on a 5M corpus of child-directed speech on several semantic tasks.

## Running the code

A complete experiment can be started by calling `two_process_nlp.job.main()`.
The function takes as input a dictionary representing the parameters of the experiment.
To run the default configuration, pass `two_process_nlp.params.param2default` to `two_process_nlp.job.main()`.
  
The code is designed to run on several machines in parallel, using [LudwigCluster](https://github.com/phueb/LudwigCluster), a command line interface for submitting Python jobs to machines owned by the [UIUC Learning & Language Lab](http://learninglanguagelab.org/).
To use `LudwigCluster`, you must be a member of the lab. 

## Process-1 Word-Embedding Architectures

- W2Vec
- RNN
- HAL, LSA

## Tasks

- Hypernyms
- Cohyponyms
- Meronyms
- Attributes
- Synonyms
- Antonyms

TODO
- Tense/Aspect 
- Student-Teacher, Doctor-Patient

## Evaluation Procedures

### Matching
consists of matching a probe with multiple correct answers

### Identification
consists of identifying correct answer from multiple-choice question

## Process-2 Architectures

- Comparator
- Classifier

## Corpora 

There are two different CHILDES corpora in the repository used as input to the word embedding models. 
`childes-20171212.txt` was generated in the same way as `childes-20180319.txt` except that a few additional steps were taken:
1) all titlecased strings were replaced with a single symbol ("TITLED")
2) all words tagged by the Python package spacy as referring to a person or organization were replaced by a single symbol ("NAME_B"if the word is the first in a span of words referring to a person or organization, and "NAME_I" if it is not the first word in a span of words referring to a person or organization)


## (Optional) Integration with LudwigCluster

* do preprocessing job before executing job.main() - this applies when running job locally or on Ludwig.
* submit tasks folder to shared drive when executing job.main() on Ludwig.
* remove aggregated data before submitting jobs (`rm /media/research_data/2ProcessNLP/2process_data.csv`)