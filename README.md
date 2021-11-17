# TrainDSMs

## Background

Research code. Under active development.


## Usage

The user defines multiple jobs (e.g. which DSM to train on which corpus) and submits each job to one of 8 machines owned by the [UIUC Learning & Language Lab](http://learninglanguagelab.org/).
To do so, we use [Ludwig](https://github.com/phueb/Ludwig), a command line interface for communicating the job submission system.
To use `Ludwig`, you must be a member of the lab. 

### Advanced

If you update the training data, e.g. `MissingAdjunct`, make sure to move the data to the file server:

```python
ludwig -r10 -e ../MissingAdjunct/missingadjunct ../MissingAdjunct/items
```

## DSM Architectures

- W2Vec
- Simple RNN, LSTM
- Transformer
- LON, CTN (graphical)


## Evaluations
All evaluations are pooled across model replications (random seed used to sample from the corpus), and items (VPS) of the same type.

### First-Rank
We assign a hit every time a model predicts that the correct instrument is the most related to a VP.

### Intra-Instrument Variance
To test whether a model differntiates between two or more instruments that are equally correct (their rank should be tied), we use an anlysis of variance comparing the variance between predicted sematnic relatedness scores assigned to the correct instruments and scores sassigend to all other instruments.

## Compatibility

Developed using Python 3.7.9 on Ubuntu 18.04

