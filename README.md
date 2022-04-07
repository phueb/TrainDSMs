# TrainDSMs

## Background

Research code. Under active development.

## Usage

The user defines multiple jobs (e.g. which DSM to train on which corpus) and submits each job to one of 8 machines owned by the [UIUC Learning & Language Lab](http://learninglanguagelab.org/).
To do so, we use [Ludwig](https://github.com/phueb/Ludwig), a command line interface for communicating the job submission system.
To use `Ludwig`, you must be a member of the lab. 

## DSM Architectures

We examined a number of distributional semantic models (DSMs), including:

- W2Vec
- Simple RNN, LSTM
- Transformer
- LON, CTN (graphical)

## Evaluation

Currently, we are using the `MissingAdjunct` corpus to evaluate the ability of models to infer a missing instrument.
This ability requires compositional generalization, given that the model has never seen the correct answer during training, 
but is provided all the components to make the correct (i.e. structurally licensed) inference.

We assign a hit every time a model predicts the structurally licensed instrument, given a verb phrase (VP).

There are many conditions, such as verb type, theme type, etc.

All evaluations are pooled across model replications (random seed used to sample from the corpus), and items (VPs) of the same type.

### Advanced

If you update the training data, e.g. `MissingAdjunct`, make sure to move the data to the file server:

```python
ludwig -r10 -e ../MissingAdjunct/missingadjunct ../MissingAdjunct/items
```

## Compatibility

Developed using Python 3.7.9 on Ubuntu 18.04

