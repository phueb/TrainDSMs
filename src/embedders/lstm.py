import time
import torch
import pyprind
import numpy as np
import sys

from src.embedders import EmbedderBase
from src import config
from src.utils import matrix_to_w2e


# TODO  is torch.utils.data useful here?


class LSTMEmbedder(EmbedderBase):
    def __init__(self, corpus_name, ):
        super().__init__(corpus_name, 'lstm')
        self.model = LSTMModel()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.LSTM.initital_lr)  # TODO SparseAdam
        self.model.cuda()

    @staticmethod
    def gen_windows(token_ids):
        # yield num_steps matrices where each matrix contains windows of size num_steps
        remainder = len(token_ids) % config.LSTM.num_steps
        for i in range(config.LSTM.num_steps):
            seq = np.roll(token_ids, i)  # rightward
            seq = seq[:-remainder]
            x = np.reshape(seq, (-1, config.LSTM.num_steps))
            y = np.roll(x, -1)
            yield i, x, y

    def gen_batches(self, token_ids, batch_size, verbose=True):  # TODO test fn
        batch_id = 0
        for window_id, x, y in self.gen_windows(token_ids):  # more memory efficient not to create all windows in data
            # exclude some rows to split x and y evenly by batch size
            shape0 = len(x)
            num_excluded = shape0 % batch_size
            if num_excluded > 0:  # in case mb_size = 1
                x = x[:-num_excluded]
                y = y[:-num_excluded]
            shape0_adj = shape0 - num_excluded
            # split into batches
            num_batches = shape0_adj // batch_size
            if verbose:
                print('Excluding {} windows due to fixed batch size'.format(num_excluded))
                print('{}/{} Generating {:,} batches with size {}...'.format(
                window_id + 1, config.LSTM.num_steps, num_batches, batch_size))
            for x_b, y_b in zip(np.vsplit(x, num_batches),
                                np.vsplit(y, num_batches)):
                yield batch_id, x_b, y_b[:, -1]
                batch_id += 1

    def calc_pp(self, numeric_docs):
        print('Calculating perplexity...')
        self.model.eval()
        self.model.batch_size = 1  # TODO probably better to do on CPU - or find batch size that excludes least samples
        errors = 0
        batch_id = 0
        token_ids = np.hstack(numeric_docs)
        num_windows = len(token_ids)
        pbar = pyprind.ProgBar(num_windows, stream=sys.stdout)
        for batch_id, x_b, y_b in self.gen_batches(token_ids, self.model.batch_size, verbose=False):
            pbar.update()
            inputs = torch.cuda.LongTensor(x_b.T)  # requires [num_steps, mb_size]
            targets = torch.cuda.LongTensor(y_b)
            hidden = self.model.init_hidden()  # this must be here to re-init graph
            outputs, hidden = self.model(inputs, hidden)
            #
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(outputs.unsqueeze_(0), targets)  # need to add dimension due to mb_size = 1
            errors += loss.item()
        res = np.exp(errors / batch_id + 1)
        return res  # TODO test

    def train_epoch(self, numeric_docs, lr):
        start_time = time.time()
        self.model.train()
        self.model.batch_size = config.LSTM.batch_size
        # shuffle and flatten
        if config.LSTM.shuffle_per_epoch:
            np.random.shuffle(numeric_docs)
        token_ids = np.hstack(numeric_docs)
        for batch_id, x_b, y_b in self.gen_batches(token_ids, self.model.batch_size):
            # forward step
            inputs = torch.cuda.LongTensor(x_b.T)  # requires [num_steps, mb_size]
            targets = torch.cuda.LongTensor(y_b)
            hidden = self.model.init_hidden()  # this must be here to re-init graph
            outputs, hidden = self.model(inputs, hidden)
            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero  # TODO why put this here?
            loss = self.criterion(outputs, targets)
            loss.backward()
            if config.LSTM.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.LSTM.grad_clip)
                for p in self.model.parameters():
                    p.data.add_(-lr, p.grad.data)
            else:
                self.optimizer.step()
            # console
            if batch_id % config.LSTM.num_eval_steps == 0:
                xent_error = loss.item()
                pp = np.exp(xent_error)
                secs = time.time() - start_time
                print("batch {:,} perplexity: {:8.2f} | seconds elapsed in epoch: {:,.0f} ".format(batch_id, pp, secs))

    def train(self):
        # split data
        train_numeric_docs = []
        valid_numeric_docs = []
        test_numeric_docs = []
        for doc in self.numeric_docs:
            if np.random.binomial(1, config.LSTM.train_percent):
                train_numeric_docs.append(doc)
            else:
                if np.random.binomial(1, 0.5):  # split valid and test docs evenly
                    valid_numeric_docs.append(doc)
                else:
                    test_numeric_docs.append(doc)  # TODO test splitting
        print('Num docs in train {} valid {} test {}'.format(
            len(train_numeric_docs), len(valid_numeric_docs), len(test_numeric_docs)))
        # train loop
        lr = config.LSTM.initital_lr
        for epoch in range(config.LSTM.num_epochs):
            print('/Starting epoch {}'.format(epoch))
            lr_decay = config.LSTM.lr_decay_base ** max(epoch - config.LSTM.num_epochs_with_flat_lr, 0)
            lr = lr * lr_decay  # decay lr if it is time
            self.train_epoch(train_numeric_docs, lr)
            print('\nValidation perplexity at epoch {}: {:8.2f}'.format(
                epoch, self.calc_pp(valid_numeric_docs)))
        print('Test Perplexity: {:8.2f}'.format(self.calc_pp(test_numeric_docs)))
        embed_mat = self.model.wx.weight.detach().cpu().numpy()  # TODO
        w2e = matrix_to_w2e(embed_mat, self.vocab)
        return w2e


class LSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_size = config.LSTM.batch_size
        self.dropout = torch.nn.Dropout(config.LSTM.dropout_prob)  # TODO is this useful?
        self.wx = torch.nn.Embedding(config.Corpora.num_vocab, config.LSTM.embed_size)
        self.lstm = torch.nn.LSTM(input_size=config.LSTM.embed_size,
                                  hidden_size=config.LSTM.embed_size,
                                  num_layers=config.LSTM.num_layers)
        self.wy = torch.nn.Linear(in_features=config.LSTM.embed_size,
                                  out_features=config.Corpora.num_vocab)
        self.init_weights()

    def init_weights(self):
        self.wx.weight.data.uniform_(-config.LSTM.embed_init_range, config.LSTM.embed_init_range)
        self.wy.bias.data.fill_(0.0)
        self.wy.weight.data.uniform_(-config.LSTM.embed_init_range, config.LSTM.embed_init_range)

    def init_hidden(self):
        weight = next(self.parameters()).data
        return (torch.autograd.Variable(weight.new(config.LSTM.num_layers,
                                                   self.batch_size,
                                                   config.LSTM.embed_size).zero_()),
                torch.autograd.Variable(weight.new(config.LSTM.num_layers,
                                                   self.batch_size,
                                                   config.LSTM.embed_size).zero_()))

    def forward(self, inputs, hidden):  # expects [num_steps, mb_size] tensor
        embeds = self.wx(inputs)
        outputs, hidden = self.lstm(embeds, hidden)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[-1])
        logits = self.wy(self.dropout(final_outputs))
        return logits, hidden  # TODO don't need to return hidden

