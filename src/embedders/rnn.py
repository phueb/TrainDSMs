import time
import torch
import pyprind
import numpy as np
import sys

from src.embedders.base import EmbedderBase
from src.params import RNNParams
from src import config


# TODO  is torch.utils.data useful here?

class RNNEmbedder(EmbedderBase):
    def __init__(self, param2val, job_name):
        super().__init__(param2val, job_name)
        self.rnn_type = param2val['rnn_type']
        self.embed_size = param2val['embed_size']
        self.train_percent = param2val['train_percent']
        self.num_eval_steps = param2val['num_eval_steps']
        self.shuffle_per_epoch = param2val['shuffle_per_epoch']
        self.embed_init_range = param2val['embed_init_range']
        self.dropout_prob = param2val['dropout_prob']
        self.num_layers = param2val['num_layers']
        self.num_steps = param2val['num_steps']
        self.batch_size = param2val['batch_size']
        self.num_epochs = param2val['num_epochs']
        self.learning_rate = param2val['learning_rate']
        self.grad_clip = param2val['grad_clip']
        #
        self.name = self.rnn_type
        self.model = None
        self.criterion = None
        self.optimizer = None

    def gen_windows(self, token_ids):
        # yield num_steps matrices where each matrix contains windows of size num_steps
        remainder = len(token_ids) % self.num_steps
        for i in range(self.num_steps):
            seq = np.roll(token_ids, i)  # rightward
            seq = seq[:-remainder] if remainder != 0 else seq
            x = np.reshape(seq, (-1, self.num_steps))
            y = np.roll(x, -1)
            yield i, x, y

    def gen_batches(self, token_ids, batch_size, verbose):
        batch_id = 0
        for step_id, x, y in self.gen_windows(token_ids):  # more memory efficient not to create all windows in data
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
                    step_id + 1, self.num_steps, num_batches, batch_size))
            for x_b, y_b in zip(np.vsplit(x, num_batches),
                                np.vsplit(y, num_batches)):
                yield batch_id, x_b, y_b[:, -1]
                batch_id += 1

    def calc_pp(self, numeric_docs, verbose):
        if verbose:
            print('Calculating perplexity...')
        self.model.eval()
        self.model.batch_size = 1  # TODO find batch size that excludes least samples
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
            logits = self.model(inputs, hidden)
            #
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(logits.unsqueeze_(0), targets)  # need to add dimension due to mb_size = 1
            errors += loss.item()
        res = np.exp(errors / batch_id + 1)
        return res

    def train_epoch(self, numeric_docs, lr, verbose):
        start_time = time.time()
        self.model.train()
        self.model.batch_size = self.batch_size
        # shuffle and flatten
        if self.shuffle_per_epoch:
            np.random.shuffle(numeric_docs)
        token_ids = np.hstack(numeric_docs)
        for batch_id, x_b, y_b in self.gen_batches(token_ids, self.model.batch_size, verbose):
            # forward step
            inputs = torch.cuda.LongTensor(x_b.T)  # requires [num_steps, mb_size]
            targets = torch.cuda.LongTensor(y_b)
            hidden = self.model.init_hidden()  # this must be here to re-init graph
            logits = self.model(inputs, hidden)
            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero  # TODO why put this here?
            loss = self.criterion(logits, targets)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                for p in self.model.parameters():
                    p.data.add_(-lr, p.grad.data)
            else:
                self.optimizer.step()
            # console
            if batch_id % self.num_eval_steps == 0 and verbose:
                xent_error = loss.item()
                pp = np.exp(xent_error)
                secs = time.time() - start_time
                print("batch {:,} perplexity: {:8.2f} | seconds elapsed in epoch: {:,.0f} ".format(batch_id, pp, secs))

    def train(self, verbose=True):
        # init
        self.model = TorchRNN(self.rnn_type, self.num_layers, self.embed_size, self.batch_size, self.embed_init_range)
        self.model.cuda()  # call this before constructing optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate[0])
        # split data
        train_numeric_docs = []
        valid_numeric_docs = []
        test_numeric_docs = []
        for doc in self.numeric_docs:
            if np.random.binomial(1, self.train_percent):
                train_numeric_docs.append(doc)
            else:
                if np.random.binomial(1, 0.5):  # split valid and test docs evenly
                    valid_numeric_docs.append(doc)
                else:
                    test_numeric_docs.append(doc)
        print('Num docs in train {} valid {} test {}'.format(
            len(train_numeric_docs), len(valid_numeric_docs), len(test_numeric_docs)))
        print('Training rnn...')
        # train loop
        lr = self.learning_rate[0]  # initial
        decay = self.learning_rate[1]
        num_epochs_without_decay = self.learning_rate[2]
        pbar = pyprind.ProgBar(self.num_epochs, stream=sys.stdout)
        for epoch in range(self.num_epochs):
            if verbose:
                print('/Starting epoch {} with lr={}'.format(epoch, lr))
            lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
            lr = lr * lr_decay  # decay lr if it is time
            self.train_epoch(train_numeric_docs, lr, verbose)
            if verbose:
                print('\nValidation perplexity at epoch {}: {:8.2f}'.format(
                    epoch, self.calc_pp(valid_numeric_docs, verbose)))
            else:
                pbar.update()
        if verbose:
            print('Test Perplexity: {:8.2f}'.format(self.calc_pp(test_numeric_docs, verbose)))
        wx = self.model.wx.weight.detach().cpu().numpy()
        embed_mat = self.standardize_embed_mat(wx)
        self.w2e = self.embeds_to_w2e(embed_mat, self.vocab)


class TorchRNN(torch.nn.Module):
    def __init__(self, rnn_type, num_layers, embed_size, batch_size, embed_init_range):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.embed_init_range = embed_init_range
        #
        self.wx = torch.nn.Embedding(config.Corpus.num_vocab, self.embed_size)
        if self.rnn_type == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.rnn_type == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "rnn_type".')
        self.rnn = self.cell(input_size=self.embed_size,
                             hidden_size=self.embed_size,
                             num_layers=self.num_layers,
                             dropout=self.dropout_prob if self.num_layers > 1 else 0)
        self.wy = torch.nn.Linear(in_features=self.embed_size,
                                  out_features=config.Corpus.num_vocab)
        self.init_weights()

    def init_weights(self):
        self.wx.weight.data.uniform_(-self.embed_init_range, self.embed_init_range)
        self.wy.bias.data.fill_(0.0)
        self.wy.weight.data.uniform_(-self.embed_init_range, self.embed_init_range)

    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            res = (torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.embed_size).zero_()),
                   torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.embed_size).zero_()))
        else:
            res = torch.autograd.Variable(weight.new(self.num_layers,
                                                     self.batch_size,
                                                     self.embed_size).zero_())
        return res

    def forward(self, inputs, hidden):  # expects [num_steps, mb_size] tensor
        embeds = self.wx(inputs)
        outputs, hidden = self.rnn(embeds, hidden)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[-1])
        logits = self.wy(final_outputs)
        return logits

