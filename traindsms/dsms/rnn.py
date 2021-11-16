import time
import torch
import pyprind
import numpy as np
import sys
from typing import List, Tuple


from traindsms.params import RNNParams


class RNN:
    def __init__(self,
                 params: RNNParams,
                 vocab: Tuple[str],
                 seq_num: List[List[int]],
                 ):
        self.params = params
        self.vocab = vocab
        self.seq_num = seq_num
        self.vocab_size = len(vocab)

        self.model = TorchRNN(self.params.rnn_type,
                              self.params.num_layers,
                              self.params.embed_size,
                              self.params.batch_size,
                              self.params.embed_init_range,
                              self.vocab_size)
        self.model.cuda()  # call this before constructing optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.params.learning_rate[0])

        self.t2e = None

    def gen_windows(self, token_ids):
        """
        yield collections of input and output sequences (x, and y).

        x and y are matrices of shape (num sequences, seq_len)
        """
        remainder = len(token_ids) % self.params.seq_len
        for i in range(self.params.seq_len):
            seq = np.roll(token_ids, i)  # rightward
            # make divisible so that we can reshape
            seq = seq[:-remainder] if remainder != 0 else seq
            x = np.reshape(seq, (-1, self.params.seq_len))
            # get next-token for each token in x
            y = np.roll(x, -1)

            # the last x, y pair must be excluded because rolling results in last y = first x which is not correct
            x = x[:-1]
            y = y[:-1]

            yield i, x, y

    def gen_batches(self, token_ids, batch_size):
        batch_id = 0
        for step_id, x, y in self.gen_windows(token_ids):  # more memory efficient not to create all windows in data

            # exclude some rows to split x and y evenly by batch size
            shape0 = len(x)
            num_excluded = shape0 % batch_size
            if num_excluded > 0:  # in case mb_size = 1
                x = x[:-num_excluded]
                y = y[:-num_excluded]
            shape0_adj = shape0 - num_excluded

            # yield excluded x,y pairs (when they do not evenly fit into a batch)
            if num_excluded > 0:
                yield batch_id, x[-num_excluded:, :], y[-num_excluded:, -1]  # todo test
                batch_id += 1

            # split into batches
            num_batches = shape0_adj // batch_size
            for x_b, y_b in zip(np.vsplit(x, num_batches),
                                np.vsplit(y, num_batches)):
                yield batch_id, x_b, y_b[:, -1]
                batch_id += 1

    def calc_pp(self,
                seq_num: List[List[int]],  # sequences of token IDs
                verbose: bool,
                batch_size_eval=1,
                ):
        if verbose:
            print('Calculating perplexity...')

        self.model.eval()

        loss_total = 0
        batch_id = 0
        token_ids = np.hstack(seq_num)
        for batch_id, x_b, y_b in self.gen_batches(token_ids, batch_size_eval):
            # feed-forward
            inputs = torch.LongTensor(x_b).cuda()
            targets = torch.LongTensor(y_b).cuda()
            logits = self.model(inputs)

            # compute loss
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(logits, targets)
            loss_total += loss.item()

        res = np.exp(loss_total / batch_id + 1)
        return res

    def train_epoch(self, seq_num, lr, verbose):
        start_time = time.time()
        self.model.train()

        # shuffle and flatten
        if self.params.shuffle_per_epoch:
            np.random.shuffle(seq_num)

        token_ids = np.hstack(seq_num)  # a single vector of all token IDs
        for batch_id, x_b, y_b in self.gen_batches(token_ids, self.params.batch_size):

            # forward step
            inputs = torch.LongTensor(x_b).cuda()  # batch_first=True
            targets = torch.LongTensor(y_b).cuda()
            logits = self.model(inputs)

            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(logits, targets)
            loss.backward()
            if self.params.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.grad_clip)
                for p in self.model.parameters():
                    p.data.add_(-lr, p.grad.data)  # TODO lr decay only happens with grad clipping
            else:
                self.optimizer.step()

    def train(self, verbose=True, calc_pp_train=True):

        # split data
        train_seq_num = []
        valid_seq_num = []
        test_seq_num = []
        for seq_num_i in self.seq_num:
            if np.random.binomial(1, self.params.train_percent):
                train_seq_num.append(seq_num_i)
            else:
                if np.random.binomial(1, 0.5):  # split valid and test docs evenly
                    valid_seq_num.append(seq_num_i)
                else:
                    test_seq_num.append(seq_num_i)
        print('Num docs in train {} valid {} test {}'.format(
            len(train_seq_num), len(valid_seq_num), len(test_seq_num)))

        if calc_pp_train:
            pp_train = self.calc_pp(train_seq_num, verbose)
            print(f'\nTrain perplexity before training: {pp_train:8.2f}')

        # train loop
        lr = self.params.learning_rate[0]  # initial
        decay = self.params.learning_rate[1]
        num_epochs_without_decay = self.params.learning_rate[2]
        pbar = pyprind.ProgBar(self.params.num_epochs, stream=sys.stdout)
        for epoch in range(self.params.num_epochs):
            if verbose:
                print()
                print(f'Starting epoch {epoch} with lr={lr:.6f}')
            lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
            lr = lr * lr_decay  # decay lr if it is time

            # train on one epoch
            self.train_epoch(train_seq_num, lr, verbose)

            if self.params.train_percent < 1.0:
                pp_val = self.calc_pp(valid_seq_num, verbose)
                if verbose:
                    print('\nValidation perplexity at epoch {}: {:8.2f}'.format(epoch, pp_val))

            if not verbose:
                pbar.update()

        if self.params.train_percent < 1.0:
            pp_val = self.calc_pp(valid_seq_num, verbose)
            if verbose:
                print('\nValidation perplexity after training: {:8.2f}'.format(pp_val))

        if calc_pp_train:
            pp_train = self.calc_pp(train_seq_num, verbose)
            print(f'\nTrain perplexity after training: {pp_train:8.2f}')

        wx = self.model.wx.weight.detach().cpu().numpy()
        self.t2e = {t: embedding for t, embedding in zip(self.vocab, wx)}


class TorchRNN(torch.nn.Module):
    def __init__(self,
                 rnn_type: str,
                 num_layers: int,
                 embed_size: int,
                 batch_size: int,
                 embed_init_range: float,
                 vocab_size: int,
                 ):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.embed_init_range = embed_init_range

        self.wx = torch.nn.Embedding(vocab_size, self.embed_size)
        if self.rnn_type == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.rnn_type == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "rnn_type".')
        self.rnn = self.cell(input_size=self.embed_size,
                             hidden_size=self.embed_size,
                             num_layers=self.num_layers,
                             batch_first=True,
                             dropout=self.dropout_prob if self.num_layers > 1 else 0)
        self.wy = torch.nn.Linear(in_features=self.embed_size,
                                  out_features=vocab_size)
        self.init_weights()

    def init_weights(self):
        self.wx.weight.data.uniform_(-self.embed_init_range, self.embed_init_range)
        self.wy.bias.data.fill_(0.0)
        self.wy.weight.data.uniform_(-self.embed_init_range, self.embed_init_range)

    def forward(self, inputs):
        embeds = self.wx(inputs)
        outputs, hidden = self.rnn(embeds)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[:, -1])
        logits = self.wy(final_outputs)

        # keep first dim
        if len(inputs) == 1:
            logits = torch.unsqueeze(logits, 0)

        return logits

