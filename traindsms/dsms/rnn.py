import torch
import pyprind
import numpy as np
import sys
from typing import List, Dict, Optional
from collections import defaultdict
import pandas as pd
from pathlib import Path

from traindsms.params import RNNParams


class RNN:
    def __init__(self,
                 params: RNNParams,
                 token2id: Dict[str, int],
                 seq_num: List[List[int]],  # sequences of token IDs, "numeric sequences"
                 df_blank: pd.DataFrame,
                 instruments: List[str],
                 save_path: Path,
                 ):
        self.params = params
        self.token2id = token2id
        self.seq_num = seq_num
        self.df_blank = df_blank
        self.instruments = instruments
        self.save_path = save_path

        self.vocab_size = len(token2id)
        self.id2token = {i: token for token, i in self.token2id.items()}

        self.model = TorchRNN(self.params.rnn_type,
                              self.params.num_layers,
                              self.params.embed_size,
                              self.params.embed_init_range,
                              self.params.dropout_prob,
                              self.vocab_size)
        self.model.cuda()  # call this before constructing optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adagrad(self.model.parameters(),
                                             lr=self.params.learning_rate,
                                             lr_decay=self.params.lr_decay,
                                             weight_decay=self.params.weight_decay)

        self.t2e = None
        self.performance = defaultdict(list)

    def gen_batches(self,
                    seq_num: List[List[int]],  # sequences of token IDs
                    batch_size: Optional[int] = None,
                    ):
        """
        generate sequences for predicting next-tokens.

        Note:
        each token in each sequence must be predicted during training.
        this function does not return moving windows.
        """

        if batch_size is None:
            batch_size = self.params.batch_size

        # shuffle and flatten
        np.random.shuffle(seq_num)

        # get seq lengths
        seq_lengths = set([len(s) for s in seq_num])

        # batch by sequence length to avoid padding
        for seq_len in seq_lengths:
            seq_sized = (s for s in seq_num if len(s) == seq_len)

            # collect sequences into batch
            seq_b = []
            while len(seq_b) < batch_size:
                try:
                    seq_b.append(next(seq_sized))
                except StopIteration:
                    break

            yield seq_b

    def calc_pp(self,
                seq_num: List[List[int]],  # sequences of token IDs
                verbose: bool,
                ):
        if verbose:
            print('Calculating perplexity...')

        self.model.eval()

        loss_total = 0
        num_loss = 0
        for seq_b in self.gen_batches(seq_num):

            # forward step
            input_ids = torch.LongTensor(seq_b).cuda()[:, :-1]  # batch_first=True
            logits = self.model(input_ids)  # logits at all time steps [batch_size * seq_len, vocab_size]

            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            labels = torch.LongTensor(seq_b).cuda()[:, 1:]
            labels = torch.flatten(labels)
            loss = self.criterion(logits,  # [batch_size * seq_len, vocab_size]
                                  labels)  # [batch_size * seq_len]
            loss_total += loss.item()
            num_loss += 1

        res = np.exp(loss_total / num_loss)
        return res

    def train_epoch(self,
                    seq_num: List[List[int]],  # sequences of token IDs
                    ) -> None:
        self.model.train()

        for seq_b in self.gen_batches(seq_num):  # generates batches of complete sequences

            # forward step
            input_ids = torch.LongTensor(seq_b).cuda()[:, :-1]  # batch_first=True
            logits = self.model(input_ids)  # logits at all time steps [batch_size * seq_len, vocab_size]

            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            labels = torch.LongTensor(seq_b).cuda()[:, 1:]
            labels = torch.flatten(labels)
            loss = self.criterion(logits,  # [batch_size * seq_len, vocab_size]
                                  labels)  # [batch_size * seq_len]
            loss.backward()

            # gradient clipping + update weights
            if self.params.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.params.grad_clip,
                                               norm_type=2)
            self.optimizer.step()

    def train(self,
              verbose: bool = True,
              calc_pp_train_during_training: bool = True,
              calc_pp_train_after_training: bool = False,
              score_exp2b: bool = True,
              ):

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
        print(f'Num sequences in train={len(train_seq_num):,}')
        print(f'Num sequences in valid={len(valid_seq_num):,}')
        print(f'Num sequences in test ={len(test_seq_num):,}')

        # get unique sequences in train data for evaluating train_pp
        train_seq_num_unique = []
        for s in train_seq_num:
            if s not in train_seq_num_unique:
                train_seq_num_unique.append(s)
        print(f'Num unique sequences in train ={len(train_seq_num_unique):,}')

        if calc_pp_train_during_training:
            pp_train = self.calc_pp(train_seq_num_unique, verbose)
            self.performance['epoch'].append(0)
            self.performance['pp_train'].append(pp_train)
            print(f'Train perplexity at epoch {0}: {pp_train:8.2f}')

        # train loop
        pbar = pyprind.ProgBar(self.params.num_epochs, stream=sys.stdout)
        for epoch in range(1, self.params.num_epochs + 1):
            self.performance['epoch'].append(epoch)

            if verbose:
                print(f'Epoch {epoch:>6}', flush=True)

            # train on one epoch
            self.train_epoch(train_seq_num)

            # evaluate on experiment 2b
            if score_exp2b:
                self.fill_in_blank_df_and_save(epoch)

            if self.params.train_percent < 1.0:
                pp_val = self.calc_pp(valid_seq_num, verbose)
                self.performance['pp_val'].append(pp_val)
                if verbose:
                    print(f'Validation perplexity at epoch {epoch}: {pp_val:8.2f}')

            if calc_pp_train_during_training:
                pp_train = self.calc_pp(train_seq_num_unique, verbose)
                self.performance['pp_train'].append(pp_train)
                print(f'Train perplexity at epoch {epoch}: {pp_train:8.2f}')

            if not verbose:
                pbar.update()

        if self.params.train_percent < 1.0:
            pp_val = self.calc_pp(valid_seq_num, verbose)
            if verbose:
                print(f'Validation perplexity after training: {pp_val:8.2f}')

        if calc_pp_train_after_training:
            pp_train = self.calc_pp(train_seq_num, verbose)
            self.performance['pp_train'].append(pp_train)
            self.performance['epoch'].append(self.performance['epoch'][-1] + 1)
            print(f'Train perplexity after training: {pp_train:8.2f}')

        # evaluate predictions
        seq_tok_eval = [
            'John preserve pepper with'.split(),
            'John preserve orange with'.split(),
            'John repair blender with'.split(),
            'John repair bowl with'.split(),
            'John pour tomato-juice with'.split(),
            'John decorate cookie with'.split(),
            'John carve turkey with'.split(),
            'John heat tilapia with'.split(),

        ]
        with torch.no_grad():
            x_b = [[self.token2id[t] for t in tokens] for tokens in seq_tok_eval]
            logits = self.model.predict_next_token(input_ids=torch.LongTensor(x_b).cuda())
            logits_batch = logits.cpu().numpy()  # logits at last time step, [batch_size=8, vocab_size]

        for tokens, logits in zip(seq_tok_eval, logits_batch):
            predicted_token_id = np.argmax(logits, axis=0)
            print([f'{t:>12}' for t in tokens])
            print([f'{" ":>12}'] * (len(tokens)) + [f'{self.id2token[predicted_token_id]:>12}'])
            print()

        # save token embeddings
        wx = self.model.wx.weight.detach().cpu().numpy()
        self.t2e = {t: embedding for t, embedding in zip(self.token2id, wx)}

    def get_performance(self) -> Dict[str, List[float]]:
        return self.performance

    def calc_native_sr_scores(self,
                              verb: str,
                              theme: str,
                              instruments: List[str],
                              verbose: bool = True,
                              ) -> List[float]:
        """
        use language modeling based prediction task to calculate "native" sr scores
        """

        # TODO does Agent need to be in input to perform well on exp2b?

        # prepare input
        token_ids = [self.token2id['John'], self.token2id[verb], self.token2id[theme]]
        if 'with' in self.token2id:
            token_ids.append(self.token2id['with'])

        # get logits (at last time step)
        with torch.no_grad():
            x_b = [token_ids]
            logits_at_last_step = self.model.predict_next_token(torch.LongTensor(x_b).cuda())  # [1, vocab_size]
            logits_at_last_step = logits_at_last_step.squeeze()  # [vocab_size]

        # these are printed to console
        exp_vps = {'preserve pepper',
                   # 'preserve orange',
                   # 'repair blender',
                   # 'repair bowl',
                   # 'pour tomato-juice',
                   # 'decorate cookie',
                   # 'carve turkey',
                   # 'heat tilapia',
                   }

        # get scores
        scores = []
        for instrument in instruments:
            token_id = self.token2id[instrument]
            sr = logits_at_last_step[token_id].item()
            scores.append(sr)

            if verbose and verb + ' ' + theme in exp_vps:
                print(f'{verb} {theme} {instrument:>12} : {sr: .4f}')

        if verbose and verb + ' ' + theme in exp_vps:
            print()

        return scores

    def fill_in_blank_df_and_save(self, epoch: int):
        """
        fill in blank data frame with semantic-relatedness scores
        """
        self.model.eval()

        df_results = self.df_blank.copy()

        for verb_phrase, row in self.df_blank.iterrows():
            verb_phrase: str
            verb, theme = verb_phrase.split()
            scores = self.calc_native_sr_scores(verb, theme, self.instruments)
            df_results.loc[verb_phrase] = [row['verb-type'], row['theme-type'], row['phrase-type']] + scores

        df_results.to_csv(self.save_path / f'df_sr_{epoch:06}.csv')


class TorchRNN(torch.nn.Module):
    def __init__(self,
                 rnn_type: str,
                 num_layers: int,
                 embed_size: int,
                 embed_init_range: float,
                 dropout_prob: float,
                 vocab_size: int,
                 ):
        super().__init__()
        self.rnn_type = rnn_type
        self.embed_size = embed_size
        self.vocab_size = vocab_size

        self.wx = torch.nn.Embedding(vocab_size, self.embed_size)
        if self.rnn_type == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.rnn_type == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "rnn_type".')
        self.rnn = self.cell(input_size=embed_size,
                             hidden_size=embed_size,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout_prob)
        self.wy = torch.nn.Linear(in_features=embed_size,
                                  out_features=vocab_size)

        # init weights
        self.wx.weight.data.uniform_(-embed_init_range, embed_init_range)
        max_w = np.sqrt(1 / self.embed_size)
        self.wy.weight.data.uniform_(-max_w, max_w)
        self.wy.bias.data.fill_(0.0)

    def predict_next_token(self, input_ids):
        embeds = self.wx(input_ids)
        outputs, hidden = self.rnn(embeds)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[:, -1])
        logits = self.wy(final_outputs)

        # keep first dim
        if len(input_ids) == 1:
            logits = torch.unsqueeze(logits, 0)

        return logits

    def forward(self, input_ids):
        embeds = self.wx(input_ids)
        outputs, hidden = self.rnn(embeds)  # this returns all time steps
        logits = self.wy(outputs.reshape(-1, self.embed_size))

        # keep first dim
        if len(input_ids) == 1:
            logits = torch.unsqueeze(logits, 0)

        return logits

