"""
A transformer based language model.

Based on https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py

"""
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from typing import List, Dict
import torch
import numpy as np
from datasets import Dataset
import pandas as pd
from pathlib import Path

from traindsms.params import TransformerParams

PAD = '<pad>'


class Transformer:
    def __init__(self,
                 params: TransformerParams,
                 token2id: Dict[str, int],
                 seq_num: List[List[int]],
                 df_blank: pd.DataFrame,
                 instruments: List[str],
                 save_path: Path,
                 eos: str,
                 ):
        self.params = params
        self.token2id = token2id
        self.seq_num = seq_num
        self.df_blank = df_blank
        self.instruments = instruments
        self.save_path = save_path
        self.eos = eos

        self.token2id[PAD] = len(self.token2id)
        self.vocab_size = len(self.token2id)

        self.id2token = {i: token for token, i in self.token2id.items()}

        # no gpt2 tokenizer needed because vocab is defined by corpus

        if self.params.transformer_type == 'gpt2':
            config = GPT2Config(vocab_size=self.vocab_size,
                                n_positions=params.seq_len,  # max sequence length
                                n_ctx=params.seq_len,
                                n_embd=params.embed_size,
                                n_layer=params.num_layers,
                                n_head=params.num_heads,
                                n_inner=params.inner_size,  # dimensionality of the inner feed-forward layers
                                activation_function="gelu_new",
                                resid_pdrop=params.resid_pdrop,  # dropout probability for fully connected layers
                                embd_pdrop=0.1,
                                attn_pdrop=0.1,
                                layer_norm_epsilon=1e-5,
                                initializer_range=params.initializer_range,
                                scale_attn_weights=True,
                                use_cache=True,
                                bos_token_id=None,
                                eos_token_id=self.token2id[eos],
                                scale_attn_by_inverse_layer_idx=False,
                                reorder_and_upcast_attn=False,
                                )
            self.model = GPT2LMHeadModel(config)
        else:
            raise AttributeError(f'Did not recognize transformer_type "{params.transformer_type}"')

        self.t2e = None

        training_args = TrainingArguments(output_dir=str(self.save_path),
                                          per_device_train_batch_size=self.params.batch_size,
                                          per_device_eval_batch_size=self.params.batch_size,
                                          learning_rate=self.params.learning_rate,
                                          weight_decay=self.params.weight_decay,
                                          adam_beta2=self.params.adam_beta2,
                                          adam_epsilon=self.params.adam_epsilon,
                                          max_grad_norm=1.0,
                                          num_train_epochs=self.params.num_epochs,
                                          save_strategy='no',  # do not save checkpoints
                                          evaluation_strategy='epoch',  # compute loss on eval dataset every epoch
                                          do_train=True,
                                          disable_tqdm=True,
                                          )

        # padding and attention mask
        input_ids_all = []
        labels_all = []
        attention_mask_all = []
        for seq_num_i in self.seq_num:
            if self.params.seq_len < len(seq_num_i):
                raise ValueError('"seq_len" must be larger than largest number of tokens in input.')
            padded = np.full(self.params.seq_len, self.token2id[PAD], dtype=np.int32)
            padded[:len(seq_num_i)] = seq_num_i
            input_ids = padded
            attention_mask = np.array(input_ids != self.token2id[PAD], dtype=np.int32)
            # collect
            input_ids_all.append(torch.LongTensor(input_ids).cuda())
            labels_all.append(torch.LongTensor(input_ids).cuda())
            attention_mask_all.append(torch.LongTensor(attention_mask).cuda())

        # make dataset for Trainer
        data_in_dict = {'input_ids': input_ids_all,
                        'labels': labels_all,
                        'attention_mask': attention_mask_all,
                        }
        train_dataset = Dataset.from_dict(data_in_dict)

        self.trainer = Trainer(self.model,
                               args=training_args,
                               train_dataset=train_dataset,
                               eval_dataset=train_dataset,
                               tokenizer=None,
                               data_collator=None,
                               )

    def train(self):

        self.trainer.train()

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

        if 'with' not in self.token2id:
            seq_tok_eval = [s[:-1] for s in seq_tok_eval]

        # evaluate predictions
        for tokens in seq_tok_eval:
            token_ids = [self.token2id[t] for t in tokens]
            outputs = self.model(input_ids=torch.LongTensor(token_ids).cuda())
            logits = outputs['logits'].detach().cpu().numpy()  # [seq_len, vocab_size]
            print([f'{self.id2token[i]:>12}' for i in token_ids])
            print([f'{" ":>12}'] + [f'{self.id2token[i]:>12}' for i in np.argmax(logits, axis=1)])
            print([f'{" ":>12}'] + [f'{logits[n, i]}'[:12] for n, i in enumerate(np.argmax(logits, axis=1))])
            print()

        self.t2e = {token: row for token, row in zip(self.token2id,
                                                     self.model.get_input_embeddings().weight.detach().cpu())}

    def get_performance(self) -> Dict[str, List[float]]:
        """
        get eval_loss from log_history saved in trainer.state after training

        Note: eval_loss is on the training data. there is no distinct eval dataset
        """

        res = {'epoch': [],
               'eval_loss': []}

        log_history: List[Dict[str, float]] = self.trainer.state.log_history
        for log_i in log_history:
            try:
                res['eval_loss'].append(log_i['eval_loss'])
            except KeyError:  # some logs only contain "loss" instead of "train_loss"
                continue
            else:
                res['epoch'].append(log_i['epoch'])

        return res

    def calc_native_sr_scores(self,
                              verb: str,
                              theme: str,
                              instruments: List[str],
                              ) -> List[float]:
        """
        use language modeling based prediction task to calculate "native" sr scores
        """

        # todo does the model need an agent in the input?

        # prepare input
        token_ids = [self.token2id['John'], self.token2id[verb], self.token2id[theme]]
        if 'with' in self.token2id:
            token_ids.append(self.token2id['with'])

        # get logits
        with torch.no_grad():
            outputs = self.model(input_ids=torch.LongTensor(token_ids).cuda())
        logits = outputs['logits']  # (seq_len, vocab_size)
        logits_at_with = logits[-1].cpu().numpy()

        scores = []
        for instrument in instruments:

            token_id = self.token2id[instrument]
            sr = logits_at_with[token_id].item()
            scores.append(sr)

        return scores

    def fill_in_blank_df_and_save(self, epoch: int):
        """
        fill in blank data frame with semantic-relatedness scores
        """

        # TODO add this to train loop (get rid of huggingface Trainer?)
        # todo https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L1017

        df_results = self.df_blank.copy()

        for verb_phrase, row in self.df_blank.iterrows():
            verb, theme = verb_phrase.split()
            scores = self.calc_native_sr_scores(verb, theme, self.instruments)
            df_results.loc[verb_phrase] = [row['verb-type'], row['theme-type'], row['phrase-type']] + scores

        df_results.to_csv(self.save_path / f'df_sr_{epoch:06}.csv')
