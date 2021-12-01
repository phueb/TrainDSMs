from transformers import GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from typing import List, Dict
import torch
import numpy as np
from datasets import Dataset

from traindsms.params import TransformerParams


class TransformerDSM:
    def __init__(self,
                 params: TransformerParams,
                 token2id: Dict[str, int],
                 seq_num: List[List[int]],
                 output_dir: str
                 ):
        self.params = params
        self.token2id = token2id
        self.seq_num = seq_num
        self.output_dir = output_dir

        self.vocab_size = len(self.token2id)

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
                                resid_pdrop=params.dropout_prob,  # dropout probability for fully connected layers
                                embd_pdrop=0.1,
                                attn_pdrop=0.1,
                                layer_norm_epsilon=1e-5,
                                initializer_range=0.02,
                                scale_attn_weights=True,
                                use_cache=True,
                                bos_token_id=None,
                                eos_token_id=None,
                                scale_attn_by_inverse_layer_idx=False,
                                reorder_and_upcast_attn=False,
                                )
            self.model = GPT2LMHeadModel(config)
        else:
            raise AttributeError(f'Did not recognize transformer_type "{params.transformer_type}"')

        self.t2e = None

        training_args = TrainingArguments(output_dir=self.output_dir,
                                          per_device_train_batch_size=self.params.batch_size,
                                          per_device_eval_batch_size=self.params.batch_size,
                                          learning_rate=self.params.learning_rate,
                                          weight_decay=0.0,
                                          max_grad_norm=1.0,
                                          num_train_epochs=self.params.num_epochs,
                                          save_strategy='no',  # do not save checkpoints
                                          evaluation_strategy='epoch',  # compute loss on eval dataset every epoch
                                          do_train=True,
                                          disable_tqdm=True,
                                          )

        # https://huggingface.co/transformers/_modules/transformers/models/gpt2/modeling_gpt2.html#GPT2Model.forward

        # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py

        # TODO how to do batching?

        # make dataset for Trainer
        data_in_dict = {'input_ids': [torch.LongTensor(seq_num_i).cuda() for seq_num_i in self.seq_num],
                        'labels': [torch.LongTensor(seq_num_i).cuda() for seq_num_i in self.seq_num],
                        # 'attention_mask': [[1] * len(seq_num_i) for seq_num_i in self.seq_num],
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

        # evaluate predictions
        id2token = {i: token for token, i in self.token2id.items()}
        for seq_num_i in self.seq_num[::100]:
            outputs = self.model(input_ids=torch.LongTensor(seq_num_i).cuda())
            logits = outputs['logits'].detach().cpu().numpy()  # [seq_len, vocab_size]
            print('Input:')
            print([id2token[i] for i in seq_num_i])
            print('Output:')
            print([id2token[i] for i in np.argmax(logits, axis=1)])

        self.t2e = {token: row for token, row in zip(self.token2id,
                                                     self.model.get_input_embeddings().weight.detach().cpu())}

    def get_performance(self) -> Dict[str, List[float]]:
        """get train_loss from log_history saved in trainer.state after training"""

        res = {'epoch': [],
               'eval_loss': []}

        log_history: List[Dict[str, float]] = self.trainer.state.log_history
        for log_i in log_history:
            try:
                res['eval_loss'].append(log_i['eval_loss'])
            except KeyError: # some logs only contain "loss" instead of "train_loss"
                continue
            else:
                res['epoch'].append(log_i['epoch'])

        return res
