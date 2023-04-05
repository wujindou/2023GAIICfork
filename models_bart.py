# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer


class BartModel2(nn.Module):
    def __init__(self, n_token, max_l=80, sos_id=0, pad_id=1, eos_id=2):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_l = max_l
        self.beam_size = 5

        self.tokenizer = BartTokenizer.from_pretrained('./custom_bart')
        self.model = BartForConditionalGeneration.from_pretrained("./custom_bart")

    def forward(self, inputs, attn_mask, outputs=None, infer=False):
        if outputs is None:
            if infer:
                pred = self.model.generate(inputs=inputs, max_length=80, min_length=2, top_k=80, top_p=0.95, repetition_penalty=0.99, num_beams=self.beam_size, \
                    use_cache=True, early_stopping=True, no_repeat_ngram_size=3, bos_token_id=0, pad_token_id=1, eos_token_id=2)
                return self.tokenizer.batch_decode(pred, skip_special_tokens=True)
            else:
                return self.model.generate(inputs=inputs, max_length=80, min_length=2, top_k=80, top_p=0.95, repetition_penalty=0.99, num_beams=self.beam_size, \
                        use_cache=True, early_stopping=True, no_repeat_ngram_size=3, bos_token_id=0, pad_token_id=1, eos_token_id=2)
        loss = self.model(input_ids=inputs, attention_mask=attn_mask, labels=outputs)
        return loss
