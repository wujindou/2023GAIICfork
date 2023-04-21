# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer


class CustomBartModel(nn.Module):
    def __init__(self, max_l=80, sos_id=0, pad_id=1, eos_id=2):
        super().__init__()
        self.max_l = max_l
        self.beam_size = 5

        self.tokenizer = BartTokenizer.from_pretrained('./custom_pretrain/')
        self.model = BartForConditionalGeneration.from_pretrained("./custom_pretrain/")

    def forward(self, inputs, attn_mask, outputs=None, infer=False):
        if outputs is None:
            # if infer:
            pred = self.model.generate(inputs=inputs, max_length=86, min_length=2, top_k=80, top_p=0.99, temperature=0.95, length_penalty=0.95, repetition_penalty=0.95,
            num_beams=self.beam_size, use_cache=True, early_stopping=True, no_repeat_ngram_size=3, bos_token_id=0, pad_token_id=1, eos_token_id=2, do_sample=False, decoder_start_token_id=0)
            pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
            return pred
            # else:
            #     return self.model.generate(inputs=inputs, max_length=80, min_length=2, top_k=80, top_p=0.99, temperature=0.95, length_penalty=0.95, repetition_penalty=0.95,
            #     num_beams=self.beam_size, use_cache=True, early_stopping=True, no_repeat_ngram_size=3, bos_token_id=0, pad_token_id=1, eos_token_id=2, do_sample=False, decoder_start_token_id=0)

        loss = self.model(input_ids=inputs, attention_mask=attn_mask, labels=outputs)
        return loss

class PretrainBartModel(nn.Module):
    def __init__(self, n_token, sos_id=0, pad_id=1, eos_id=2):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained("./custom_pretrain")
        self.model = BartForConditionalGeneration.from_pretrained("./custom_pretrain")

    def forward(self, inputs, outputs=None, infer=False):
        loss = self.model(input_ids=inputs, labels=outputs)
        return loss
