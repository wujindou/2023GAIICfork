# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch.nn as nn
import torch
from transformers import BartModel, BartConfig, BartForConditionalGeneration

class TranslationModel(nn.Module):
    def __init__(self, n_token, output_l, d=50265, sos_id=0, pad_id=1, eos_id=2):
        super().__init__()
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.output_l = output_l
        config = BartConfig.from_pretrained('facebook/bart-base')
        if n_token is not None:
            self.output = nn.Linear(d, n_token)
            self.dropout = nn.Dropout(p=0.2)
            # self.output.weight = self.token_embedding.weight

    def forward(self, inputs, outputs=None):

        attn_mask = torch.full((inputs.shape[0], inputs.shape[1]), 1.0).to(inputs.device)
        attn_mask[inputs.eq(0)] = 0.0

        if outputs is None:
            output = self.generate(inputs, max_length=self.output_l)
            return output

        decoder_attn_mask = torch.full((outputs.shape[0], outputs.shape[1]), 1.0).to(inputs.device)
        decoder_attn_mask[outputs.eq(0)] = 0.0
        feature = self.bart(input_ids=inputs, attention_mask=attn_mask, labels=outputs)
        out = feature.logits
        out = self.dropout(out)
        out = self.output(out) #[B, L, n_token]
        return out

    def generate(self, inputs, max_length):
        inputs = inputs.to(self.bart.device)
        attn_mask = torch.full((inputs.shape[0], inputs.shape[1]), 1.0).to(inputs.device)
        attn_mask[inputs.eq(0)] = 0.0
        # for idx in range(inputs.shape[0]):
        #     outputs = self.bart.generate(input_ids=inputs[idx].unsqueeze(0), attention_mask=attn_mask[idx].unsqueeze(0), max_length=max_length, eos_token_id=self.eos_id, pad_token_id=self.pad_id, bos_token_id=self.sos_id)
        outputs = self.bart.generate(input_ids=inputs, attention_mask=attn_mask, max_length=max_length, eos_token_id=self.eos_id, pad_token_id=self.pad_id, bos_token_id=self.sos_id, early_stopping=True, num_beams=1)
        print(outputs)
        return outputs
