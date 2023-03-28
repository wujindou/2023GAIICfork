# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch.nn as nn
import torch
from transformers import BartModel, BartConfig

class TranslationModel(nn.Module):
    def __init__(self, n_token, d=768, sos_id=0, pad_id=1):
        super().__init__()
        self.sos_id = sos_id
        self.pad_id = pad_id
        config = BartConfig.from_pretrained('facebook/bart-base')
        self.bart = BartModel(config)
        if n_token is not None:
            self.output = nn.Linear(d, n_token)
            self.dropout = nn.Dropout(p=0.2)
            # self.output.weight = self.token_embedding.weight

    def forward(self, inputs, outputs=None, beam=1):
        attn_mask = torch.full((inputs.shape[0], inputs.shape[1]), 1.0).to(inputs.device)
        decoder_attn_mask = torch.full((outputs.shape[0], outputs.shape[1]), 1.0).to(inputs.device)
        attn_mask[inputs.eq(0)] = 0.0
        feature = self.bart(input_ids=inputs, attention_mask=attn_mask, decoder_input_ids=outputs, decoder_attention_mask=decoder_attn_mask)

        if outputs is None:
            return self._infer(source=feature.encoder_last_hidden_state,  top_k=beam, eos_id=2)
        out = feature.last_hidden_state
        out = self.dropout(out)
        out = self.output(out) #[B, L, n_token]
        return out

    def _infer(self, source, top_k=1, eos_id=2, mode='greedy'):
        """
        source: [B,S,E],
        """
        outputs = torch.ones((source.shape[0], 1), dtype=torch.long).to(source.device) * self.sos_id  # (K,B,1) SOS
        not_over = torch.ones((source.shape[0])).to(source.device)  # [K,B]
        assert top_k == 1

        for token_i in range(1, self.max_l):

            out = self.forward(source, outputs)  # [B, L, n_token]
            prob = nn.functional.softmax(out, dim=2)[:, -1]  # [B, n_token]
            val, idx = torch.topk(prob, 1)  # (B,1)

            outputs = torch.cat([outputs, idx[:, 0].view(-1, 1)], dim=-1)  # (B,L+1)
            not_over = torch.minimum(not_over, torch.ne(outputs[:, -1], eos_id).long())  # [B]
            if torch.sum(not_over) == 0:
                break
        return outputs  # (B,L)
