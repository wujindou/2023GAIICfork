# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch
import torch.nn as nn
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder

class BartModel(nn.Module):
    def __init__(self, n_token, max_l=80, sos_id=0, pad_id=1, eos_id=2, d=512):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_l = max_l
        self.d = d

        config = BartConfig(vocab_size=5000, max_position_embeddings=150, )
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)

        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(d, n_token)


    def forward(self, inputs, outputs=None):
        if outputs is None:
            return self._infer(inputs)
        attn_mask = torch.full((inputs.shape[0], inputs.shape[1]), 1.0).to(inputs.device)
        attn_mask[inputs.eq(1)] = 0.0
        feature = self.encoder(input_ids=inputs, attention_mask=attn_mask, output_hidden_states=True, output_attentions=True)

        out = self.decoder(encoder_hidden_states=feature[0], encoder_attention_mask=feature[2])
        out = out.last_hidden_state
        out = self.dropout(out)
        out = self.output(out) #[B, L, n_token]
        return out

    def _infer(self, source, top_k=1, mode='greedy'):
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
            not_over = torch.minimum(not_over, torch.ne(outputs[:, -1], self.eos_id).long())  # [B]
            if torch.sum(not_over) == 0:
                break
        return outputs  # (B,L)
