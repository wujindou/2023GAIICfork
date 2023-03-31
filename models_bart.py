# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration


class BartModel(nn.Module):
    def __init__(self, n_token, max_l=80, sos_id=0, pad_id=1, eos_id=2):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_l = max_l
        self.beam_size = 5

        config = BartConfig(
            vocab_size=n_token,
            max_position_embeddings=150,
            encoder_layers=6,
            encoder_attention_heads=8,
            encoder_ffn_dim=2048,
            encoder_layerdrop=0.1,
            decoder_layers=6,
            decoder_attention_heads=8,
            decoder_ffn_dim=2048,
            decoder_layerdrop=0.1,
            d_model=512,
            dropout=0.2,
            activation_dropout=0.1,
            attention_dropout=0.1,
            init_std=0.02,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            is_encoder_decoder=True,
            decoder_start_token_id=2,
            forced_eos_token_id=2,
            use_cache=True,
            num_labels=3,
        )

        self.model = BartForConditionalGeneration(config)

    def forward(self, inputs, outputs=None, val=False):
        attn_mask = torch.full((inputs.shape[0], inputs.shape[1]), 1.0).to(inputs.device)
        attn_mask[inputs.eq(1)] = 0.0
        if outputs is None:
            return self.model.generate(inputs=inputs, max_length=80, min_length=2, use_cache=True, num_beams=5, early_stopping=True)
        out = self.model(input_ids=inputs, attention_mask=attn_mask, labels=outputs, use_cache=True, output_hidden_states=True)
        out = out.last_hidden_state
        if not val:
            out = self.dropout(out)
        out = self.output(out)  # [B, L, n_token]
        return out

    def _infer(self, source, top_k=1, mode='greedy'):
        """
        source: [B,S,E],
        """
        if mode == 'greedy':
            outputs = torch.ones((source.shape[0], 1), dtype=torch.long).to(source.device) * self.sos_id  # (K,B,1) SOS
            not_over = torch.ones((source.shape[0])).to(source.device)  # [K,B]
            assert top_k == 1

            for token_i in range(1, self.max_l):

                out = self.forward(source, outputs, val=True)  # [B, L, n_token]
                prob = nn.functional.softmax(out, dim=2)[:, -1]  # [B, n_token]
                val, idx = torch.topk(prob, 1)  # (B,1)

                outputs = torch.cat([outputs, idx[:, 0].view(-1, 1)], dim=-1)  # (B,L+1)
                not_over = torch.minimum(not_over, torch.ne(outputs[:, -1], self.eos_id).long())  # [B]
                if torch.sum(not_over) == 0:
                    break
            return outputs  # (B,L)
        elif mode == 'beam_search':
            beam_size = self.beam_size
            batch_size = source.shape[0]
            outputs = torch.ones((batch_size, 1), dtype=torch.long).to(source.device) * self.sos_id  # (B,1) SOS
            not_over = torch.ones((batch_size), dtype=torch.bool).to(source.device)  # [B]

            beams = [(outputs, 0)]

            for token_i in range(1, self.max_l):

                new_beams = []

                for beam_output, beam_score in beams:
                    if not not_over.any():
                        break

                    out = self.forward(source, beam_output, val=True)  # [B, L, n_token]
                    prob = nn.functional.softmax(out, dim=2)[:, -1]  # [B, n_token]
                    top_k_probs, top_k_indices = torch.topk(prob, beam_size)  # (B, beam_size)

                    for i in range(beam_size):
                        next_output = torch.cat([beam_output, top_k_indices[:, i].unsqueeze(1)], dim=-1)  # (B, L+1)
                        next_score = beam_score - torch.log(top_k_probs[:, i])
                        end_flags = torch.eq(next_output[:, -1], self.eos_id)

                        for j in range(batch_size):
                            if not not_over[j]:
                                continue

                            if end_flags[j]:
                                not_over[j] = False

                            new_beams.append((next_output[j], next_score[j]))

                new_beams = sorted(new_beams, key=lambda x: x[1])[:beam_size]
                beams = []

                for beam_output, beam_score in new_beams:
                    if not not_over.any():
                        break
                    beams.append((beam_output, beam_score))

            return beams[0][0]  # (B,L)
