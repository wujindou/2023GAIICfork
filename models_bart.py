# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartForConditionalGeneration

class BartModel(nn.Module):
    def __init__(self, n_token, max_l=80, sos_id=0, pad_id=1, eos_id=2):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_l = max_l

        config = BartConfig(
            vocab_size=3000,
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
        self.shared = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.generate = BartForConditionalGeneration(config)

        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(config.d_model, n_token)


    def forward(self, inputs, outputs=None, val=False):
        attn_mask = torch.full((inputs.shape[0], inputs.shape[1]), 1.0).to(inputs.device)
        attn_mask[inputs.eq(1)] = 0.0
        if outputs is None:
            return self._infer2(inputs)
            # return self.generate(inputs, attention_mask=attn_mask, )
        feature = self.encoder(input_ids=inputs, attention_mask=attn_mask, output_hidden_states=True)
        out = self.decoder(input_ids=outputs, encoder_hidden_states=feature[0], encoder_attention_mask=attn_mask)
        out = out.last_hidden_state
        if not val:
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

            out = self.forward(source, outputs, val=True)  # [B, L, n_token]
            prob = nn.functional.softmax(out, dim=2)[:, -1]  # [B, n_token]
            val, idx = torch.topk(prob, 1)  # (B,1)

            outputs = torch.cat([outputs, idx[:, 0].view(-1, 1)], dim=-1)  # (B,L+1)
            not_over = torch.minimum(not_over, torch.ne(outputs[:, -1], self.eos_id).long())  # [B]
            if torch.sum(not_over) == 0:
                break
        return outputs  # (B,L)

    def _infer2(self, source, top_k=1, mode='beam_search'):
        """
        source: [B,S,E],
        """
        batch_size = source.shape[0]
        outputs = torch.ones((batch_size, self.max_l), dtype=torch.long).to(source.device) * self.pad_id  # (B, L) PAD
        outputs[:, 0] = self.sos_id  # (B,) SOS
        not_over = torch.ones(batch_size).to(source.device)  # [B]

        with torch.no_grad():
            encoder_output = self.encoder(input_ids=source)
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(source.device) * self.sos_id

            if mode == 'greedy':
                for token_i in range(1, self.max_l):
                    decoder_output = self.decoder(input_ids=decoder_input_ids,
                                                  encoder_hidden_states=encoder_output.last_hidden_state)
                    logits = self.output(decoder_output.last_hidden_state[:, -1])  # (B, V)
                    next_tokens = logits.argmax(dim=-1)  # (B,)
                    outputs[:, token_i] = next_tokens
                    decoder_input_ids = next_tokens.unsqueeze(1)
                    not_over = not_over * torch.ne(next_tokens, self.eos_id).long()  # (B,)
                    if not_over.sum() == 0:
                        break
            elif mode == 'beam_search':
                beam_width = top_k
                logits = self.generate.generate(source, max_length=self.max_l, num_beams=beam_width,
                                                early_stopping=True).sequences
                outputs[:, :logits.shape[1]] = logits

        return outputs  # (B, L)
