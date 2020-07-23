import torch
import torch.nn as nn
import numpy as np
import json


class BertConfig(object):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 dropout_prob=0.9,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.dropout = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            d = json.load(f)
        config = BertConfig(vocab_size=None)
        for key, value in d.items():
            config.__dict__[key] = value
        return config


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()

        self.tok_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.seg_embedding = nn.Embedding(2, config.hidden_size)
        self.encoders = nn.ModuleList([
            Encoder(config.hidden_size, config.num_attention_heads, config.dropout_prob) for _ in range(config.num_hidden_layers)
        ])
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x, segment_ids):
        batch_size, max_seq_length = x.shape
        mask = (x > 0).unsqueeze(1).repeat(1, max_seq_length, 1).unsqueeze(1)

        te = self.tok_embedding(x)
        pos = torch.arange(0, max_seq_length).unsqueeze(0).repeat(batch_size, 1)
        pe = self.pos_embedding(pos)
        se = self.seg_embedding(segment_ids)
        x = te + pe + se
        x = self.dropout(x)

        for encoder in self.encoders:
            x = encoder(x, mask)

        return x


class Encoder(nn.Module):
    def __init__(self, hid_size, n_heads, dropout=0.9):
        super(Encoder, self).__init__()

        self.self_attention = MultiHeadAttention(hid_size, n_heads)
        self.ffn = nn.Sequential(nn.Linear(hid_size, hid_size * 4), nn.GELU(), nn.Linear(hid_size * 4, hid_size))
        self.layer_norm = nn.LayerNorm(hid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        attention_output = self.layer_norm(x + self.dropout(attention_output))

        encoder_output = self.ffn(attention_output)
        encoder_output = self.layer_norm(attention_output + self.dropout(encoder_output))

        return encoder_output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer"""

    def __init__(self, hid_size, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.h = n_heads
        self.d_k = hid_size // n_heads

        self.w_q = nn.Linear(hid_size, hid_size)
        self.w_k = nn.Linear(hid_size, hid_size)
        self.w_v = nn.Linear(hid_size, hid_size)
        self.w_o = nn.Linear(hid_size, hid_size)

    def forward(self, query, key, value, mask=None):
        # q, k, v = [batch_size, src_len, hid_size]
        batch_size, hid_size = query.shape[0], query.shape[2]

        # q, k, v = [batch_size, src_len, hid_size]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # q, v = [batch_size, src_len, n_heads, head_size]
        # k = [batch_size, src_len, head_size, n_heads]
        q = q.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 3, 1)
        v = v.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)

        # Attention(Q, K, V) = Softmax(Q * K^T / d) * V
        attention_scores = torch.matmul(q, k) / np.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)

        attention = torch.softmax(attention_scores, dim=-1)
        y = torch.matmul(attention, v)

        # y = [batch_size, src_len, hid_size]
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, hid_size)

        return self.w_o(y)

