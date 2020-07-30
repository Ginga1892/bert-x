import torch
import torch.nn as nn
import numpy as np
import json


class AlbertConfig(object):
    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 inner_group_num=1,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.9,
                 attention_dropout_prob=0.9,
                 max_seq_length=512,
                 type_vocab_size=16,
                 initialize_range=0.02):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.inner_group_num = inner_group_num,
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_seq_length = max_seq_length
        self.type_vocab_size = type_vocab_size
        self.initialize_range = initialize_range

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            config_dict = json.load(f)
        config = AlbertConfig(vocab_size=None)
        for key, value in config_dict.items():
            config.__dict__[key] = value
        return config


class Albert(nn.Module):
    def __init__(self, config):
        super(Albert, self).__init__()
        self.config = config

        self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.embedding_size)
        self.seg_embedding = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.projector = nn.Linear(config.embedding_size, config.hidden_size)
        self.encoders = nn.ModuleList([
            EncoderGroup(config) for _ in range(config.num_hidden_layers // config.inner_group_num)])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, mask, segment_ids):
        batch_size, max_seq_length = x.shape
        mask = mask.unsqueeze(1).repeat(1, max_seq_length, 1).unsqueeze(1)

        te = self.tok_embedding(x)
        pos = torch.arange(0, max_seq_length).unsqueeze(0).repeat(batch_size, 1)
        pe = self.pos_embedding(pos)
        se = self.seg_embedding(segment_ids)
        x = te + pe + se
        x = self.dropout(x)
        x = self.projector(x)

        for group in self.encoders:
            x = group(x, mask)

        return x


class EncoderGroup(nn.Module):
    def __init__(self, config):
        super(EncoderGroup, self).__init__()
        encoder = Encoder(config.hidden_size,
                          config.num_attention_heads,
                          config.intermediate_size,
                          config.hidden_act,
                          config.hidden_dropout_prob,
                          config.attention_dropout_prob)
        self.encoder_group = nn.ModuleList([
            encoder for _ in range(config.inner_group_num)])

    def forward(self, x, mask):
        for encoder in self.encoder_group:
            x = encoder(x, mask)

        return x


class Encoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 intermediate_size,
                 hidden_act,
                 hidden_dropout_prob,
                 attention_dropout_prob):
        super(Encoder, self).__init__()

        self.self_attention = MultiHeadAttention(hidden_size, num_attention_heads, attention_dropout_prob)
        self.activation = nn.GELU() if hidden_act == 'gelu' else nn.Tanh()
        self.ffn = nn.Sequential(nn.Linear(hidden_size, intermediate_size),
                                 self.activation,
                                 nn.Linear(intermediate_size,hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        attention_output = self.layer_norm(x + self.dropout(attention_output))

        encoder_output = self.ffn(attention_output)
        encoder_output = self.layer_norm(attention_output + self.dropout(encoder_output))

        return encoder_output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob):
        super(MultiHeadAttention, self).__init__()

        self.h = num_attention_heads
        self.d_k = hidden_size // num_attention_heads

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout_prob)

    def forward(self, query, key, value, mask=None):
        # q, k, v = [batch_size, src_len, hidden_size]
        batch_size, hidden_size = query.shape[0], query.shape[2]

        # q, k, v = [batch_size, src_len, hidden_size]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # q, v = [batch_size, src_len, num_attention_heads, head_size]
        # k = [batch_size, src_len, head_size, num_attention_heads]
        q = q.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 3, 1)
        v = v.view(batch_size, -1, self.h, self.d_k).permute(0, 2, 1, 3)

        # Attention(Q, K, V) = Softmax(Q * K^T / d) * V
        attention_scores = torch.matmul(q, k) / np.sqrt(self.d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        y = torch.matmul(attention_probs, v)

        # y = [batch_size, src_len, hidden_size]
        y = y.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, hidden_size)

        return self.w_o(y)

