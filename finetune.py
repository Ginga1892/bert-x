import torch
import torch.nn as nn


class BertClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertClassification, self).__init__()

        self.bert = bert_model
        self.pooler = nn.Sequential(nn.Linear(bert_model.config.hidden_size, bert_model.config.hidden_size),
                                    nn.Tanh())
        self.fc = nn.Linear(bert_model.config.hidden_size, num_labels)

    def forward(self, x, mask, segment_ids):
        x = self.bert(x, mask, segment_ids)
        pooler_output = self.pooler(x[:, 0, :])

        return torch.log_softmax(self.fc(pooler_output), dim=-1)
