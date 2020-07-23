import torch
import torch.nn as nn
from model import Bert


class BertPreTrain(nn.Module):
    def __init__(self, bert_config):
        super(BertPreTrain, self).__init__()

        self.bert = Bert(bert_config)
        self.mlm = MaskedLanguageModel(bert_config.vocab_size, bert_config.hidden_size)
        self.nsp = NextSentencePrediction(bert_config.hidden_size)

    def forward(self, x, segment_ids, mlm_positions):
        x = self.bert(x, segment_ids)

        prediction_scores = self.mlm(x, mlm_positions)
        next_sentence_scores = self.nsp(x)

        return prediction_scores, next_sentence_scores


class MaskedLanguageModel(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(MaskedLanguageModel, self).__init__()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, mlm_positions):
        batch_size = mlm_positions.shape[0]
        return torch.tensor(
            [torch.softmax(self.fc(x), dim=-1).detach().numpy()[i, mlm_positions[i, :], :] for i in range(batch_size)])


class NextSentencePrediction(nn.Module):
    def __init__(self, hid_size):
        super(NextSentencePrediction, self).__init__()

        self.fc = nn.Linear(hid_size, 2)

    def forward(self, x):
        return torch.softmax(self.fc(x[:, 0, :]), dim=-1)
