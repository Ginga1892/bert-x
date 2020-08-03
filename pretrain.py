import torch
import torch.nn as nn
import json

from albert import AlbertConfig, Albert
from processor import PreTrainData
import tokenization


class PreTrainModel(nn.Module):
    def __init__(self, bert_model, json_file):
        super(PreTrainModel, self).__init__()
        self.bert_model = bert_model
        self.pretrain_models = {
            'nsp': None,
            'sop': None,
            'mlm': None,
            'sbo': None,
            'tif': None
        }
        self.process_stream = {
            'sp': None,
            'mlm': None,
            'sbo': None,
            'tif': None
        }

        with open(json_file, "r") as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            self.__dict__[key] = value

    def get_process_stream(self):
        if self.nsp:
            self.process_stream['sp'] = create_nsp
            self.pretrain_models['sp'] = \
                NextSentencePrediction(self.bert_model.config.hidden_size)
        elif self.sop:
            self.process_stream['sp'] = create_sop
            self.pretrain_models['sp'] = \
                SentenceOrderPrediction(self.bert_model.config.hidden_size)
        else:
            self.process_stream['sp'] = full_document_sampling
        if self.sbo:
            self.process_stream['mlm'] = create_mlm
            self.pretrain_models['mlm'] =\
                MaskedLanguageModel(self.bert_model.config.vocab_size, self.bert_model.config.hidden_size)
        else:
            self.process_stream['mlm'] = create_sbo
            self.pretrain_models['mlm'] =\
                SpanBoundaryObjective(self.bert_model.config.vocab_size, self.bert_model.config.hidden_size)
        # self.process_stream['tif'] = create_tif

    def create_instances(self, input_file, tokenizer):
        self.get_process_stream()

        pt_data = PreTrainData(input_file, tokenizer)
        for process in self.process_stream.values():
            if process:
                process(pt_data, self.bert_model.config)

        return pt_data.get_iterator(self.bert_model.config)

    def forward(self, batch):
        x = self.bert_model(batch)

        ptm_outputs = {
            'sp': None,
            'mlm': None,
            'sbo': None,
            'tif': None
        }
        for ptm in self.pretrain_models:
            if self.pretrain_models[ptm]:
                ptm_outputs[ptm] = self.pretrain_models[ptm](batch)

        return ptm_outputs


def create_nsp(ptd, config):
    return ptd.create_nsp(config)


def create_sop(ptd, config):
    return ptd.create_sop(config)


def full_document_sampling(ptd, config):
    return ptd.full_document_sampling(config)


def create_mlm(ptd, config):
    return ptd.create_mlm(config)


def create_sbo(ptd, config):
    return ptd.create_sbo(config)


class BertPreTrain(nn.Module):
    def __init__(self, bert_model):
        super(BertPreTrain, self).__init__()

        self.bert = bert_model
        self.mlm = MaskedLanguageModel(bert_model.config.vocab_size, bert_model.config.hidden_size)
        self.nsp = NextSentencePrediction(bert_model.config.hidden_size)

    def forward(self, x, mask, segment_ids, mlm_positions):
        x = self.bert(x, mask, segment_ids)

        prediction_scores = self.mlm(x, mlm_positions)
        next_sentence_scores = self.nsp(x)

        return prediction_scores, next_sentence_scores


class MaskedLanguageModel(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(MaskedLanguageModel, self).__init__()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, sequence_output, mlm_positions):
        # select the to-predict tokens from sequence output
        batch_size, max_seq_length, hidden_size = sequence_output.shape
        sequence_output = sequence_output.view(-1, hidden_size)
        mlm_positions = (mlm_positions + (torch.arange(0, batch_size) * max_seq_length).view(-1, 1)).view(-1)
        mlm_tokens_output = sequence_output.index_select(0, mlm_positions)

        # [batch_size * max_predictions_per_seq, hidden_size]
        mlm_output = self.fc(mlm_tokens_output)

        return mlm_output


class NextSentencePrediction(nn.Module):
    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh())
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output):
        first_token_output = self.pooler(sequence_output[:, 0, :])
        nsp_output = self.fc(first_token_output)

        return nsp_output


class SentenceOrderPrediction(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceOrderPrediction, self).__init__()
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh())
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output):
        first_token_output = self.pooler(sequence_output[:, 0, :])
        sop_output = self.fc(first_token_output)

        return sop_output


class SpanBoundaryObjective(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(SpanBoundaryObjective, self).__init__()

        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward(self, sequence_output, sbo_positions):
        batch_size, max_seq_length, hidden_size = sequence_output.shape
        sequence_output = sequence_output.view(-1, 1)
        sbo_positions = (sbo_positions + (torch.arange(0, batch_size) * max_seq_length).view(-1, 1)).view(-1)
        left_boundary = sequence_output.index_select(0, sbo_positions)[::2]
        right_boundary = sequence_output.index_select(0, sbo_positions)[1::2]

        return


bert_config = AlbertConfig.from_json('bert_config.json')
model = Albert(bert_config)
ptm = PreTrainModel(model, 'ptm_config.json')
iterator = ptm.create_instances('00.txt', tokenization.ChineseWordpieceTokenizer('vocab.txt'))
for batch in iterator:
    ptm(batch)
