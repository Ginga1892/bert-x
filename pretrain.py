import torch
import torch.nn as nn
import json

from processor import PreTrainData


class NextSentencePrediction(nn.Module):
    def __init__(self, hidden_size):
        super(NextSentencePrediction, self).__init__()
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh())
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, mini_batch_ref):
        sequence_output = mini_batch_ref['sequence_output']

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

    def forward(self, mini_batch_ref):
        sequence_output = mini_batch_ref['sequence_output']

        first_token_output = self.pooler(sequence_output[:, 0, :])
        sop_output = self.fc(first_token_output)

        return sop_output


class MaskedLanguageModel(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(MaskedLanguageModel, self).__init__()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, mini_batch_ref):
        sequence_output = mini_batch_ref['sequence_output']
        mlm_positions = mini_batch_ref['mlm_positions']

        # select the to-predict tokens from sequence output
        batch_size, max_seq_length, hidden_size = sequence_output.shape
        sequence_output = sequence_output.view(-1, hidden_size)
        mlm_positions = (mlm_positions + (torch.arange(0, batch_size) * max_seq_length).view(-1, 1)).view(-1)
        mlm_tokens_output = sequence_output.index_select(0, mlm_positions)

        # [batch_size * max_predictions_per_seq, hidden_size]
        mlm_output = self.fc(mlm_tokens_output)

        return mlm_output


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


class BertPreTrain(nn.Module):
    def __init__(self, bert_model):
        super(BertPreTrain, self).__init__()

        self.bert = bert_model
        self.mlm = MaskedLanguageModel(bert_model.config.vocab_size, bert_model.config.hidden_size)
        self.nsp = NextSentencePrediction(bert_model.config.hidden_size)

    def forward(self, mini_batch_ref):
        x = mini_batch_ref['input_ids']
        mask = mini_batch_ref['input_mask']
        segment_ids = mini_batch_ref['segment_ids']
        mini_batch_ref['sequence_output'] = self.bert(x, mask, segment_ids)

        prediction_scores = self.mlm(mini_batch_ref)
        next_sentence_scores = self.nsp(mini_batch_ref)

        return prediction_scores, next_sentence_scores


class PreTrainModel(nn.Module):
    def __init__(self, ptm_config_file, bert_model):
        super(PreTrainModel, self).__init__()
        self.transformer = bert_model
        self.pretrain_models = {
            'nsp': none(),
            'sop': none(),
            'mlm': none(),
            'sbo': none(),
            'tif': none()
        }
        self.process_stream = []

        self.get_process_stream(ptm_config_file)

    def get_process_stream(self, ptm_config_file):
        with open(ptm_config_file, 'r') as reader:
            config_dict = json.load(reader)
            for key, value in config_dict.items():
                self.__dict__[key] = value

        # For sentence prediction models
        if self.nsp:
            self.process_stream.append(create_nsp)
            self.pretrain_models['nsp'] = \
                NextSentencePrediction(self.transformer.config.hidden_size)
        elif self.sop:
            self.process_stream.append(create_sop)
            self.pretrain_models['sop'] = \
                SentenceOrderPrediction(self.transformer.config.hidden_size)
        else:
            self.process_stream.append(full_document_sampling)
        # For masked language models
        if not self.sbo:
            self.process_stream.append(create_mlm)
            self.pretrain_models['mlm'] =\
                MaskedLanguageModel(self.transformer.config.vocab_size, self.transformer.config.hidden_size)
        else:
            self.process_stream.append(create_sbo)
            self.pretrain_models['sbo'] =\
                SpanBoundaryObjective(self.transformer.config.vocab_size, self.transformer.config.hidden_size)
        # self.process_stream['tif'] = create_tif

    def create_instances(self, input_file, tokenizer):
        pt_data = PreTrainData(input_file, tokenizer)
        for process in self.process_stream:
            if process:
                process(pt_data, self.transformer.config)

        return pt_data.get_iterator(self.transformer.config)

    def forward(self, mini_batch_ref):
        # Transformer model
        x = mini_batch_ref['input_ids']
        mask = mini_batch_ref['input_mask']
        segment_ids = mini_batch_ref['segment_ids']
        mini_batch_ref['sequence_output'] = self.transformer(x, mask, segment_ids)

        ptm_outputs = {
            'sp': None,
            'mlm': None,
            'sbo': None,
            'tif': None
        }
        for ptm, ptm_func in self.pretrain_models.items():
            if ptm_func:
                ptm_outputs[ptm] = ptm_func(mini_batch_ref)

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


def none():
    pass


from albert import AlbertConfig, Albert
import tokenization

bert_config = AlbertConfig.from_json('bert_config.json')
transformer = Albert(bert_config)
bertx = PreTrainModel('ptm_config.json', transformer)
train_iterator = bertx.create_instances('00.txt', tokenization.ChineseWordpieceTokenizer('vocab.txt'))
for batch in train_iterator:
    bertx(batch)
