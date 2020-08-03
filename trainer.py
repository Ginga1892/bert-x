import torch
import torch.nn as nn
from pretrain import BertPreTrain
from finetune import BertClassification


class Trainer(object):
    def __init__(self, bert_model):
        self.bert_model = bert_model

    def pre_train(self, iterator, do_train, num_train_epochs=3, learning_rate=1e-5):
        print("***** Pre-training *****")
        model = BertPreTrain(self.bert_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        if do_train:
            print("***** Running training *****")
            model.train()

            for e in range(num_train_epochs):
                epoch_loss = 0
                for batch in iterator:
                    input_ids = batch['input_ids']
                    input_mask = batch['input_mask']
                    segment_ids = batch['segment_ids']
                    mlm_positions = batch['mlm_positions']
                    mlm_ids = batch['mlm_ids']
                    mlm_weights = batch['mlm_weights']
                    nsp_label = batch['nsp_label']

                    mlm_output, nsp_output = model(input_ids, input_mask, segment_ids, mlm_positions)

                    mlm_loss = get_mlm_loss(mlm_output, mlm_ids, mlm_weights)
                    nsp_loss = get_nsp_loss(nsp_output, nsp_label)

                    total_loss = mlm_loss + nsp_loss
                    epoch_loss += total_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                print('Epoch {}: loss = {}'.format(e + 1, epoch_loss))

    def fine_tune(self, iterator, do_train, num_train_epochs=3, learning_rate=1e-5):
        print("***** Fine tuning *****")
        model = BertClassification(self.bert_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        if do_train:
            print("***** Running training *****")
            model.train()

            for e in range(num_train_epochs):
                epoch_loss = 0
                for batch in iterator:
                    input_ids = batch['input_ids']
                    input_mask = batch['input_mask']
                    segment_ids = batch['segment_ids']
                    label = batch['label']

                    prediction_scores = model(input_ids, input_mask, segment_ids)

                    loss = criterion(prediction_scores, label)
                    epoch_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print('Epoch {}: loss = {}'.format(e + 1, epoch_loss))


def get_mlm_loss(mlm_output, mlm_ids, mlm_weights):
    mlm_ids, mlm_weights = mlm_ids.view(-1), mlm_weights.view(-1, 1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(mlm_output.masked_fill(mlm_weights == 0, 0), mlm_ids)

    return loss


def get_nsp_loss(nsp_output, nsp_label):
    nsp_label = nsp_label.view(-1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(nsp_output, nsp_label)

    return loss
