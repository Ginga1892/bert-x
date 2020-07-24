import torch
import torch.nn as nn
from pretrain import BertPreTrain
import tokenization
from processor import BertData
from model import BertConfig, Bert


class Trainer(object):
    def __init__(self, bert_model):
        self.bert_model = bert_model

    def pre_train(self, iterator, do_train, num_train_epochs=3, learning_rate=1e-5):
        model = BertPreTrain(self.bert_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

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
                    nsp_labels = batch['nsp_labels']

                    prediction_scores, next_sentence_scores = model(input_ids, input_mask, segment_ids, mlm_positions)

                    mlm_mask = mlm_weights.unsqueeze(2).repeat(1, 1, self.bert_model.config.vocab_size)
                    mlm_loss = criterion(prediction_scores.masked_fill(mlm_mask == 0, 0).permute(0, 2, 1), mlm_ids)

                    nsp_loss = criterion(next_sentence_scores, nsp_labels)

                    total_loss = mlm_loss + nsp_loss
                    epoch_loss += total_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                print('Epoch {}: loss = {}'.format(e + 1, epoch_loss))
