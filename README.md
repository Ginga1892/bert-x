# BERT - X

A flexible BERT-type model in PyTorch. Here you can try types of models (BERT, ALBERT, SpanBERT, etc.) in easy mode.



## Usage

* Original BERT

```python
# Define the dataset, where we use Chinese wwm tokenizer
input_files = ['corpus.txt']
tokenizer = tokenization.ChineseWordpieceTokenizer('vocab.txt')
bert_data = BertData(input_files, tokenizer)
# Create training instances on the dataset, including MLM and NSP
bert_data.create_instances(max_seq_length=128,
                           dupe_factor=2,
                           mlm_prob=0.15,
                           max_predictions_per_seq=20)
# Get the training iterator with mini-batch
train_iterator = bert_data.get_iterator(batch_size=32)

# Define the BERT model
bert_config = BertConfig.from_json('bert_config.json')
model = Bert(bert_config)

# Define the trainer for the model
trainer = Trainer(model, bert_data)
# Run pretraining
trainer.pre_train(iterator=train_iterator,
                  do_train=True,
                  num_train_epochs=3,
                  learning_rate=1e-4)
```

