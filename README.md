# BERT - X

An unified model of BERTs in PyTorch. Here you can try multiple models (BERT, ALBERT, SpanBERT, etc.) in easy mode.

## Supported Models

* BERT
* ALBERT
* SpanBERT

## Flexibility

We encourage you to create your own BERT model based on flexible pretrain models (PTMs).

## Easy-Usage

* Original BERT

```python
# Define the pre-training dataset, where we use Chinese wwm tokenizer
input_files = ['corpus.txt']
tokenizer = tokenization.ChineseWordpieceTokenizer('vocab.txt')
bert_data = PreTrainData(input_files, tokenizer)
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
trainer = Trainer(model)
# Run pre-training
trainer.pre_train(iterator=train_iterator,
                  do_train=True,
                  num_train_epochs=3,
                  learning_rate=1e-4)

# Define the fine-tuning dataset
glue_data = FineTuneData(train_file, dev_file, test_file, tokenizer)
# Create instances and get the iterator
gule_data.create_instances(max_seq_length=128)
iterator = bert_data.get_iterator(batch_size=32)
# Run fine-tuning
trainer.fine_tune(iterator=iterator,
                  do_train=True,
                  num_train_epochs=3,
                  learning_rate=1e-4)
```

