import torch
import random
import tokenization


class BertData(object):
    def __init__(self, input_files, vocab_file, tokenizer):
        self.vocab = {}
        index = 0
        with open(vocab_file, "r", encoding='utf-8') as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                self.vocab[token] = index
                index += 1
        self.vocab_size = len(self.vocab)

        self.documents = [[]]
        for input_file in input_files:
            with open(input_file, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        self.documents.append([])
                    else:
                        tokens = tokenizer.tokenize(line)
                        if tokens:
                            self.documents[-1].append(tokens)

        self.documents = [x for x in self.documents if x]
        random.shuffle(self.documents)

        self.instances = None
        self.iterator = None

    def create_instances(self, max_seq_length, dupe_factor,
                         masked_lm_prob, max_predictions_per_seq):
        self.max_seq_length = max_seq_length
        self.max_predictions_per_seq = max_predictions_per_seq

        vocab_words = list(self.vocab.keys())
        instances = []
        for _ in range(dupe_factor):
            for document_index in range(len(self.documents)):
                instances.extend(
                    self.create_nsp(document_index,
                                    max_seq_length, masked_lm_prob,
                                    max_predictions_per_seq, vocab_words)
                )

        random.shuffle(instances)
        self.instances = instances
        return self.instances

    def get_iterator(self, batch_size):
        def to_tensor(list_data):
            for key, value in list_data.items():
                list_data[key] = torch.LongTensor(value)
            return list_data

        features = [
            'input_ids',
            'input_mask',
            'segment_ids',
            'mlm_positions',
            'mlm_ids',
            'mlm_weights',
            'nsp_labels'
        ]

        iterator = []
        batch = dict(zip(features, [[] for _ in features]))
        num_instances = 0
        for instance in self.instances:
            input_ids = [self.vocab[token] if token in self.vocab else self.vocab['[UNK]'] for token in instance['tokens']]
            input_mask = [1] * len(input_ids)
            segment_ids = instance['segment_ids']
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            mlm_positions = instance['mlm_positions']
            mlm_ids = [self.vocab[label] if label in self.vocab else self.vocab['[UNK]'] for label in instance['mlm_labels']]
            mlm_weights = [1.0] * len(mlm_ids)
            while len(mlm_positions) < self.max_predictions_per_seq:
                mlm_positions.append(0)
                mlm_ids.append(0)
                mlm_weights.append(0.0)

            nsp_labels = int(instance['is_next'])

            batch['input_ids'].append(input_ids)
            batch['input_mask'].append(input_mask)
            batch['segment_ids'].append(segment_ids)
            batch['mlm_positions'].append(mlm_positions)
            batch['mlm_ids'].append(mlm_ids)
            batch['mlm_weights'].append(mlm_weights)
            batch['nsp_labels'].append(nsp_labels)

            num_instances = (num_instances + 1) % batch_size
            if num_instances == 0:
                iterator.append(to_tensor(batch))
                batch = dict(zip(features, [[] for _ in features]))

        self.iterator = iterator
        return self.iterator

    def create_nsp(self, document_index,
                   max_seq_length, masked_lm_prob,
                   max_predictions_per_seq, vocab_words):
        document = self.documents[document_index]
        # [CLS], [SEP], [SEP]
        max_num_tokens = max_seq_length - 3
        target_seq_length = max_num_tokens

        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = random.randint(1, len(current_chunk) - 1) if len(current_chunk) >= 2 else 1

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    target_b_length = target_seq_length - len(tokens_a)
                    # random sentence
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_next = False
                        # random document
                        random_document_index = random.randint(0, len(self.documents) - 1)
                        while random_document_index == document_index:
                            random_document_index = random.randint(0, len(self.documents) - 1)
                        random_document = self.documents[random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # next sentence
                    else:
                        is_next = True
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                            if len(tokens_b) >= target_b_length:
                                break
                    total_length = len(tokens_a) + len(tokens_b)
                    while total_length > max_num_tokens:
                        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                        if random.random() < 0.5:
                            del trunc_tokens[0]
                        else:
                            trunc_tokens.pop()
                        total_length -= 1

                    tokens = []
                    segment_ids = []
                    tokens.append("[CLS]")
                    segment_ids.append(0)
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)
                    tokens.append("[SEP]")
                    segment_ids.append(0)
                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)
                    tokens.append("[SEP]")
                    segment_ids.append(1)

                    tokens, mlm_positions, mlm_labels = self.create_mlm(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words)
                    instance = {
                        'tokens': tokens,
                        'segment_ids': segment_ids,
                        'is_next': is_next,
                        'mlm_positions': mlm_positions,
                        'mlm_labels': mlm_labels
                    }
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1

        return instances

    @staticmethod
    def create_mlm(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words):
        candidates = []
        for i in range(len(tokens)):
            if tokens[i] == '[CLS]' or tokens[i] == '[SEP]':
                continue
            candidates.append(i)
        random.shuffle(candidates)

        output_tokens = list(tokens)
        num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
        mlms = []
        for index in candidates:
            if len(mlms) >= num_to_predict:
                break
            if random.random() < 0.8:
                masked_token = '[MASK]'
            else:
                if random.random() < 0.5:
                    masked_token = masked_token = tokens[index]
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]
            output_tokens[index] = masked_token
            mlms.append([index, tokens[index]])
        mlms = sorted(mlms, key=lambda x: x[0])

        mlm_positions = []
        mlm_labels = []
        for mlm in mlms:
            mlm_positions.append(mlm[0])
            mlm_labels.append(mlm[1])

        return output_tokens, mlm_positions, mlm_labels
