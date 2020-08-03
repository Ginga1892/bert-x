import torch
import numpy as np
import random
import csv


class PreTrainData(object):
    def __init__(self, input_file, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.documents = [[]]
        with open(input_file, "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    self.documents.append([])
                else:
                    tokens = self.tokenizer.tokenize(line)
                    if tokens:
                        self.documents[-1].append(tokens)

        self.documents = [x for x in self.documents if x]
        random.shuffle(self.documents)

        self.instances = None
        self.iterator = None

    def create_instances(self, max_seq_length, dupe_factor,
                         mlm_prob, max_predictions_per_seq,
                         method='nsp'):
        self.max_seq_length = max_seq_length
        self.max_predictions_per_seq = max_predictions_per_seq

        vocab_words = list(self.vocab.keys())
        self.instances = []
        if method == 'nsp':
            for _ in range(dupe_factor):
                for document_index in range(len(self.documents)):
                    self.instances.extend(
                        self.create_nsp(
                            document_index, max_seq_length, mlm_prob,
                            max_predictions_per_seq, vocab_words))
        else:
            for _ in range(dupe_factor):
                for document_index in range(len(self.documents)):
                    self.instances.extend(
                        self.create_sop(
                            document_index, max_seq_length, mlm_prob,
                            max_predictions_per_seq, vocab_words))

        random.shuffle(self.instances)
        return self.instances

    def get_iterator(self, config):
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
            'nsp_label'
        ]

        self.iterator = []
        batch = dict(zip(features, [[] for _ in features]))
        num_instances = 0
        for instance in self.instances:
            input_ids = [
                self.vocab[token] if token in self.vocab else self.vocab['[UNK]'] for token in instance['tokens']]
            input_mask = [1] * len(input_ids)
            segment_ids = instance['segment_ids']
            while len(input_ids) < config.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            mlm_positions = instance['mlm_positions']
            mlm_ids = [
                self.vocab[label] if label in self.vocab else self.vocab['[UNK]'] for label in instance['mlm_labels']]
            mlm_weights = [1.0] * len(mlm_ids)
            while len(mlm_positions) < config.max_predictions_per_seq:
                mlm_positions.append(0)
                mlm_ids.append(0)
                mlm_weights.append(0.0)

            nsp_label = 0 if instance['is_next'] else 1

            batch['input_ids'].append(input_ids)
            batch['input_mask'].append(input_mask)
            batch['segment_ids'].append(segment_ids)
            batch['mlm_positions'].append(mlm_positions)
            batch['mlm_ids'].append(mlm_ids)
            batch['mlm_weights'].append(mlm_weights)
            batch['nsp_label'].append(nsp_label)

            num_instances = (num_instances + 1) % config.batch_size
            if num_instances == 0:
                self.iterator.append(to_tensor(batch))
                batch = dict(zip(features, [[] for _ in features]))

        return self.iterator

    def create_nsp(self, config):
        sp_instances = []
        for document in self.documents:
            # [CLS], [SEP], [SEP]
            max_num_tokens = config.max_seq_length - 3
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
                        # Random sentence
                        if len(current_chunk) == 1 or random.random() < 0.5:
                            is_next = False
                            # Random document
                            random_document_index = random.randint(0, len(self.documents) - 1)
                            random_document = self.documents[random_document_index]
                            random_start = random.randint(0, len(random_document) - 1)
                            for j in range(random_start, len(random_document)):
                                tokens_b.extend(random_document[j])
                                if len(tokens_b) >= target_b_length:
                                    break
                            num_unused_segments = len(current_chunk) - a_end
                            i -= num_unused_segments
                        # Next sentence
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

                        instance = {
                            'tokens': tokens,
                            'segment_ids': segment_ids,
                            'is_next': is_next,
                        }
                        instances.append(instance)
                    current_chunk = []
                    current_length = 0
                i += 1

            sp_instances.extend(instances)

        self.instances = sp_instances

    def create_sop(self, config):
        sp_instances = []
        for document in self.documents:
            # [CLS], [SEP], [SEP]
            max_num_tokens = config.max_seq_length - 3
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
                        # Random sentence
                        if len(current_chunk) == 1 or random.random() < 0.5:
                            is_next = False
                            for j in range(a_end, len(current_chunk)):
                                tokens_b.extend(current_chunk[j])
                            # Swap
                            tokens_a, tokens_b = tokens_b, tokens_a
                        # Next sentence
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

                        instance = {
                            'tokens': tokens,
                            'segment_ids': segment_ids,
                            'is_next': is_next
                        }
                        instances.append(instance)
                    current_chunk = []
                    current_length = 0
                i += 1

            sp_instances.extend(instances)

        self.instances = sp_instances

    def create_mlm(self, config):
        vocab_words = list(self.vocab.keys())
        mlm_instances = []
        for instance in self.instances:
            tokens = instance['tokens']

            candidates = []
            for i in range(len(tokens)):
                if tokens[i] == '[CLS]' or tokens[i] == '[SEP]':
                    continue
                if len(candidates) >= 1 and tokens[i].startswith("##"):
                    candidates[-1].append(i)
                candidates.append([i])
            random.shuffle(candidates)

            output_tokens = list(tokens)
            num_to_predict = min(config.max_predictions_per_seq,
                                 max(1, int(round(len(tokens) * config.mlm_prob))))
            mlms = []
            for candidate in candidates:
                if len(mlms) >= num_to_predict:
                    break
                if len(mlms) + len(candidate) > num_to_predict:
                    continue
                for index in candidate:
                    if random.random() < 0.8:
                        masked_token = '[MASK]'
                    else:
                        if random.random() < 0.5:
                            masked_token = masked_token = tokens[index]
                        else:
                            masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]
                    output_tokens[index] = masked_token
                    mlms.append([index, tokens[index]])

            assert len(mlms) <= num_to_predict
            mlms = sorted(mlms, key=lambda x: x[0])

            mlm_positions = []
            mlm_labels = []
            for mlm in mlms:
                mlm_positions.append(mlm[0])
                mlm_labels.append(mlm[1])

            instance['tokens'] = output_tokens
            instance['mlm_positions']: mlm_positions
            instance['mlm_labels']: mlm_labels
            mlm_instances.append(instance)

        self.instances = mlm_instances

    def create_sbo(self, config):
        vocab_words = list(self.vocab.keys())
        sbo_instances = []
        for instance in self.instances:
            tokens = instance['tokens']

            candidates = []
            for i in range(len(tokens)):
                if tokens[i] == '[CLS]' or tokens[i] == '[SEP]':
                    continue
                if len(candidates) >= 1 and tokens[i].startswith("##"):
                    candidates[-1].append(i)
                candidates.append([i])
            random.shuffle(candidates)

            output_tokens = list(tokens)
            num_to_predict = min(config.max_predictions_per_seq,
                                 max(1, int(round(len(tokens) * config.mlm_prob))))

            maksed = set()
            while len(maksed) <= num_to_predict:
                span_length = random.choice(np.random.geometric(p=0.2, size=10))
                random_start = random.choice([x[0] for x in candidates])
                for i in range(span_length):
                    if random_start + i >= len(tokens):
                        break
                    maksed.add(random_start + i)

            mlms = []
            for candidate in candidates:
                if len(mlms) >= num_to_predict:
                    break
                if candidate[0] in maksed:
                    if len(mlms) + len(candidate) > num_to_predict:
                        continue
                    for index in candidate:
                        if random.random() < 0.8:
                            masked_token = '[MASK]'
                        else:
                            if random.random() < 0.5:
                                masked_token = masked_token = tokens[index]
                            else:
                                masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]
                        output_tokens[index] = masked_token
                        mlms.append([index, tokens[index]])

            assert len(mlms) <= num_to_predict
            mlms = sorted(mlms, key=lambda x: x[0])

            mlm_positions = []
            mlm_labels = []
            for mlm in mlms:
                mlm_positions.append(mlm[0])
                mlm_labels.append(mlm[1])

            sbo_boundaries = []
            sbo_labels = []
            on_left = True
            lb = None
            for i in range(len(mlm_positions)):
                if i == len(mlm_positions) - 1 or mlm_positions[i + 1] - mlm_positions[i] > 1:
                    on_left = False
                if on_left:
                    if not lb:
                        lb = i
                else:
                    rb = i
                    if lb:
                        sbo_boundaries.append([mlm_positions[lb] - 1, mlm_positions[rb] + 1])
                        sbo_labels.append(mlm_labels[lb: rb + 1])
                    else:
                        sbo_boundaries.append([mlm_positions[rb] - 1, mlm_positions[rb] + 1])
                        sbo_labels.append([mlm_labels[rb]])
                    on_left = True
                    lb = None

            instance['tokens'] = output_tokens
            instance['mlm_positions'] = mlm_positions
            instance['mlm_labels'] = mlm_labels
            instance['sbo_boundaries'] = sbo_boundaries
            instance['sbo_labels'] = sbo_labels
            sbo_instances.append(instance)

        self.instances = sbo_instances


class FineTuneData(object):
    def __init__(self, train_file, dev_file, test_file, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.train_file = train_file

    def create_instances(self, max_seq_length):
        self.train_instances = self._create_instances(self.train_file, max_seq_length)
        self.dev_instances = self._create_instances(self.dev_file, max_seq_length)
        self.test_instances = self._create_instances(self.test_file, max_seq_length)

        return self.train_instances, self.dev_instances, self.test_instances

    def _create_instances(self, dataset_file, max_seq_length):
        max_num_tokens = max_seq_length - 3
        with open(dataset_file, "r", encoding='utf-8') as reader:
            csv_reader = csv.reader(reader, delimiter='\t')
            instances = []
            for line in csv_reader:
                # guid = line[0]
                tokens_a = self.tokenizer.tokenize(line[1])
                tokens_b = self.tokenizer.tokenize(line[2])
                label = line[-1]

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
                instance = {
                    'tokens': tokens,
                    'segment_ids': segment_ids,
                    'label': label
                }
                instances.append(instance)

        return instances

    def get_iterator(self, batch_size):
        def to_tensor(list_data):
            for key, value in list_data.items():
                list_data[key] = torch.LongTensor(value)
            return list_data

        features = [
            'input_ids',
            'input_mask',
            'segment_ids',
            'label'
        ]

        self.iterator = []
        batch = dict(zip(features, [[] for _ in features]))
        num_instances = 0
        for instance in self.instances:
            input_ids = [
                self.vocab[token] if token in self.vocab else self.vocab['[UNK]'] for token in instance['tokens']]
            input_mask = [1] * len(input_ids)
            segment_ids = instance['segment_ids']
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            label = 0 if instance['label'] else 1

            batch['input_ids'].append(input_ids)
            batch['input_mask'].append(input_mask)
            batch['segment_ids'].append(segment_ids)
            batch['label'].append(label)

            num_instances = (num_instances + 1) % batch_size
            if num_instances == 0:
                self.iterator.append(to_tensor(batch))
                batch = dict(zip(features, [[] for _ in features]))

        return self.iterator
