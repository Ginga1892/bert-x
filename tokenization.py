import jieba


class ChineseWordpieceTokenizer(object):
    """Runs Chinese WordPiece tokenization using jieba.

    Example:
            input = "哈密瓜"
            output = ["哈", "##密", "##瓜"]
    """

    def __init__(self, vocab_file, unk_token="[UNK]", max_input_chars_per_word=200):
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
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        tokens = jieba.lcut(text)

        output_tokens = []
        for token in tokens:
            if len(token) == 1:
                if token in self.vocab:
                    output_tokens.append(token)
                else:
                    output_tokens.append(self.unk_token)
            else:
                is_bad = False
                sub_tokens = ' '.join(token).split()
                for i in range(len(sub_tokens)):
                    if sub_tokens[i] not in self.vocab:
                        is_bad = True
                        break
                    if i == 0:
                        continue
                    sub_tokens[i] = '##' + sub_tokens[i]
                if is_bad:
                    output_tokens.append(self.unk_token)
                else:
                    output_tokens.extend(sub_tokens)

        return output_tokens
