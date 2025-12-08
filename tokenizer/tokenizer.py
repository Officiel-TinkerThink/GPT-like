import re
from utils.text_processing import read_and_build

class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = { i:s for i, s in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = { i:s for i, s in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else 
                        "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?_"()\'])', r'\1', text)
        return text

if __name__ == '__main__':
    file_path = 'the-verdict.txt'
    # read the text file and build vocab
    special_tokens = ['<|endoftext|>', '<|unk|>']
    vocab = read_and_build(file_path, special_tokens)
    # print(list(vocab.items())[-5:])
    

    text1 = 'Hello, do you like tea?'
    text2 = 'In the sunlit terraces of the palace.'
    text = ' <|endoftext|>'.join([text1, text2])
    print(text)

    tokenizer = SimpleTokenizerV2(vocab)
    print(tokenizer.encode(text))