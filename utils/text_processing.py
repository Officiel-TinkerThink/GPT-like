import urllib.request
import re
import yaml

def download_text(url, target_path):
    urllib.request.urlretrieve(url, target_path)

def build_vocab(text: str, special_tokens) -> dict:
    preprocessed = re.split(r'([,.:;?_"()\']|--|\s)', text)
    preprocessed = [
        item.strip() for item in preprocessed if item.strip()
    ]
    all_tokens = sorted(list(set(preprocessed)))
    if special_tokens is not None:
        all_tokens.extend(list(special_tokens))
    vocab = {token:integer for integer, token in enumerate(all_tokens)}
    return vocab

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text 

def read_and_build(file_path, special_tokens=None):
    text = read_file(file_path)
    return build_vocab(text, special_tokens)

def read_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt")
    file_path = 'the-verdict.txt'
    download_text(url, file_path)
    text = read_file(file_path)
    vocab = build_vocab(text)
    print(list(vocab.items())[-5:])
    