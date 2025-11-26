from tokenizers import Tokenizer
import re

def create_bpe():
    tokenizer = Tokenizer.from_pretrained("Shu-vi/Russian_BPE_Tokenizer_16k")
    def _inner(text: str):
        return tokenizer.encode(text).tokens
    return _inner

def tokenize_naive(text: str):
    # Простая токенизация по пробелам (и отделяем лишние пунктуации у концов)
    parts = text.split()
    tokens = [p.strip("«»()[]{}.,:;!?\"'“”—–…") for p in parts if p.strip("«»()[]{}.,:;!?\"'“”—–…")]
    return tokens

def tokenize_regex(text: str):
    return re.compile(r"[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*", flags=re.UNICODE).findall(text)