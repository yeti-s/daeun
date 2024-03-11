import re
from .korean import latin_to_hangul, number_to_hangul, divide_hangul, g2pk
from .symbols import symbols
_cleaner_cleans = re.compile('['+'^'.join(symbols)+']')


def korean_cleaners(text):
    '''Pipeline for Korean text'''
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = g2pk(text)
    text = divide_hangul(text)
    text = re.sub(r'([\u3131-\u3163])$', r'\1.', text)
    return text

def jk_cleaners(text):
    text = re.sub(r'\[KO\](.*?)\[KO\]', lambda x: korean_cleaners(x.group(1))+' ', text)
    text = re.sub(r'\[PREPROCESSED\](.*?)\[PREPROCESSED\]', lambda x: x.group(1)+' ', text)
    text = ''.join(_cleaner_cleans.findall(text))
    return text