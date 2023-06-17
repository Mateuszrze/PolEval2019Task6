import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize as nlkt_tokenize

def replace_specials(text):
    
    specials = {
        '\u0026gt;' : '>',
        '\u0026lt;' : '<',
        '\u0026amp;' : '&',
        '\\n' : ' ',
        '\\r' : ' ',
        '\\"' : '"'
        
    }
    
    for special in specials.keys():
        text = text.replace(special, specials[special])
    
    return text
    
def naive_tokenize(text):
    return text.split(' ')

def tokenize_with_nlkt(text):
    
    text_clean = replace_specials(text)
    text_lowered = text_clean.lower()
    tokenized = nlkt_tokenize(text_lowered)
    
    return tokenized
    

class DataReader:
    
    def __init__(self, default_tokenize):
        
        self.default_tokenize = default_tokenize
    
    def read_data(self, filepath, tokenize = None):
        
        if tokenize is None:
            tokenize = self.default_tokenize
        
        data = []
        
        my_file = open(filepath, 'r')
        for line in tqdm(my_file.readlines()):
            data.append(tokenize(line))
        
        return data
    
    def read_tags(self, filepath):
        
        tags = []
        
        my_file = open(filepath, 'r')
        for line in tqdm(my_file.readlines()):
            tags.append(int(line))
        
        return tags
