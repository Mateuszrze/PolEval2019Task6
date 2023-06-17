import numpy as np
from tqdm import tqdm

import utils

class TweetToVec:
    
    def __init__(self, method = 'fixed_length', L = None):
        
        self.method = method
        self.L = L
    
    def read_embeddings_from_file(self, filename):
        
        self.embeddings = dict()
        
        f = open(filename, 'r')
        lines = f.readlines()
        
        w_str, n_str = lines[0].split(' ')
        self.W = int(w_str)
        self.N = int(n_str)
        
        for line in tqdm(lines[1:]):
            vec, word = utils.str_to_vector(line, starts_with_word=True)
            self.embeddings[word] = vec
        
    
    def get_embedding(self, word):
        
        if word in self.embeddings.keys():
            return self.embeddings[word]
        return None
    
    def get_list_of_embeddings(self, tweet):
        
        vec = []
        for word in tweet:
            current_embedding = self.get_embedding(word)
            if current_embedding is not None:
                vec.append(current_embedding)
                
        return vec
        
    def get_null_embedding(self):
        
        return np.zeros(self.N)

    def long_enough(self, vec):
        
        return (len(vec) >= self.L)
    
    def fixed_length(self, tweet):
        
        vec = self.get_list_of_embeddings(tweet)
        
        while not self.long_enough(vec):
            vec.append(self.get_null_embedding())
        
        vec = vec[:self.L]
        print("!", np.array(vec).shape)
        vec = np.concatenate(np.array(vec))
        print("!", len(vec))
        
        return vec
    
    def average(self, tweet):
        
        vec = self.get_list_of_embeddings(tweet)
        vec = np.array(vec)
        #TODO: check if proper axis etc.
        vec = np.sum(axis = 0) / vec.shape[0]
        
        return vec
    
    def take_all(self, tweet):
        
        vec = self.get_list_of_embeddings(tweet)
        vec = np.array(vec)
        
        vec = np.concatenate(vec)
        
        return vec
        
    
    def translate_to_vec(self, tweet):
        '''
        tweet: list of strings
        Already filtered tweet, given as list of tokens/lemmas
        '''
        
        if self.method == 'fixed_length':
            return self.fixed_length(tweet)
        
        if self.method == 'average':
            return self.average(tweet)
        
        if self.method == 'take_all':
            return self.take_all(tweet)
        
