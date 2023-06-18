import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize as nlkt_tokenize

import utils

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
		
		tokenize_methods = {
			'nlkt' : tokenize_with_nlkt,
			'naive' : naive_tokenize	
		}
		
		self.default_tokenize = tokenize_methods[default_tokenize]
		self.datasets = {}
    
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
    
	def read_dataset(self, description):
	
		training_tweets_file = description['training tweets']
		training_tags_file = description['training tags']
		
		test_tweets_file = description['test tweets']
		test_tags_file = description['test tags']
		
		training_tweets = self.read_data(training_tweets_file)
		training_tags = self.read_tags(training_tags_file)
		
		test_tweets = self.read_data(test_tweets_file)
		test_tags = self.read_tags(test_tags_file)
		
		
		dataset = {
			'training tweets' : training_tweets,
			'training tags' : training_tags,
			'test tweets' : test_tweets,
			'test tags' : test_tags
			}
		
		name = description['name']
		self.datasets[name] = dataset

	def read_embeddings(self, filename):
	
		vectors = dict()
        
		f = open(filename, 'r')
		lines = f.readlines()
        
		w_str, n_str = lines[0].split(' ')
		W = int(w_str)
		N = int(n_str)
        
		for line in tqdm(lines[1:]):
			vec, word = utils.str_to_vector(line, starts_with_word=True)
			vectors[word] = vec
        
		embeddings = {
			'N' : N,
			'W' : W,
			'vectors' : vectors
			}
		
		return embeddings
			
	
		
	def get_training_data(self, name):
	
		ds = self.datasets[name]
		return ds['training tweets'], ds['training tags']
	
	def get_test_data(self, name):
	
		ds = self.datasets[name]
		return ds['test tweets'], ds['test tags']
	
	def get_dataset(self, name):
	
		ds = self.datasets[name]
		return ds
	
		
