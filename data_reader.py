import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize as nlkt_tokenize

try:
	from transformers import BertTokenizer	
except:
	print("Warning! BERT not installed (if you are Rzepa, it's no problem)")

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

def delete_spam(text):

	spam_list = [
		'@anonymized_account'
		]
	
	for spam in spam_list:
		text = text.replace(spam, '')
	
	return text

def no_cleaning(text):
	return text

def strict_cleaning(text):

	text_clean = replace_specials(text)
	text_very_clean = delete_spam(text)
	return text_very_clean

def naive_tokenize(text):
	return text.split(' ')

def tokenize_with_nlkt(text):
    
	text_lowered = text.lower()
	tokenized = nlkt_tokenize(text_lowered)
    
	return tokenized
    
def tokenize_for_bert(text):

	bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	bert_tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")


class DataReader:
    
	def __init__(self, default_tokenize, cleaning_policy = 'standard'):
		
		tokenize_methods = {
			'nlkt' : tokenize_with_nlkt,
			'naive' : naive_tokenize,	
			'for_bert' : tokenize_for_bert
		}
		
		cleaning_policies = {
			'no' : no_cleaning,
			'standard' : replace_specials,
			'strict' : strict_cleaning,
		}
		
		self.default_tokenize = tokenize_methods[default_tokenize]
		self.default_clean = cleaning_policies[cleaning_policy]
		self.datasets = {}
    
	def read_data(self, filepath, tokenize = None, use_tqdm = True):
	 
		if tokenize is None:
			tokenize = self.default_tokenize
		clean_text = self.default_clean
        
		data = []
        
		my_file = open(filepath, 'r')
		i = my_file.readlines()
		if use_tqdm:
			i = tqdm(i)
		for line in i:
			data.append(tokenize(clean_text(line)))
        
		return data
    
	def read_tags(self, filepath, use_tqdm = True):
	        
		tags = []
        
		my_file = open(filepath, 'r')
		i = my_file.readlines()
		if use_tqdm:
			i = tqdm(i)
		for line in i:
			tags.append(int(line))
        
		return tags
    
	def read_dataset(self, description, use_tqdm = True):
	
		training_tweets_file = description['training tweets']
		training_tags_file = description['training tags']
		
		test_tweets_file = description['test tweets']
		test_tags_file = description['test tags']
		
		training_tweets = self.read_data(training_tweets_file, use_tqdm=use_tqdm)
		training_tags = self.read_tags(training_tags_file, use_tqdm=use_tqdm)
		
		test_tweets = self.read_data(test_tweets_file, use_tqdm=use_tqdm)
		test_tags = self.read_tags(test_tags_file, use_tqdm=use_tqdm)
		
		
		dataset = {
			'training tweets' : training_tweets,
			'training tags' : training_tags,
			'test tweets' : test_tweets,
			'test tags' : test_tags
			}
		
		name = description['name']
		self.datasets[name] = dataset

	def read_embeddings(self, filename, use_tqdm = True):
	
		vectors = dict()
        
		f = open(filename, 'r')
		lines = f.readlines()
        
		w_str, n_str = lines[0].split(' ')
		W = int(w_str)
		N = int(n_str)
        
		l = lines[1:]
		if use_tqdm:
			l = tqdm(l)
		for line in l:
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
	
		
