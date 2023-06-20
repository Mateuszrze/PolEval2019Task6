import numpy as np
from tqdm import tqdm
import copy
import torch

import utils

class TweetToVec:
    
	def __init__(self, embeddings, method = 'fixed_length', L = None):
        
		self.N = embeddings['N']
		self.W = embeddings['N']
		self.embeddings = embeddings['vectors']
        
		self.method = method

		self.L = L
    
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
		vec = np.concatenate(np.array(vec))
        
		return vec
	
	def fixed_length_2d(self, tweet):
        
		vec = self.get_list_of_embeddings(tweet)
        
		while not self.long_enough(vec):
			vec.append(self.get_null_embedding())
        
		vec = vec[:self.L]
        
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
        
		vec = np.concatenate(np.array(vec))
        
		return vec
        	
	def take_all_2d(self, tweet):
        
		vec = self.get_list_of_embeddings(tweet)
		vec = np.array(vec)
                
		return vec


	def translate_single(self, tweet):
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
		
		if self.method == 'take_all_2d':
			return self.take_all_2d(tweet)
		
		if self.method == 'fixed_length_2d':
			return self.fixed_length_2d(tweet)
	
	def translate_to_vectors(self, tweets):
		
		vectors = []
		for tweet in tweets:
			vec = self.translate_single(tweet)
			vectors.append(vec)
		
		return vectors
	
	def group_into_batches(self, vecs, batch_size = 32):
		
		batches = []
		current_batch = []
    
		for v in vecs:
		
			current_batch.append(v)
			if len(current_batch) == batch_size:
				batches.append(np.array(current_batch))
				current_batch = []
		
		if len(current_batch):
			batches.append(np.array(current_batch))
		
		return batches
	
	def to_tensor(self, data):
		#TODO: watch out for cpu/gpu!
		return torch.from_numpy(data)
	
	def get_batched_data(self, vectors, tags, batch_size = 32, as_tensors = True):
		
		batched_vectors = self.group_into_batches(vectors, batch_size)
		batched_tags = self.group_into_batches(tags, batch_size)
		
		if as_tensors:
			batched_vectors = [self.to_tensor(batch).float() for batch in batched_vectors]
			batched_tags = [self.to_tensor(batch).long() for batch in batched_tags]
		
		return batched_vectors, batched_tags
	
	def vectorize_dataset(self, dataset):
	
		vectorized_dataset = copy.deepcopy(dataset)
		
		vectorized_dataset['training tweets'] = self.translate_to_vectors(dataset['training tweets'])
		vectorized_dataset['test tweets'] = self.translate_to_vectors(dataset['test tweets'])
		if 'validation tweets' in vectorized_dataset.keys():
			vectorized_dataset['validation tweets'] = self.translate_to_vectors(dataset['validation tweets'])

		return vectorized_dataset
	
	def batch_dataset(self, dataset, batch_size = 32, as_tensors = True, random_shuffle = False):
		
		batched_dataset = copy.deepcopy(dataset)

		if random_shuffle:
			batched_dataset = utils.random_shuffle_train_dataset(batched_dataset)
		
		batched_vectors, batched_tags = self.get_batched_data(vectors = batched_dataset['training tweets'], tags = batched_dataset['training tags'], batch_size = batch_size, as_tensors = as_tensors)
		
		batched_dataset['training tweets'] = batched_vectors
		batched_dataset['training tags'] = batched_tags
		
		batched_vectors, batched_tags = self.get_batched_data(vectors = batched_dataset['test tweets'], tags = batched_dataset['test tags'], batch_size = len(batched_dataset['test tweets']), as_tensors = as_tensors)
		
		batched_dataset['test tweets'] = batched_vectors[0]
		batched_dataset['test tags'] = batched_tags[0]

		if 'validation tweets' in batched_dataset.keys():
			batched_vectors, batched_tags = self.get_batched_data(vectors = batched_dataset['validation tweets'], tags = batched_dataset['validation tags'], batch_size = batch_size, as_tensors = as_tensors)
			
			batched_dataset['validation tweets'] = batched_vectors
			batched_dataset['validation tags'] = batched_tags


		return batched_dataset

	def vectorize_and_batch_dataset(self, dataset, batch_size, equalize_training_classes):
		dataset = copy.deepcopy(dataset)
		if equalize_training_classes:
			dataset = utils.equalize_training_classes(dataset)
		vectorized = self.vectorize_dataset(dataset)
		batched = self.batch_dataset(vectorized, batch_size)
		return batched