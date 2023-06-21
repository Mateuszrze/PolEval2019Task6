import numpy as np
from tqdm import tqdm
import copy
import torch

def group_into_batches(vecs, batch_size = 32):
		
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

def group_into_batches_bert(vectors, batch_size = 32):
		
	batches = []
	current_batch_ids = []
	current_batch_attention = []
    
	for v in vectors:
		
		current_batch_ids.append(v['input_ids'])
		current_batch_attention.append(v['attention_mask'])
		if len(current_batch_ids) == batch_size:
			current_batch = {}
			current_batch['input_ids'] = torch.cat(current_batch_ids)
			current_batch['attention_mask'] = torch.cat(current_batch_attention)
			batches.append(current_batch)
			current_batch_ids = []
			current_batch_attention = []
		
	if len(current_batch_ids):
		current_batch = {}
		current_batch['input_ids'] = torch.cat(current_batch_ids)
		current_batch['attention_mask'] = torch.cat(current_batch_attention)
		batches.append(current_batch)
		
	return batches

def to_tensor(data):
	#TODO: watch out for cpu/gpu!
	return torch.from_numpy(data)

def get_batched_data(vectors, tags, batch_size = 32):
		
		batched_vectors = group_into_batches_bert(vectors, batch_size)
		batched_tags = group_into_batches(tags, batch_size)
		
		
		batched_tags = [to_tensor(batch).long() for batch in batched_tags]
		
		return batched_vectors, batched_tags


def batch_dataset(dataset, batch_size = 32, random_shuffle = False):
		
		batched_dataset = copy.deepcopy(dataset)

		if random_shuffle:
			batched_dataset = utils.random_shuffle_train_dataset(batched_dataset)
		
		batched_vectors, batched_tags = get_batched_data(vectors = batched_dataset['training tweets'], tags = batched_dataset['training tags'], batch_size = batch_size)
		
		batched_dataset['training tweets'] = batched_vectors
		batched_dataset['training tags'] = batched_tags
		
		batched_vectors, batched_tags = get_batched_data(vectors = batched_dataset['test tweets'], tags = batched_dataset['test tags'], batch_size = len(batched_dataset['test tweets']))
		
		batched_dataset['test tweets'] = batched_vectors[0]
		batched_dataset['test tags'] = batched_tags[0]

		if 'validation tweets' in batched_dataset.keys():
			batched_vectors, batched_tags = get_batched_data(vectors = batched_dataset['validation tweets'], tags = batched_dataset['validation tags'], batch_size = len(batched_dataset['validation tweets']))
			
			batched_dataset['validation tweets'] = batched_vectors[0]
			batched_dataset['validation tags'] = batched_tags[0]


		return batched_dataset


