import numpy as np
import subprocess
import copy



def str_to_vector(vec_str, starts_with_word=False):
    
	splitted_vec_str = vec_str.split(' ')
    
	if starts_with_word:
		word = splitted_vec_str[0]
		splitted_vec_str = splitted_vec_str[1:]
    
	vec = np.array([float(vi) for vi in splitted_vec_str])
    
	if starts_with_word:
		return vec, word
    
	return vec



def save_results(predictions, filename):
	
	f = open(filename, 'w')
	
	for pred in predictions:
		f.write(str(int(pred)) + '\n')
	


def random_shuffle_train_dataset(dataset):
	new_dataset = copy.deepcopy(dataset)
	p = np.random.permutation(len(dataset['training tweets']))
	for i in range(len(dataset['training tweets'])):
		new_dataset['training tweets'][i] = dataset['training tweets'][p[i]] 
		new_dataset['training tags'][i] = dataset['training tags'][p[i]] 
	return new_dataset


# this function augments dataset to have more or less the same number of answers 
def equalize_training_classes(dataset):
	dataset = copy.deepcopy(dataset)
	vals = [[], [], []]
	for i in range(len(dataset['training tags'])):
		vals[dataset['training tags'][i]].append(dataset['training tweets'][i])
	size = max(len(vals[0]), len(vals[1]), len(vals[2]))
	def add_to(l):
		start_size = len(l)
		while len(l) < size:
			l.append(l[np.random.randint(0, start_size)])
		return l
	dataset['training tweets'] = []
	dataset['training tags'] = []
	for j in range(3):
		if len(vals[j]) > 0:
			for i in add_to(vals[j]):
				dataset['training tweets'].append(i)
				dataset['training tags'].append(j)
	random_shuffle_train_dataset(dataset)
	return dataset

def extract_validation_from_training(dataset, percent=0.1):
	size = int(percent * len(dataset['training tweets']))
	new_dataset = copy.deepcopy(dataset)
	new_dataset['training tweets'] = []
	new_dataset['training tags'] = []
	new_dataset['validation tweets'] = []
	new_dataset['validation tags'] = []
	validation_indexes = set(np.random.default_rng(0).choice(len(dataset['training tweets']), size, replace=False))
	for i in range(len(dataset['training tweets'])):
		if i in validation_indexes:
			new_dataset['validation tweets'].append(dataset['training tweets'][i])
			new_dataset['validation tags'].append(dataset['training tags'][i])		
		else:
			new_dataset['training tweets'].append(dataset['training tweets'][i])
			new_dataset['training tags'].append(dataset['training tags'][i])
	return new_dataset