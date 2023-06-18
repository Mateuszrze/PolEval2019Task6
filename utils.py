import numpy as np
import subprocess

	

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
	

