from tqdm import tqdm

#TODO: get there all tweets
all_tweets = train_1 + train_2 + test_1 + test_2

def get_necessary_words(tweets):
	necessary_words = []
	for tweet in tqdm(tweets):
		for word in tweet:
			necessary_words.append(word)
	necessary_words = set(necessary_words)
	return necessary_words
    	
def save_filtered_embeddings(source_filename, necessary_words, result_filename):
	f = open(source_filename, 'r')
	lines = f.readlines()
        
	w_str, n_str = lines[0].split(' ')
	W = int(w_str)
	N = int(n_str)
        
	resf = open(result_filename, 'w')
        
	for line in tqdm(lines[1:]):
		word = line.split(' ')[0]
		if word in necessary_words:
			resf.write(line)
			
ne = get_necessary_words(all_tweets)
save_filtered_embeddings('embeddings/emb1.txt', ne, 'embeddings/kraby.txt')
