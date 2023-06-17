import numpy as np

class AbstractClassifier:
    
    def __init__(self, classifier):
        
        self.classifier = classifier
    
    def classify(self, tweet):
        
        return np.argmax(self.classifier.tweet_class_distribution(tweet))
            
    def classify_many(self, tweets):
        
        answer = [self.classify(tweet) for tweet in tweets]
        
        return answer
    
    def run_and_save(self, tweets, filename):
        
        answer = self.classify_many(tweets)
        
        f = open(filename, 'w+')
        for ans in answer:
            f.write(str(ans) + '\n')
