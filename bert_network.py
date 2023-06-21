import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformers import BertModel


class BertNetwork(nn.Module):
    
	def __init__(self, model, loss, bert_model_name = 'bert-base-cased'):
        
		super().__init__()
		self.bercior = BertModel.from_pretrained(bert_model_name)
		self.model = model
		self.loss = loss
		self.init_params()
        
	def init_params(self):
	
		with torch.no_grad():
			for name, p in self.model.named_parameters():
				if "weight" in name:
					p.normal_(0, np.sqrt(1 / (2 * p.size(dim = 1))))
				elif "bias" in name:
					p.zero_()
    
	def forward(self, x):
		
		spam, output_from_bert = self.bercior(input_ids = x['input_ids'].to(self.device), attention_mask = x['attention_mask'].to(self.device), return_dict=False)
		final_results = self.model(output_from_bert)
		return final_results
	
	def predict(self, x):
		
		predictions = self.forward(x)
		return torch.argmax(predictions, dim = 1)
    

	def train(self, training_data, training_classes, epochs, get_optimizer):
	
		optimizer = get_optimizer(self.model.parameters())
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
		
		use_cuda = torch.cuda.is_available()
		self.device = torch.device("cuda" if use_cuda else "cpu")
		self.to(self.device)
        
		for epoch in range(epochs):
			tot_loss = 0
			
			for (input_batch, true_classes) in tqdm(zip(training_data, training_classes)):
			
				preds = self.forward(input_batch) 
				loss = self.loss(preds, true_classes.to(self.device))
				tot_loss += loss
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print(f'After epoch {epoch} tot_loss = {tot_loss}')
			scheduler.step()
	
		
