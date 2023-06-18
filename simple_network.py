import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNetwork(nn.Module):
    
	def __init__(self, model, loss):
        
		super().__init__()
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
        
		return self.model(x)
	
	def predict(self, x):
		
		predictions = self.model(x)
		return torch.argmax(predictions, dim = 1)
    

	def train(self, training_data, training_classes, epochs, get_optimizer):


		optimizer = get_optimizer(self.model.parameters())
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        
		for epoch in range(epochs):
			tot_loss = 0
			losses = []
			for (input_batch, true_classes) in zip(training_data, training_classes):
			
				preds = self.model(input_batch) 
				loss = self.loss(preds, true_classes)
				tot_loss += loss
				losses.append(float(loss))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print(f'After epoch {epoch} tot_loss = {tot_loss}')
			scheduler.step()
	
	
