import torch
from tqdm import tqdm
import numpy as np
from torch import nn, optim
import torch
import torch.nn as nn
import torch.optim as optim

# first comment to the model: it learned that returning 0 is a good solution and kept doing that all the time. 
# Thus, we altered the dataset to contain the same amount of results for 0, 1 and 2
class LSTMModel(nn.Module):
    def __init__(self, size = 256, embedding_dim = 100, num_layers = 2, dropout = 0.2, number_of_output_classes = 2, L = 30, prediction_method = 'sum', learning_rate=0.0001):
        super(LSTMModel, self).__init__()
        self.lstm_size = size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.L = L
        
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=dropout,
        )
        self.fc = nn.Sequential(nn.Linear(self.lstm_size, number_of_output_classes))
        if prediction_method == 'last':
           self.postprocess_predictions = self.compute_predictions2
        if prediction_method == 'sum':
           self.postprocess_predictions = self.compute_predictions3
        if prediction_method == 'max':
           self.postprocess_predictions = self.compute_predictions
        self.epoch = 0
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, prev_state):
        output, state = self.lstm(x, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self):
        return (torch.zeros(self.num_layers, self.L, self.lstm_size),
                torch.zeros(self.num_layers, self.L, self.lstm_size))

    def init_params(self):

        with torch.no_grad():
            for name, p in self.named_parameters():
                if "weight" in name:
                    p.normal_(0, np.sqrt(1 / (2 * p.size(dim = 1))))
                elif "bias" in name:
                    p.zero_()

    def compute_predictions(self, predictions):
        results = [[-1000000000 for j in range(predictions.shape[2])] for i in range(predictions.shape[0])]
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[2]):
                for k in range(predictions.shape[1]):
                    results[i][j] = max(results[i][j], predictions[i][k][j])
        predictions = predictions.sum(axis = 1)
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                predictions[i][j] = results[i][j]
        return predictions

    def compute_predictions2(self, predictions):
        return predictions[:, -1, :]
    
    def compute_predictions3(self, predictions):
        return predictions.sum(axis = 1)
    
    def predict(self, x):
        state_h, state_c = self.init_state()
        predictions, (_, _) = self(x, (state_h, state_c))
        return torch.argmax(self.postprocess_predictions(predictions), dim = 1)
    
    def train_model(self, gen_batched, epochs):
        self.train()
        
        for epoch in range(epochs):
            text, tags = gen_batched()
            tot_loss = 0
            losses = []
            for (input_batch, true_classes) in zip(text, tags):
                state_h, state_c = self.init_state()
                y_pred, (state_h, state_c) = self(input_batch, (state_h, state_c))
                loss = self.criterion(self.postprocess_predictions(y_pred), true_classes)
                tot_loss += loss
                losses.append(float(loss))

                state_h = state_h.detach()
                state_c = state_c.detach()            

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'After epoch {self.epoch} tot_loss = {tot_loss}', end = '')
            self.scheduler.step()
            self.epoch += 1
                
        