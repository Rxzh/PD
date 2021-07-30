import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F



params = {'n_features':10,
        'n_hidden':100,
        'n_output' :1,
        'optimizer':'SGD',
        'loss_func':'MSE'}

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(params['n_features'], params['n_hidden'])
        self.fc2 = nn.Linear(params['n_hidden'], params['n_output'])


    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class Model:
    def __init__(self, params = params):

        self.net = Net(params)

        if params['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.2)
        if params['loss_func'] == 'MSE':
            self.loss_func = nn.MSELoss()



    def fit(self,x ,y, n_epochs = 20, n_steps = 200):

        for epoch in range(n_epochs):
            running_loss = 0.0
            for step in range(n_steps):
                
                #zero the gradients
                self.optimizer.zero_grad()

                #forward + backward + optimize
                prediction = self.net(x)     # input x and predict based on x
                loss = self.loss_func(prediction, y)     # must be (1. nn output, 2. targets
                loss.backward()              # backpropagation, compute gradients
                self.optimizer.step()        # apply gradients


                #Statistics
                running_loss += loss.item()
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss))
                running_loss = 0.0

    def predict(self, x):
        return self.net(x)