import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sqrtm import sqrtm

class MyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)
        self.dropout1=nn.Dropout(0.1)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)
        self.leakyrelu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.hidden1(x)
        x = self.leakyrelu1(x)
        res_1=x
        x = self.hidden2(x)
        x = self.leakyrelu2(x)
        x = self.dropout1(x)
        x=x+res_1
        x = self.output_layer(x)
        return x


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size):
        super(MyRNN, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True, nonlinearity='tanh')
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.view(-1, input_dim)  # Reshape input to (batch_size*seq_len, input_dim)
        x = self.linear(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, hidden_size)
        output, _ = self.rnn(x)
        output = self.output_layer(output[:, -1, :])  # Take the last time-step output
        return output


class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LinearNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
