# Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# CNN_Section
# CNN_0
class CNN_0(nn.Module):
    def __init__(self):
        super(CNN_0, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 73, 256)
        self.fc2 = nn.Linear(256, 4)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # print(x.shape)
        x = x.view(-1, 64 * 73)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
# CNN_1
class CNN_1(nn.Module):
    def __init__(self):
      super(CNN_1, self).__init__()
      self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU())
      self.pool1 = nn.MaxPool1d(kernel_size=2)
      self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU())
      self.pool2 = nn.MaxPool1d(kernel_size=2)
      self.fc1 = nn.Linear(64 * 73, 256)
      self.fc2 = nn.Linear(256, 4)
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 73)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
# LSTM_0
class LSTM_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_0, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
