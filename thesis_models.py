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
"""
Batch Normalization + Activation Function
"""
class CNN_1(nn.Module):
    def __init__(self):
      super(CNN_1, self).__init__()
      self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5),
            nn.BatchNorm1d(128),
            nn.Tanh())
      self.pool1 = nn.MaxPool1d(kernel_size=2)
      self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.Tanh())
      self.pool2 = nn.MaxPool1d(kernel_size=2)
      self.fc1 = nn.Linear(64 * 73, 256)
    #   self.dropout = nn.Dropout(0.2)
      self.fc2 = nn.Linear(256, 4)
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 73)
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = nn.ELU()(x)
        x = nn.Tanh()(x)
        x = self.fc2(x)
        return x
# CNN_2
"""
沒有比較好，反而有點overfitting，但可以參考一下調整的地方
"""
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=7), #增加kernel size, 增加維度
            nn.BatchNorm1d(256),
            nn.ELU())
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5), #增加kernel size
            nn.BatchNorm1d(128),
            nn.ELU())
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 34, 256)
        # self.dropout1 = nn.Dropout(0.1) #增加dropout
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        # print(x.shape) #Debug用
        x = x.view(-1, 64 * 34)
        x = self.fc1(x)
        # x = self.dropout1(x)
        # x = F.relu(x)
        x = nn.ELU()
        x = self.fc2(x)
        return x
# LSTM Section
# LSTM_0
class LSTM_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_0, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

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
# LSTM_1
"""
增加Activation Function
"""
class LSTM_1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_1, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Activation Function
        out = self.relu(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
# LSTM_2_bidirectional
"""
增加可以設定Bidirection是否開啟，調用時最後一個引數要設定True(使用)或False(不使用)
"""
class LSTM_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(LSTM_2, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        # self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*num_directions)
        # Activation Function
        # out = self.relu(out)
        out = self.elu(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
# LSTM_3_bidirectional_attention
"""
增加可以設定Bidirection是否開啟的同時，增加Attention Mechanism

This implementation uses a dot-product attention mechanism, which computes attention scores by taking the dot product
between a weight matrix U and the LSTM output tensor out, and the dot product between another weight matrix W and the 
final hidden state of the LSTM. 

These scores are then passed through a non-linear activation function (tanh) and multiplied by a weight matrix v to get 
the final attention weights. 

Finally, the attention-weighted sum of the LSTM output tensor is computed, flattened, and passed through a 
fully connected layer to generate the final output.
"""
class LSTM_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(LSTM_3, self).__init__()   
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.U = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
        self.W = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
        self.v = nn.Linear(hidden_size * self.num_directions, 1)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*num_directions)
        # Compute attention scores
        u_out = self.U(out)
        w_out = self.W(out[:, -1, :])
        scores = self.v(torch.tanh(u_out + w_out.unsqueeze(1)))
        attention_weights = self.softmax(scores)
        # Compute attention weighted sum
        context = torch.bmm(attention_weights.permute(0,2,1), out)
        # Flatten context tensor for feeding into the fully connected layer
        context = context.view(-1, self.hidden_size * self.num_directions)
        # Decode the context tensor
        out = self.fc(context)
        return out