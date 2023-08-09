"""
Code from https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark/blob/main/Scripts/Models/LSTM.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


batch_first =True




class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size ,num_classes , rnndropout):
        super().__init__()
        self.device=device
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(rnndropout)
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True)



    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device) if len(
            x.size()) > 2 else torch.zeros(1, self.hidden_size).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device) if len(
            x.size()) > 2 else torch.zeros(1, self.hidden_size).to(self.device)
        h0 = h0.float()
        c0 = c0.float()
        x = self.drop(x)
        output, _ = self.rnn(x, (h0, c0))
        output = self.drop(output)
        output = output[:, -1, :] if len(
            x.size()) > 2 else output[-1, :]
        out = self.fc(output)
        out = F.softmax(out, dim=1) if len(
            x.size()) > 2 else F.softmax(out, dim=0)
        return out
