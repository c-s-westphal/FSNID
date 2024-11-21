import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class mine_net(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = self.fc2(output)
        return output

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleMLP, self).__init__()
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        log_probs = self.net(x)
        return log_probs

class SimpleLSTM(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_features, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)  # Apply softmax to the output dimension

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.linear(last_hidden_state)
        log_probs = self.log_softmax(output)
        return log_probs

class SimpleGRU(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes, num_layers=1):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_features, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden_state = gru_out[:, -1, :]
        output = self.linear(last_hidden_state)
        log_probs = self.log_softmax(output)
        return log_probs

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class SimpleTCN(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes, num_layers=1, kernel_size=2, dropout=0.2):
        super(SimpleTCN, self).__init__()
        layers = []
        num_levels = num_layers
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_features if i == 0 else hidden_size
            layers += [TemporalBlock(in_channels, hidden_size, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # TCN for text classification expects (batch_size, input_features, seq_length), hence transpose first
        x = x.transpose(1, 2)
        y = self.tcn(x)
        o = self.linear(y[:, :, -1])
        return self.log_softmax(o)