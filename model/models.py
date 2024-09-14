import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FxRNN(nn.Module):
    def __init__(self, n_classes, input_size=87, hidden_size=128, num_layers=2):
        super(FxRNN, self).__init__()
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out = nn.Linear(60, self.n_classes)

    def forward(self, t):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, t.size(0), self.hidden_size).to(t.device)
        c0 = torch.zeros(self.num_layers, t.size(0), self.hidden_size).to(t.device)

        # Pass through LSTM
        out, _ = self.lstm(t, (h0, c0))

        # Pass the last hidden state to fully connected layers
        t = F.relu(self.fc1(out[:, -1, :]))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t



class SettingsRNN(nn.Module):

    def __init__(self, n_settings, input_size, hidden_size, num_layers=2):
        super(SettingsRNN, self).__init__()
        self.n_settings = n_settings
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 60)
        self.out = nn.Linear(60, self.n_settings)

    def forward(self, t):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, t.size(0), self.hidden_size).to(t.device)
        c0 = torch.zeros(self.num_layers, t.size(0), self.hidden_size).to(t.device)

        # Pass through LSTM
        out, _ = self.lstm(t, (h0, c0))

        # Pass the last hidden state to fully connected layers
        t = F.relu(self.fc1(out[:, -1, :]))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        t = F.tanh(t)

        return t


class MultiRNN(nn.Module):

    def __init__(self, n_classes, n_settings, input_size, hidden_size, num_layers=2):
        super(MultiRNN, self).__init__()
        self.n_classes = n_classes
        self.n_settings = n_settings
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Common LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fx network
        self.fc1_a = nn.Linear(hidden_size, 120)
        self.fc2_a = nn.Linear(120, 60)
        self.out_a = nn.Linear(60, self.n_classes)

        # Settings network
        self.fc1_b = nn.Linear(hidden_size, 120)
        self.fc2_b = nn.Linear(120, 60)
        self.out_b = nn.Linear(60, self.n_settings)

    def forward(self, t):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, t.size(0), self.hidden_size).to(t.device)
        c0 = torch.zeros(self.num_layers, t.size(0), self.hidden_size).to(t.device)

        # Pass through LSTM
        out, _ = self.lstm(t, (h0, c0))

        # Fx path
        t_a = F.relu(self.fc1_a(out[:, -1, :]))
        t_a = F.relu(self.fc2_a(t_a))
        t_a = self.out_a(t_a)

        # Settings path
        t_b = F.relu(self.fc1_b(out[:, -1, :]))
        t_b = F.relu(self.fc2_b(t_b))
        t_b = self.out_b(t_b)
        t_b = F.tanh(t_b)

        return t_a, t_b


class SettingsNetCondRNN(nn.Module):
    def __init__(self, n_settings, mel_shape, num_embeddings, embedding_dim=50, rnn_hidden_size=128, rnn_layers=2):
        super().__init__()
        self.n_settings = n_settings
        self.mel_shape = mel_shape
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers

        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.fc0 = nn.Linear(in_features=embedding_dim, out_features=self.mel_shape[0] * self.mel_shape[1])

        # Adjust input_size to match the actual input size after concatenation
        self.lstm = nn.LSTM(input_size=88,  # Adjusted input_size to match the actual input
                            hidden_size=self.rnn_hidden_size,
                            num_layers=self.rnn_layers,
                            batch_first=True,
                            bidirectional=True)

        self.fc1 = nn.Linear(in_features=self.rnn_hidden_size * 2 * self.mel_shape[0], out_features=120)
        self.batchNorm3 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.batchNorm4 = nn.BatchNorm1d(num_features=60)
        self.out = nn.Linear(in_features=60, out_features=self.n_settings)

    def forward(self, t, c):
        # (0.1) embedding layer
        c = self.emb(c)  # c shape: [batch_size, embedding_dim]

        # (0.2) dense layer to match the mel spectrogram shape
        c = self.fc0(c)  # c shape: [batch_size, mel_shape[0] * mel_shape[1]]

        # Reshape c to match the dimensions needed for concatenation with t
        c = c.view(t.size(0), 1, self.mel_shape[0], self.mel_shape[1])  # c shape: [batch_size, 1, 128, 87]

        # Expand dimensions to match t
        c = c.unsqueeze(1)  # c shape: [batch_size, 1, 1, 128, 87]

        # Permute `c` to match `t`'s dimensions for correct alignment
        c = c.permute(0, 1, 3, 2, 4)  # c shape: [batch_size, 1, 128, 1, 87]

        # (1) Now, `t` and `c` should have matching shapes
        #print(t.shape)  # Debugging: [batch_size, 1, 128, 1, 87]
        #print(c.shape)  # Debugging: [batch_size, 1, 128, 1, 87]

        # (2) Concatenate along the last dimension
        t = torch.cat((t, c), dim=-1)  # Concatenate along the last dimension, resulting in [batch_size, 1, 128, 1, 174]

        # (3) Reshape t for the RNN input
        t = t.view(t.size(0), t.size(2), -1)  # Shape: [batch_size, time_steps, input_size]

        # (4) Apply LSTM
        t, _ = self.lstm(t)  # Shape after LSTM: [batch_size, sequence_length, 2 * rnn_hidden_size]

        # (5) Flatten LSTM output for fully connected layers
        t = t.contiguous().view(t.size(0), -1)  # Shape: [batch_size, 2 * rnn_hidden_size * sequence_length]

        # (6) Hidden dense layers
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        # (7) Output layer
        t = self.out(t)
        t = torch.tanh(t)

        return t





