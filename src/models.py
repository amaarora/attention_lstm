import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output = [batch size, seq_len, hidden_dim]
        attention_scores = self.attn(lstm_output)
        # attention_scores = [batch size, seq_len, 1]
        attention_scores = attention_scores.squeeze(2)
        # attention_scores = [batch size, seq_len]
        return F.softmax(attention_scores, dim=1)


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        tokenizer,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=3,
        n_layers=2,
        bidirectional=True,
        dropout=0.1,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(len(tokenizer.get_vocab()), embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        # Dense layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, text):
        # text = [batch size, sent_length]
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        # hidden = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs


class SentimentClassifierWithSoftAttention(nn.Module):
    def __init__(
        self,
        tokenizer,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=3,
        n_layers=2,
        bidirectional=True,
        dropout=0.1,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(len(tokenizer.get_vocab()), embedding_dim)

        # attention
        self.attention = Attention(hidden_dim * 2 if bidirectional else hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        # Dense layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, text):
        # text = [batch size, sent_length]
        embedded = self.embedding(text)
        lstm_output, (hidden, _) = self.lstm(embedded)
        # lstm_output = [batch size, seq_len, hidden_dim*num_directions]
        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        attention_weights = self.attention(lstm_output)
        # attention_weights = [batch size, seq_len]
        attention_weights = attention_weights.unsqueeze(2)
        weighted = lstm_output * attention_weights
        # weighted = [batch size, seq_len, hidden_dim]

        weighted_sum = weighted.sum(dim=1)
        # weighted_sum = [batch size, hidden_dim]

        dense_outputs = self.fc(weighted_sum)
        # dense_outputs = [batch size, output_dim]

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs
