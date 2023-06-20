import torch.nn as nn
import torch


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
