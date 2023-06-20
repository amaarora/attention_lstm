import torch.optim as optim
from model import SentimentClassifier
import torch.nn as nn
from dataset import TweetDataset
from torch.utils.data import DataLoader
import torch, wandb
from tqdm import tqdm
from omegaconf import OmegaConf


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        # Move tensors to GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids)
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()
        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()
    return epoch_loss / len(data_loader)


def evaluate_one_epoch(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            # Move tensors to GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids)
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()

            # Compute loss
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss / len(data_loader), epoch_accuracy / len(data_loader)


def main():
    cfg = OmegaConf.load("./config.yml")
    run = wandb.init(
        project="bilstm-attention", job_type="train", save_code=True, config=cfg
    )
    # Creating instances of training and validation dataset
    train_set = TweetDataset(filename="../data/train.csv", maxlen=48)
    val_set = TweetDataset(filename="../data/test.csv", maxlen=48)

    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=4)

    # Instantiate the model
    model = SentimentClassifier(
        train_set.tokenizer,
        cfg.embedding_dim,
        cfg.hidden_dim,
        cfg.output_dim,
        cfg.n_layers,
        cfg.bidirectional,
        cfg.dropout,
    )
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Move model and loss function to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    # Training loop
    for epoch in range(cfg.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate_one_epoch(
            model, val_loader, criterion, device
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch: {epoch+1} | Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | Val accuracy: {val_accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
