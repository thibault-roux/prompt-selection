import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle


# 1. Dataset Class
class SentenceVectorDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len, split="train"):
        file_map = {
            "train": "stsb_multi_mt_fr_train.pkl",
            "dev": "stsb_multi_mt_fr_dev.pkl",
            "test": "stsb_multi_mt_fr_test.pkl",
        }
        file_path = f"{data_path}/{file_map[split]}"
        
        # Load the data
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Combine sentences and embeddings
        self.sentences = data["sentence1"] + data["sentence2"]
        self.embeddings = data["sentence1_emb"] + data["sentence2_emb"]
        
        # Tokenizer and padding
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        embedding = self.embeddings[idx]
        
        # Tokenize and pad the sentence
        tokenized = self.tokenizer.texts_to_sequences([sentence])[0]
        padded = self._pad_sequence(tokenized, self.max_seq_len)
        
        return torch.tensor(padded, dtype=torch.long), torch.tensor(embedding, dtype=torch.float32)
    
    def _pad_sequence(self, sequence, max_len):
        return sequence[:max_len] + [0] * (max_len - len(sequence))


# Tokenizer
class SimpleTokenizer:
    def __init__(self, captions):
        self.word2idx = {}
        self.idx2word = {}
        self.build_vocab(captions)
    
    def build_vocab(self, captions):
        all_words = set(" ".join(captions).split())
        self.word2idx = {word: idx + 1 for idx, word in enumerate(all_words)}
        self.word2idx["<PAD>"] = 0
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def texts_to_sequences(self, captions):
        return [[self.word2idx.get(word, 0) for word in caption.split()] for caption in captions]


# 2. Define CNN Encoder for Grayscale Images
class EncoderCNN(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(EncoderCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, encoded_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, images):
        x = self.relu(self.fc1(images))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x


# 3. Define RNN Decoder
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, encoded_dim, max_caption_length):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(encoded_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.max_caption_length = max_caption_length
    
    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        features = features.unsqueeze(1)  # Add sequence dimension
        inputs = torch.cat((features, embeddings), dim=1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc2(lstm_out)
        return outputs


# 4. Training Loop
def train_model(encoder, decoder, dataloader, criterion, optimizer, vocab_size, device, num_epochs=10):
    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for captions, images in dataloader:
            captions = captions.to(device)
            images = images.to(device)

            # Prepare shifted captions
            captions_in = captions[:, :-1]  # Input captions
            captions_out = captions[:, 1:]  # Target captions
            
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions_in)
            
            # Compute loss
            outputs = outputs[:, :-1, :].contiguous()  # Align dimensions
            loss = criterion(outputs.view(-1, vocab_size), captions_out.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    # Example Usage
    data_path = "./data"

    # Build tokenizer from training sentences
    with open(f"{data_path}/stsb_multi_mt_fr_train.pkl", "rb") as f:
        train_data = pickle.load(f)

    all_sentences = train_data["sentence1"] + train_data["sentence2"]
    tokenizer = SimpleTokenizer(all_sentences)
    vocab_size = len(tokenizer.word2idx)

    # Hyperparameters
    max_seq_len = 50
    batch_size = 32
    num_epochs = 10
    input_dim = 1024  # Grayscale images are 32x32 flattened to 1024
    encoded_dim = 256
    embed_size = 256
    hidden_size = 512
    learning_rate = 0.001

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and data loaders
    train_dataset = SentenceVectorDataset(data_path, tokenizer, max_seq_len, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Models
    encoder = EncoderCNN(input_dim, encoded_dim)
    decoder = DecoderRNN(vocab_size, embed_size, hidden_size, encoded_dim, max_seq_len)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Train the model
    train_model(encoder, decoder, train_loader, criterion, optimizer, vocab_size, device, num_epochs)
