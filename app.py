import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

# Define a simple dataset for training
class SimpleTextDataset(Dataset):
    def __init__(self, text, seq_length=32):
        self.text = text
        self.seq_length = seq_length
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        # Convert text to integer indices
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx+self.seq_length]
        target_seq = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(input_seq), torch.tensor(target_seq)

# Transformer Model
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embed_size))  # Max sequence length of 1000
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        output = self.transformer(x, x)  # Decoder input is the same as encoder
        logits = self.fc_out(output)
        return logits

# Training the Model
def train_model(model, dataset, epochs=10, batch_size=32, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_seq, target_seq in dataloader:
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output.view(-1, len(dataset.vocab)), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Text generation function
def generate_text(model, start_text, max_len=100, temperature=1.0):
    model.eval()
    input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in start_text]).unsqueeze(0)  # (1, len)
    generated = start_text

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_seq)
            logits = logits[:, -1, :] / temperature  # Use only last timestep
            probs = torch.softmax(logits, dim=-1)
            predicted_idx = torch.multinomial(probs, 1)
            predicted_char = dataset.idx_to_char[predicted_idx.item()]
            generated += predicted_char
            try:
                input_seq = torch.cat([input_seq[:, 1:], predicted_idx], dim=1)
            except Exception as e:
                print("exception" + str(e))

    return generated

# Example usage
text_data = "The quick brown fox jumps over the lazy dog. " * 100  # Example text data
dataset = SimpleTextDataset(text_data, seq_length=32)

# Initialize the model
model = TransformerLM(vocab_size=dataset.vocab_size, embed_size=64, num_heads=4, num_layers=2, hidden_size=256)

# Train the model
print("Start")
train_model(model, dataset, epochs=5, batch_size=16)
print("train_model done")

# Generate text
generated_text = generate_text(model, "The quick brown")
print("generate_text done")
print(generated_text)
print("End")
