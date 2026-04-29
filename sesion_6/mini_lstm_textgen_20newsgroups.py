# Mini modelo generativo de texto (palabra a palabra) con PyTorch y 20 Newsgroups
# -------------------------------------------------------------------------
# Entrena un LSTM pequeño para predecir la siguiente palabra en secuencias de texto.
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import numpy as np
import random

# 1. Cargar y preprocesar el corpus
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
texts = train.data

# Tokenización simple (split por espacio, minúsculas)
tokens = []
for text in texts:
    tokens.extend(text.lower().split())

# Limitar vocabulario a las 3000 palabras más frecuentes
vocab_size = 3000
most_common = Counter(tokens).most_common(vocab_size-2)
itos = ['<PAD>', '<UNK>'] + [w for w, _ in most_common]
stoi = {w: i for i, w in enumerate(itos)}

def encode(word):
    return stoi.get(word, stoi['<UNK>'])
def decode(idx):
    return itos[idx] if idx < len(itos) else '<UNK>'

tokens_encoded = [encode(w) for w in tokens]

# 2. Crear secuencias de entrada/salida (contexto de 5 palabras -> siguiente palabra)
SEQ_LEN = 5
inputs = []
targets = []
for i in range(len(tokens_encoded) - SEQ_LEN):
    inputs.append(tokens_encoded[i:i+SEQ_LEN])
    targets.append(tokens_encoded[i+SEQ_LEN])
inputs = np.array(inputs)
targets = np.array(targets)

# 3. Dataset y DataLoader
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(inputs, targets)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# 4. Modelo LSTM pequeño
torch.manual_seed(42)
class MiniLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

model = MiniLSTM(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 5. Entrenamiento (pocas épocas para ejemplo)
for epoch in range(3):
    total_loss = 0
    for Xb, yb in dataloader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 6. Generación de texto a partir de un prompt
def generate_text(prompt, length=20):
    model.eval()
    words = prompt.lower().split()
    context = [encode(w) for w in words[-SEQ_LEN:]]
    context = [stoi['<PAD>']] * (SEQ_LEN - len(context)) + context
    generated = words.copy()
    for _ in range(length):
        x = torch.tensor([context], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
            next_idx = torch.argmax(logits, dim=1).item()
        next_word = decode(next_idx)
        generated.append(next_word)
        context = context[1:] + [next_idx]
    return ' '.join(generated)

# Ejemplo de uso:
prompt = "this is a simple"
print("Prompt:", prompt)
print("Generated:", generate_text(prompt, length=20))
