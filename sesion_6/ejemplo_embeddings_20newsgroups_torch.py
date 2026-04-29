# Ejemplo: Clasificación de textos con embeddings semánticos y 20 Newsgroups usando PyTorch
# -------------------------------------------------------------------------
# Este script muestra cómo usar un modelo de HuggingFace (MiniLM) en PyTorch
# para obtener embeddings semánticos y entrenar un clasificador simple.
# -------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report
import numpy as np

# 1. Cargar el dataset 20 Newsgroups
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# 2. Tokenizar y obtener embeddings con un modelo de HuggingFace
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# Función para obtener el embedding promedio de cada texto
def get_embeddings(texts, batch_size=32):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=128)
            outputs = model(**encoded)
            # Usar el embedding [CLS] o el promedio de los embeddings de las palabras
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

X_train = get_embeddings(train.data)
X_test = get_embeddings(test.data)
y_train = train.target
y_test = test.target

# 3. Clasificador simple en PyTorch (una capa lineal)
import torch.nn as nn
import torch.optim as optim

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
num_classes = len(train.target_names)
clf = SimpleClassifier(input_dim, num_classes)

# Entrenamiento
EPOCHS = 10
BATCH_SIZE = 64
optimizer = optim.Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

for epoch in range(EPOCHS):
    clf.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    for i in range(0, X_train_tensor.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        optimizer.zero_grad()
        outputs = clf(batch_x)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# 4. Evaluación
clf.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    logits = clf(X_test_tensor)
    y_pred = logits.argmax(dim=1).cpu().numpy()
print(classification_report(y_test, y_pred, target_names=test.target_names))
