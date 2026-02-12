import torch
from torchvision import datasets, transforms
from model import Net
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = Net().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))

full_train = datasets.MNIST("./data", train=True, download=True, transform=transform)

train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size

train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model.eval()

import random
image, label = test_dataset[random.randint(0, len(test_dataset)-1)]

with torch.no_grad():
    image = image.unsqueeze(0).to(device)
    output = model(image)
    pred = output.argmax(1).item()

print("True:", label)
print("Pred:", pred)

def evaluate(loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return acc, precision, f1, cm


train_acc, train_prec, train_f1, train_cm = evaluate(train_loader)
val_acc, val_prec, val_f1, val_cm = evaluate(val_loader)
test_acc, test_prec, test_f1, test_cm = evaluate(test_loader)

print("\n====== METRICS ======")
print(f"Train -> Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, F1: {train_f1:.4f}")
print(f"Val   -> Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, F1: {val_f1:.4f}")
print(f"Test  -> Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, F1: {test_f1:.4f}")

plt.figure(figsize=(8,6))
sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Confusion Matrix")
plt.show()