import torch
from torchvision import datasets, transforms
from model import Net


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

model = Net().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

import random
image, label = test_dataset[random.randint(0, len(test_dataset)-1)]

with torch.no_grad():
    image = image.unsqueeze(0).to(device)
    output = model(image)
    pred = output.argmax(1).item()

print("True:", label)
print("Pred:", pred)