import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import IngredientDataset
from model import get_model
from tqdm import tqdm
from utils import get_predictions

# CONFIG
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# DATASET
dataset = IngredientDataset("labels.csv", "train", transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

# LOSS + OPTIMIZER
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# TRAIN LOOP
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(dataloader)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = get_predictions(outputs)

        correct += (preds == labels.int()).sum().item()
        total += labels.numel()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader)
    accuracy = correct / total

    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "model.pth")

# SAVE MODEL
print("Model saved!")