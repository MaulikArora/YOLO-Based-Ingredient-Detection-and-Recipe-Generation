import torch
from PIL import Image
from torchvision import transforms
from model import get_model
from utils import get_predictions
import pandas as pd

import requests

def get_recipe(ingredients):
    prompt = f"""
    Create a simple recipe using these ingredients: {', '.join(ingredients)}.
    Give:
    - Dish name
    - Ingredients list
    - Steps (short)
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

# LOAD CLASSES
df = pd.read_csv("labels.csv")
classes = df.columns[1:]

# LOAD MODEL
model = get_model()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        preds = get_predictions(outputs)[0]

        probs = torch.sigmoid(outputs)[0]

        for i, p in enumerate(probs):
            print(f"{classes[i]}: {p:.2f}")

    detected = [classes[i] for i in range(len(classes)) if preds[i] == 1]
    return detected


# TEST
if __name__ == "__main__":
    img_path = "C:/Users/LENOVO/Downloads/Harvest feast on a wooden board.png"
    result = predict(img_path)
    recipe = get_recipe(result)
    print("Detected:", result)
    print("\nRecipe:\n", recipe)