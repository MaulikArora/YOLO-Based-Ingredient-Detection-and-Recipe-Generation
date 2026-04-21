import os
os.environ["GGML_CUDA"] = "0"

from ultralytics import YOLO
from gpt4all import GPT4All
import random, json

MODEL_PATH = "runs/detect/train6/weights/best.pt"
TEST_IMAGES_PATH = r"C:/Visual Studio Code/ingredai/valid/images"
CONF_THRESHOLD = 0.1
LLM_MODEL_DIR = r"C:/Users/LENOVO/AppData/Local/nomic.ai/GPT4All"
LLM_MODEL_NAME = "Phi-3-mini-4k-instruct.Q4_0.gguf"

sample_image = random.choice(os.listdir(TEST_IMAGES_PATH))
image_path = os.path.join(TEST_IMAGES_PATH, sample_image)
print(f"\nSelected image: {sample_image}\n")

model = YOLO(MODEL_PATH)
results = model.predict(source="C:/Visual Studio Code/ingredai/test_images/test_food4.png", imgsz=960, conf=CONF_THRESHOLD, save=True, show=False)
pred = results[0]

detected = []
if len(pred.boxes) > 0:
    for box in pred.boxes:
        cls_id = int(box.cls)
        conf = round(float(box.conf), 2)
        detected.append((model.names[cls_id], conf))

if detected:
    print("Detected ingredients:")
    for name, conf in detected:
        print(f"   - {name} ({conf*100:.1f}% confidence)")
else:
    print("No ingredients detected (or all below confidence threshold).")

if detected:
    ingredients = ", ".join([d[0] for d in detected])
    prompt = (
        f"You are a professional chef AI. Using these ingredients: {ingredients}, "
        f"suggest 3 realistic recipes. "
        f"Each recipe should include a title, short description, and key ingredients. "
        f"Keep it concise and human-sounding."
    )
else:
    prompt = (
        "No clear ingredients detected. Suggest 3 simple vegetarian dishes commonly seen in kitchens."
    )

try:
    print("\nGenerating recipes with Phi-3 Instruct (CPU mode)...\n")
    llm = GPT4All(
        model_name=LLM_MODEL_NAME,
        model_path=LLM_MODEL_DIR,
        allow_download=False
    )
    response = llm.generate(prompt, max_tokens=300, temp=0.7)
    print("GPT4All's response:\n")
    print(response.strip())
except Exception as e:
    print("Error running GPT4All:", e)
    print(f"\nPrompt that would have been sent:\n{prompt}\n")

print("\nResults saved in:", results[0].save_dir)
