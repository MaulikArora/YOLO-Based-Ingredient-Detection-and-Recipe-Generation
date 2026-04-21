from ultralytics import YOLO

model = YOLO("runs/detect/train6/weights/best.pt")

# Run detection on your new downloaded image
results = model.predict(
    source="C:/Visual Studio Code/ingredai/test_images/test_food2.png",
    imgsz=640,     # match training image size
    conf=0.1,      # adjust confidence threshold as needed
    save=True,     # saves output with bounding boxes
    show=True      # displays result window (optional)
)

print(f"Detections saved at: {results[0].save_dir}")
