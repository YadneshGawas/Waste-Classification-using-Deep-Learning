from ultralytics import YOLO

# Load a model pre-trained on ImageNet (e.g., YOLOv8)
model = YOLO("yolo11n.pt")  # Pretrained on ImageNet

# Train on your custom dataset
results = model.train(data=r"C:\Users\yadne\Desktop\24CP20\Resources\Models\yolo11n.pt", epochs=10, freeze=10)  # Freeze first 10 layers

print(results)