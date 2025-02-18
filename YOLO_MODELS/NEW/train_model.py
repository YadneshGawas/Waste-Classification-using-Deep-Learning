if __name__ == "__main__":
    from ultralytics import YOLO
    import torch

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = 0  # Use GPU (device index 0)
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"  # Use CPU
        print("CUDA not available. Using CPU.")

    # Load the model (replace 'yolo11n.pt' with your model)
    model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\NEW\runs\detect\train3\weights\best.pt")

    # Train the model
    results = model.train(data=r"C:\Users\yadne\Desktop\24CP20\Resources\Models\YOLO_MODELS\NEW\dataset\data.yaml", epochs=20, imgsz=640, device=device)
