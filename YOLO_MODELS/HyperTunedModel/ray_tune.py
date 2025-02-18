from ray import tune
from ray.tune import Tuner, TuneConfig
from ultralytics import YOLO
import torch

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 1️⃣ Function to fine-tune YOLO
def fine_tune_yolo(config):
    """Fine-tune a trained YOLO model using different hyperparameters."""
    model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Datasets\Train3\runs\detect\train3\weights\best.pt")  

    model.train(
        data=r"C:\Users\yadne\Desktop\24CP20\Resources\Datasets\Train3\data.yaml",
        epochs=5,
        imgsz=640,
        lr0=config["lr0"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        optimizer="AdamW",
        device=device,
        resume=False  # Ensure training starts fresh
    )

# 2️⃣ Define hyperparameter search space
search_space = {
    "lr0": tune.loguniform(1e-5, 1e-2),
    "momentum": tune.uniform(0.6, 0.98),
    "weight_decay": tune.loguniform(1e-4, 1e-2)
}

# 3️⃣ Run Ray Tune for hyperparameter optimization
tuner = Tuner(
    fine_tune_yolo,
    param_space=search_space,
    tune_config=TuneConfig(
        metric="metrics/val_loss",  # Ensure correct metric
        mode="min",
        num_samples=3
    ),
)

print("🔄 Running hyperparameter tuning...")
results = tuner.fit()

# 4️⃣ Get the best hyperparameters
best_result = results.get_best_result(metric="metrics/val_loss", mode="min")
best_hyperparameters = best_result.config

print("✅ Best Hyperparameters Found:", best_hyperparameters)

# 5️⃣ Fine-tune with best hyperparameters
print("🔄 Retraining with best hyperparameters...")
model = YOLO(r"C:\Users\yadne\Desktop\24CP20\Resources\Datasets\Train3\runs\detect\train3\weights\best.pt")  

model.train(
    data=r"C:\Users\yadne\Desktop\24CP20\Resources\Datasets\Train3\data.yaml",
    epochs=10,
    imgsz=640,
    lr0=best_hyperparameters["lr0"],
    momentum=best_hyperparameters["momentum"],
    weight_decay=best_hyperparameters["weight_decay"],
    device=device,
    resume=False
)

print("🎯 Fine-tuning completed! Model is optimized.")
