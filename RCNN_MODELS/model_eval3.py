import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import os

# Set device (cuda if available, otherwise cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the model and dataset
model_path = '/content/drive/MyDrive/your_model.pth'  # Change to your model's path
dataset_path = '/content/drive/MyDrive/test_dataset'  # Change to your dataset's path

# Load the model
model = torch.load(model_path)  # Make sure to load the model correctly (check if you need model.eval())
model.to(device)  # Move the model to the selected device

# Load the test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Example size; modify as per your model's requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
])

test_dataset = datasets.ImageFolder(dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Adjust batch size as needed

# Set model to evaluation mode
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        if batch is None or len(batch) < 2:
            continue  # Skip empty or incorrectly formatted batches

        images, targets = batch  # Extract images and targets
        images = images.to(device)  # Move images to device

        # Get model predictions
        outputs = model(images)

        # Ensure outputs are valid
        if outputs is None or len(outputs) == 0:
            continue  # Skip if no predictions

        # Assuming output is a tensor of class scores; modify if necessary
        _, predicted_labels = torch.max(outputs, 1)  # Get the predicted class labels

        # Convert to CPU and numpy arrays
        pred_labels = predicted_labels.detach().cpu().numpy().tolist()
        true_labels = targets.detach().cpu().numpy().tolist()

        all_preds.extend(pred_labels)
        all_labels.extend(true_labels)

# Ensure lists are not empty before computing metrics
if not all_labels or not all_preds:
    print("No valid predictions or labels found. Cannot compute metrics.")
else:
    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
