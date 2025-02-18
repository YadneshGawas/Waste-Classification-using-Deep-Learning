import torch
import json
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from pycocotools.coco import COCO
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Debugging function to check COCO dataset format
def check_coco_format(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        assert 'images' in data, "Missing 'images' field"
        assert 'annotations' in data, "Missing 'annotations' field"
        assert 'categories' in data, "Missing 'categories' field"
        print(f"COCO JSON {json_path} is correctly formatted.")
    except Exception as e:
        print(f"Error in COCO JSON {json_path}: {e}")
        exit(1)

# Custom dataset class for COCO format
class COCODataset(Dataset):
    def __init__(self, root, transforms=None):
        annotation_file = os.path.join(root, "_annotations.coco.json")
        self.root = root
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {'boxes': boxes, 'labels': labels}
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target
    
    def __len__(self):
        return len(self.img_ids)

# Paths
dataset_root = r"C:\Users\yadne\Desktop\24CP20\Resources\Models\RCNN_MODELS\COCO_SET2"
train_dir = os.path.join(dataset_root, "train")
val_dir = os.path.join(dataset_root, "valid")
test_dir = os.path.join(dataset_root, "test")

# Check dataset format
debug_dirs = [train_dir, val_dir, test_dir]
for dir in debug_dirs:
    check_coco_format(os.path.join(dir, "_annotations.coco.json"))

# Define transformations
transform = transforms.Compose([transforms.ToTensor()])

# Create datasets
dataset_test = COCODataset(test_dir, transforms=transform)
dataset_val = COCODataset(val_dir, transforms=transform)

# Data loaders
batch_size = 4  # <-- Change batch size here if needed (8 is recommended for RTX 4060)
train_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the model and dataset
model_path = r"C:\Users\yadne\Desktop\24CP20\Resources\Models\faster_rcnn_3760.pth"

# Load the model

# Define the model architecture (use the same model you trained with)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Adjust if using a custom model
num_classes = len(dataset_test.coco.cats) + 1  # Categories + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# Load the state dictionary
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, targets in test_loader:
        images = [img.to(device) for img in images]

        # Get model predictions
        outputs = model(images)
        
        for output, target in zip(outputs, targets):
            if 'labels' in output and 'labels' in target:
                pred_labels = output['labels'].detach().cpu().numpy().tolist()
                true_labels = target['labels'].detach().cpu().numpy().tolist()
                all_preds.extend(pred_labels)
                all_labels.extend(true_labels)

# Ensure valid predictions
if not all_labels or not all_preds:
    print("No valid predictions or labels found. Cannot compute metrics.")
else:
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
