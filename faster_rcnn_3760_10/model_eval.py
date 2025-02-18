import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import Dataset

# ---- Load Dataset ----
class COCODataset(Dataset):
    def __init__(self, root, transforms=None, use_annotations=True):
        self.root = root
        self.transforms = transforms
        self.use_annotations = use_annotations

        if use_annotations:
            annotation_file = os.path.join(root, "_annotations.coco.json")
            self.coco = COCO(annotation_file)
            self.img_ids = list(self.coco.imgs.keys())
        else:
            # Load images directly if no annotations are used
            self.img_ids = [f for f in os.listdir(root) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __getitem__(self, index):
        if self.use_annotations:
            img_id = self.img_ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.root, img_info['file_name'])
        else:
            img_path = os.path.join(self.root, self.img_ids[index])

        image = Image.open(img_path).convert("RGB")

        if self.use_annotations:
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {'boxes': boxes, 'labels': labels}
        else:
            target = {}  # Empty target if no annotations are used

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.img_ids)


# Dataset paths
dataset_root = r"C:\Users\yadne\Desktop\24CP20\Resources\Models\RCNN_MODELS\COCO_SET"
val_dir = os.path.join(dataset_root, "test")  # Adjust if needed

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load validation dataset
val_dataset = COCODataset(root=val_dir, transforms=transform, use_annotations=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Set device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of classes (from your trained model)
num_classes = len(val_dataset.coco.cats) + 1

# Load the model with the correct number of classes
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

# Modify the box predictor to match trained model's classes
in_features = model.roi_heads.box_predictor.cls_score.in_features  # Get input features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load(r"C:\Users\yadne\Desktop\24CP20\Resources\Models\faster_rcnn_3760_5.pth", map_location=device))

# Move model to device
model.to(device)
model.eval()  # Set to evaluation mode

print("Model loaded successfully with correct class count!")


# ---- Evaluation ----
all_preds = []
all_labels = []

with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]

        # Get model predictions
        outputs = model(images)

        for output, target in zip(outputs, targets):
            if 'labels' in output and 'labels' in target and len(output['labels']) > 0:
                pred_labels = output['labels'].detach().cpu().numpy().tolist()
                true_labels = target['labels'].detach().cpu().numpy().tolist()

                min_length = min(len(pred_labels), len(true_labels))
                all_preds.extend(pred_labels[:min_length])
                all_labels.extend(true_labels[:min_length])

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
