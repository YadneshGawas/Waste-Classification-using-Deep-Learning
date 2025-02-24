import torch
import torchvision
import torchvision.transforms as T
import os
import json
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch import autocast

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
transform = T.Compose([T.ToTensor()])

# Create datasets
dataset_train = COCODataset(train_dir, transforms=transform)
dataset_val = COCODataset(val_dir, transforms=transform)

# Data loaders
batch_size = 4  # <-- Change batch size here if needed (8 is recommended for RTX 4060)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(dataset_train.coco.cats) + 1  # Categories + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", {device})
model.to(device)

# Optimizer & Mixed Precision
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scaler = torch.GradScaler("cuda")  # Enable AMP

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        print("Running")
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            loss_dict=model(images,targets)
            losses=sum(loss for loss in loss_dict.values())
            
        scaler.scale(losses).backward()  # Scale loss for stability
        scaler.step(optimizer)
        scaler.update()  # Update the scaler

        total_loss += losses.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

torch.save(model.state_dict(), "faster_rcnn_3760_10.pth")
print("Training complete!")
