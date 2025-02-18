import torch
import torchvision
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor

# 🟢 1️⃣ Custom Dataset Class for COCO Format
class COCODataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transforms = transforms or ToTensor()  # Ensure conversion to tensor

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        image = self.transforms(image)  # Convert PIL Image to PyTorch tensor

        return image, img_id

    def __len__(self):
        return len(self.img_ids)


# 🟢 2️⃣ Paths
dataset_root = r"C:\Users\yadne\Desktop\24CP20\Resources\Models\RCNN_MODELS\COCO_SET"
val_dir = os.path.join(dataset_root, "valid")
test_dir = os.path.join(dataset_root, "test")

val_json = os.path.join(val_dir, "_annotations.coco.json")
test_json = os.path.join(test_dir, "_annotations.coco.json")

model_path = r"C:\Users\yadne\Desktop\24CP20\Resources\Models\RCNN_MODELS\faster_rcnn_custom.pth"  # Update with your trained .pth model

# 🟢 3️⃣ Define DataLoader
transform = ToTensor()
val_dataset = COCODataset(val_dir, val_json, transforms=transform)
test_dataset = COCODataset(test_dir, test_json, transforms=transform)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 🟢 4️⃣ Load Trained Faster R-CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = len(val_dataset.coco.cats) + 1

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# 🟢 5️⃣ Function to Evaluate Using COCO API
def evaluate_coco(model, data_loader, annotation_json, result_json):
    results = []  # Store results

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for images, img_ids in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img_id, output in zip(img_ids, outputs):
                boxes = output['boxes'].detach().cpu().numpy()
                scores = output['scores'].detach().cpu().numpy()
                labels = output['labels'].detach().cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    result = {
                        "image_id": int(img_id),
                        "category_id": int(label),
                        "bbox": [float(x) for x in box],  # Convert bbox to Python float
                        "score": float(score)  # Convert score to Python float
                    }
                    results.append(result)

    # Save results to a separate JSON file
    with open(result_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved evaluation results to {result_json}")

    # 🟢 Run COCO Evaluation
    coco_gt = COCO(annotation_json)  # Load ground truth
    coco_dt = coco_gt.loadRes(result_json)  # Load model predictions

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return results


# 🟢 6️⃣ Run Evaluation with COCO Metrics
print("\n--- Evaluating on Validation Set ---")
val_results_json = "val_results.json"  # Save validation results separately
evaluate_coco(model, val_loader, val_json, val_results_json)

print("\n--- Evaluating on Test Set ---")
test_results_json = "test_results.json"  # Save test results separately
evaluate_coco(model, test_loader, test_json, test_results_json)
