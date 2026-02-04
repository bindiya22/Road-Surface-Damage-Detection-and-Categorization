import os
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class CFG:
    dataset_dir = "/kaggle/input/rdd-2022/RDD_SPLIT"
    work_dir = "/kaggle/working/rdd2022-class"
    img_size = 224
    batch_size = 32
    epochs = 50
    lr = 1e-4
    patience = 7
    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_map = {
        0: "longitudinal_crack",
        1: "transverse_crack",
        2: "alligator_crack",
        3: "other_damage",
        4: "pothole"
    }

print(f"Using device: {CFG.device}")

def create_classification_folders(root, split):
    img_dir = os.path.join(root, split, "images")
    lbl_dir = os.path.join(root, split, "labels")
    new_root = os.path.join(CFG.work_dir, split)
    os.makedirs(new_root, exist_ok=True)
    print(f"Reorganizing {split} set...")
    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            if not lines:
                continue
            cls_id = int(lines[0].split()[0])
        cls_name = CFG.class_map.get(cls_id, "unknown")
        cls_folder = os.path.join(new_root, cls_name)
        os.makedirs(cls_folder, exist_ok=True)
        shutil.copy(img_path, os.path.join(cls_folder, img_name))
    print(f"{split} set ready at {new_root}")

create_classification_folders(CFG.dataset_dir, "train")
create_classification_folders(CFG.dataset_dir, "val")

train_transforms = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.RandomResizedCrop(CFG.img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(CFG.work_dir, "train"), transform=train_transforms)
val_ds = datasets.ImageFolder(os.path.join(CFG.work_dir, "val"), transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

print(f"Dataset Loaded | Classes: {train_ds.classes}")

model = timm.create_model("maxvit_tiny_tf_224.in1k", pretrained=True, num_classes=len(train_ds.classes))
model = model.to(CFG.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
scaler = torch.cuda.amp.GradScaler()

def train_one_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc="Train", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total

def validate():
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Valid", leave=False):
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = 100. * correct / total
    return total_loss / total, acc, preds_all, labels_all

best_acc = 0
patience_counter = 0

for epoch in range(CFG.epochs):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc, preds, labels = validate()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{CFG.epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_maxvit_rdd2022.pth")
        print(f"New best model saved (Val Acc: {best_acc:.2f}%)")
    else:
        patience_counter += 1
    if patience_counter >= CFG.patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

print("Training Complete.")
print(f"Best Validation Accuracy: {best_acc:.2f}%")

model.load_state_dict(torch.load("best_maxvit_rdd2022.pth"))
model.eval()
_, final_acc, preds, labels = validate()
print(f"Final Validation Accuracy: {final_acc:.2f}%")
print("Classification Report:")
print(classification_report(labels, preds, target_names=train_ds.classes))
print("Confusion Matrix:")
print(confusion_matrix(labels, preds))
