import os, torch, timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

train_img_dir = "/kaggle/input/rdd-2022/RDD_SPLIT/train/images"
train_label_dir = "/kaggle/input/rdd-2022/RDD_SPLIT/train/labels"
val_img_dir = "/kaggle/input/rdd-2022/RDD_SPLIT/val/images"
val_label_dir = "/kaggle/input/rdd-2022/RDD_SPLIT/val/labels"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
epochs = 50
lr = 1e-4
patience = 8

def load_yolo_labels(img_dir, label_dir):
    paths, labels = [], []
    for img_file in os.listdir(img_dir):
        if img_file.endswith(".jpg"):
            paths.append(os.path.join(img_dir, img_file))
            lbl_file = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))
            if os.path.exists(lbl_file):
                with open(lbl_file, "r") as f:
                    lines = f.readlines()
                    labels.append(int(lines[0].split()[0]) if lines else 0)
            else:
                labels.append(0)
    return paths, labels

train_paths, train_labels = load_yolo_labels(train_img_dir, train_label_dir)
val_paths, val_labels = load_yolo_labels(val_img_dir, val_label_dir)
num_classes = max(train_labels + val_labels) + 1

print(f"Dataset loaded {len(train_paths)} train, {len(val_paths)} val, {num_classes} classes")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self): 
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        lbl = self.labels[idx]
        if self.transform: 
            img = self.transform(img)
        return img, lbl

train_loader = DataLoader(CustomDataset(train_paths, train_labels, transform_train),
                          batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(CustomDataset(val_paths, val_labels, transform_val),
                        batch_size=batch_size, shuffle=False, num_workers=2)

class SwinWithDropout(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.fc(x)

model = SwinWithDropout(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

best_acc = 0.0
wait = 0
train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        _, preds = out.max(1)
        correct += preds.eq(lbls).sum().item()
        total += lbls.size(0)

    train_acc = 100 * correct / total
    train_loss /= total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            val_loss += loss.item() * imgs.size(0)
            _, preds = out.max(1)
            val_correct += preds.eq(lbls).sum().item()
            val_total += lbls.size(0)

    val_acc = 100 * val_correct / val_total
    val_loss /= val_total
    scheduler.step(val_acc)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "/kaggle/working/best_swin_transformer_full.pth")
        print("Best model saved.")
        wait = 0
    else:
        wait += 1
        print(f"No improvement for {wait} epoch(s)")
        if wait >= patience:
            print("Early stopping triggered.")
            break

print(f"Training complete : Best Validation Accuracy: {best_acc:.2f}%")

history = pd.DataFrame({
    "Epoch": range(1, len(train_acc_list)+1),
    "Train_Acc": train_acc_list,
    "Val_Acc": val_acc_list,
    "Train_Loss": train_loss_list,
    "Val_Loss": val_loss_list
})
history.to_csv("/kaggle/working/swin_training_historyfull.csv", index=False)
print("Training history saved to /kaggle/working/swin_training_historyfull.csv")
