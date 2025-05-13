import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, cohen_kappa_score
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model
import matplotlib.pyplot as plt

# Local paths
train_csv_path = r"C:/Users/MANIDEEP/Downloads/deep learning/training.csv"
val_csv_path = r"C:/Users/MANIDEEP/Downloads/deep learning/validation.csv"
train_images_path = r"C:/Users/MANIDEEP/Downloads/deep learning/train_images"
preprocessed_train_images_path = r"C:/Users/MANIDEEP/Downloads/deep learning/processed_images/train"
preprocessed_val_images_path = r"C:/Users/MANIDEEP/Downloads/deep learning/processed_images/validation"

# Create folders if not exist
os.makedirs(preprocessed_train_images_path, exist_ok=True)
os.makedirs(preprocessed_val_images_path, exist_ok=True)

# Load CSV files
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

print(f"Training data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")

# Image preprocessing
def preprocess_image(image_path, save_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    gray_img_clahe = clahe.apply(gray_img)
    img_clahe = np.zeros_like(img)
    for i in range(3):
        img_clahe[..., i] = clahe.apply(img[..., i])
    img_final = cv2.addWeighted(img_clahe, 4, cv2.GaussianBlur(img_clahe, (0, 0), 30), -4, 128)
    cv2.imwrite(save_path, img_final)
    return img, gray_img_clahe, img_final

# Dataset class
class APTOSDataset(Dataset):
    def __init__(self, dataframe, images_path, preprocessed_images_path, transform=None):
        self.dataframe = dataframe
        self.images_path = images_path
        self.preprocessed_images_path = preprocessed_images_path
        self.transform = transform

        for idx in range(len(self.dataframe)):
            img_name = os.path.join(self.images_path, self.dataframe.iloc[idx, 0] + '.png')
            save_path = os.path.join(self.preprocessed_images_path, self.dataframe.iloc[idx, 0] + '.png')
            if not os.path.exists(save_path):
                preprocess_image(img_name, save_path)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.preprocessed_images_path, self.dataframe.iloc[idx, 0] + '.png')
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.dataframe.iloc[idx, 1]
        return image, label

# Data transforms
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = APTOSDataset(train_df, train_images_path, preprocessed_train_images_path, transform=data_transforms)
val_dataset = APTOSDataset(val_df, train_images_path, preprocessed_val_images_path, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model definition
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.base_model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=5)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return x

model = CustomModel().to('cpu')

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Training loop
early_stopping_patience = 7
early_stopping_counter = 0
best_val_loss = float('inf')

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    global best_val_loss, early_stopping_counter
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for images, labels in train_loader:
            images, labels = images.to('cpu'), labels.to('cpu')
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        y_true = []
        y_pred = []
        y_pred_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to('cpu'), labels.to('cpu')
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                y_true.extend(labels.numpy())
                y_pred.extend(preds.numpy())
                y_pred_probs.extend(torch.softmax(outputs, dim=1).numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Validation loss improved. Model saved.")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")

train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)

# Evaluation
def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_probs = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cpu'), labels.to('cpu')
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
            y_pred_probs.extend(torch.softmax(outputs, dim=1).numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_pred_probs)

model.load_state_dict(torch.load('best_model.pth'))
y_true, y_pred, y_pred_probs = evaluate_model(model, val_loader)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
auc_score = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
conf_matrix = confusion_matrix(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC: {auc_score:.4f}')
print(f'Cohen\'s Kappa: {kappa:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')

# ROC Curve
fpr = {}
tpr = {}
roc_auc = {}
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_probs[:, i])
    roc_auc[i] = roc_auc_score(y_true == i, y_pred_probs[:, i])

plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
for i in range(5):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} ROC (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
