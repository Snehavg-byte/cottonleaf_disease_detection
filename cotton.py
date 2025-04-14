import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from timm import create_model
from tqdm import tqdm
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import torchvision.transforms.functional as TF

# -------------------------------
# Device configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Dataset path (confirmed)
# -------------------------------
dataset_path = r"C:\Users\SNEHA\PycharmProjects\cottonleaf\cotton dataset\CoSev\CoSev"
print("Dataset path exists:", os.path.exists(dataset_path))

# -------------------------------
# Image transforms
# -------------------------------
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------------
# Load dataset and dataloaders
# -------------------------------
dataset = ImageFolder(root=dataset_path, transform=data_transform)
class_names = dataset.classes

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# -------------------------------
# Hybrid Model
# -------------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridModel, self).__init__()
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0, global_pool="avg")
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(768 + 32, num_classes)

    def forward(self, x):
        swin_feat = self.swin(x)
        cnn_feat = self.cnn(x)
        cnn_feat = cnn_feat.view(cnn_feat.size(0), -1)
        combined = torch.cat((swin_feat, cnn_feat), dim=1)
        return self.fc(combined)

model = HybridModel(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------------
# Training loop
# -------------------------------
def train_model():
    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=running_loss / (total // 16 + 1), accuracy=100 * correct / total)

    torch.save(model.state_dict(), "hybrid_cotton_model.pth")
    print("Training completed and model saved as hybrid_cotton_model.pth.")

# -------------------------------
# Load and Predict
# -------------------------------
def load_model():
    model.load_state_dict(torch.load("hybrid_cotton_model.pth", map_location=device))
    model.eval()

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = data_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# -------------------------------
# GUI using Tkinter
# -------------------------------
def run_gui():
    def browse_file():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            try:
                prediction = predict_image(file_path)
                result_label.config(text=f"Prediction: {prediction}")

                # Display image
                img = Image.open(file_path)
                img = img.resize((224, 224))
                img_tk = ImageTk.PhotoImage(img)
                image_label.config(image=img_tk)
                image_label.image = img_tk  # keep a reference

            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")

    app = tk.Tk()
    app.title("Cotton Leaf Disease Detection")
    app.geometry("500x450")
    app.configure(bg="white")

    tk.Label(app, text="Hybrid Cotton Leaf Disease Detector", font=("Helvetica", 14, "bold"), bg="white").pack(pady=10)
    tk.Button(app, text="Select Image", command=browse_file, bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=10)

    global result_label
    result_label = tk.Label(app, text="", font=("Helvetica", 12), bg="white")
    result_label.pack(pady=10)

    global image_label
    image_label = tk.Label(app, bg="white")
    image_label.pack(pady=10)

    app.mainloop()

# -------------------------------
# Execution
# -------------------------------
if __name__ == "__main__":
    if not os.path.exists("hybrid_cotton_model.pth"):
        train_model()
    else:
        print("Model already trained. Loading weights...")

    load_model()
    run_gui()
