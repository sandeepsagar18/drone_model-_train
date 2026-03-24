import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# --- 1. ARCHITECTURE (Must remain identical to load weights) ---
class SmartDroneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.use_attn = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(out_channels, 4, batch_first=True)

    def forward(self, x):
        x = self.conv(x)
        if not self.use_attn: return x
        b, c, h, w = x.shape
        flat = x.view(b, c, h*w).transpose(1, 2)
        attn_out, _ = self.attn(flat, flat, flat)
        return x + attn_out.transpose(1, 2).view(b, c, h, w)

class SmartDroneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.SiLU())
        self.layer1 = SmartDroneBlock(64, 128, False)
        self.layer2 = SmartDroneBlock(128, 256, True)
        self.grid_size = 8
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
            nn.Conv2d(256, 6, 1) 
        )

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = self.layer2(x)
        return self.detector(x)

# --- 2. DATASET WITH AUGMENTATION ---
class VisDroneDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'annotations')
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.grid_size = 8

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        w_orig, h_orig = image.size
        target = torch.zeros((6, self.grid_size, self.grid_size))
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split(',')
                    if len(data) >= 6 and int(data[4]) > 0:
                        nx, ny = (float(data[0]) + float(data[2])/2)/w_orig, (float(data[1]) + float(data[3])/2)/h_orig
                        nw, nh = float(data[2])/w_orig, float(data[3])/h_orig
                        gx, gy = int(nx * self.grid_size), int(ny * self.grid_size)
                        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                            target[0, gy, gx] = 1.0 
                            target[1:5, gy, gx] = torch.tensor([nx, ny, nw, nh])
                            target[5, gy, gx] = nh 

        if self.transform: image = self.transform(image)
        return image, target

# --- 3. EXTENDED TRAINING ENGINE ---
def train_extended(extra_epochs=50):
    device = torch.device("cuda")
    model = SmartDroneNet().to(device)
    
    # 1. Load the 50-epoch weights
    weight_path = "smart_drone_final.pth"
    if os.path.exists(weight_path):
        print(f"🔄 Loading existing weights from {weight_path}...")
        model.load_state_dict(torch.load(weight_path))
    else:
        print("❌ Error: Could not find smart_drone_final.pth. Ensure it is in the folder.")
        return

    # 2. Add ColorJitter to help with different drone camera lighting
    aug_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    dataset = VisDroneDataset("/tmp/demo6_work/data/VisDrone2019-DET-train", transform=aug_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    
    # 3. Fine-tuning Learning Rate (Much lower than before)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5) 
    criterion = nn.BCEWithLogitsLoss()

    print(f"🚀 Extended Deep Training Started (Epochs 51 to {50 + extra_epochs}) on L40S...")
    
    for epoch in range(extra_epochs):
        epoch_loss = 0
        real_epoch = epoch + 51 # For display
        
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {real_epoch} | Fine-Tune Loss: {avg_loss:.6f}")
        
        # Save checkpoints every 10 epochs
        if real_epoch % 10 == 0:
            torch.save(model.state_dict(), f"smart_drone_deep_tuned_{real_epoch}.pth")

    # Save the final result
    torch.save(model.state_dict(), "smart_drone_final.pth")
    print("✅ Extended Deep Training Complete!")

if __name__ == "__main__":
    # You can change this to 100 if you want to go to Epoch 150
    train_extended(extra_epochs=50)