import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# --- 1. THE SMART ARCHITECTURE ---
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
            nn.Conv2d(256, 6, 1) # [Objectness, x, y, w, h, dist]
        )

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = self.layer2(x)
        return self.detector(x)

# --- 2. DATASET LOADER ---
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
                        # Normalize 0-1
                        nx, ny = (float(data[0]) + float(data[2])/2)/w_orig, (float(data[1]) + float(data[3])/2)/h_orig
                        nw, nh = float(data[2])/w_orig, float(data[3])/h_orig
                        gx, gy = int(nx * self.grid_size), int(ny * self.grid_size)
                        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                            target[0, gy, gx] = 1.0 
                            target[1:5, gy, gx] = torch.tensor([nx, ny, nw, nh])
                            target[5, gy, gx] = nh # Height as a proxy for distance

        if self.transform: image = self.transform(image)
        return image, target

# --- 3. TRAINING ENGINE ---
def train():
    device = torch.device("cuda")
    model = SmartDroneNet().to(device)
    
    # Optional: Load previous weights to continue training
    if os.path.exists("smart_drone_final.pth"):
        print("🔄 Loading existing brain to continue training...")
        model.load_state_dict(torch.load("smart_drone_final.pth"))

    dataset = VisDroneDataset("/tmp/demo6_work/data/VisDrone2019-DET-train", 
                             transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    print(f"🚀 Deep Training Started (50 Epochs) on L40S...")
    for epoch in range(50):
        epoch_loss = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}/50 | Loss: {epoch_loss/len(loader):.5f} | LR: {optimizer.param_groups[0]['lr']}")
        
        # Save every 10 epochs as a backup
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"smart_drone_checkpoint_{epoch+1}.pth")

    torch.save(model.state_dict(), "smart_drone_final.pth")
    print("✅ Deep Training Complete!")

if __name__ == "__main__":
    train()