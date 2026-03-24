import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# --- 1. ARCHITECTURE V2 (11 Output Channels) ---
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

class SmartDroneNetV2(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.SiLU())
        self.layer1 = SmartDroneBlock(64, 128, False)
        self.layer2 = SmartDroneBlock(128, 256, True)
        self.grid_size = 8
        # 6 original channels + 5 class channels = 11
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
            nn.Conv2d(256, 6 + num_classes, 1) 
        )

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = self.layer2(x)
        return self.detector(x)

# --- 2. MULTI-CLASS DATASET LOADER ---
class VisDroneDatasetV2(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'annotations')
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.grid_size = 8
        # Mapping: {VisDroneID: OurID}
        self.class_map = {1: 0, 4: 1, 5: 2, 6: 3, 9: 4} # Ped, Car, Van, Bus, Truck

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        w_orig, h_orig = image.size
        # 11 channels: [obj, x, y, w, h, dist, ped, car, van, bus, truck]
        target = torch.zeros((11, self.grid_size, self.grid_size))
        
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split(',')
                    if len(data) >= 6:
                        class_id = int(data[5])
                        if class_id in self.class_map:
                            nx, ny = (float(data[0]) + float(data[2])/2)/w_orig, (float(data[1]) + float(data[3])/2)/h_orig
                            nw, nh = float(data[2])/w_orig, float(data[3])/h_orig
                            gx, gy = int(nx * self.grid_size), int(ny * self.grid_size)
                            
                            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                                target[0, gy, gx] = 1.0 # Objectness
                                target[1:5, gy, gx] = torch.tensor([nx, ny, nw, nh])
                                target[5, gy, gx] = nh # Distance proxy
                                target[6 + self.class_map[class_id], gy, gx] = 1.0 # Class bit

        if self.transform: image = self.transform(image)
        return image, target

# --- 3. TRAINING ENGINE ---
def train_v2(epochs=50):
    device = torch.device("cuda")
    model = SmartDroneNetV2().to(device)
    
    # Check for V1 weights to jumpstart learning
    if os.path.exists("smart_drone_final.pth"):
        print("💡 Found V1 brain. Transferring knowledge to V2...")
        # We only load the backbone (stem, layer1, layer2), not the detector head
        v1_state = torch.load("smart_drone_final.pth")
        model_state = model.state_dict()
        pretrained_base = {k: v for k, v in v1_state.items() if k in model_state and 'detector' not in k}
        model_state.update(pretrained_base)
        model.load_state_dict(model_state)

    dataset = VisDroneDatasetV2("/tmp/demo6_work/data/VisDrone2019-DET-train", 
                               transform=transforms.Compose([
                                   transforms.Resize((128,128)),
                                   transforms.ColorJitter(0.2, 0.2),
                                   transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print(f"🚀 V2 Multi-Class Training Started on L40S...")
    for epoch in range(epochs):
        epoch_loss = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"V2 Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.5f}")

    torch.save(model.state_dict(), "smart_drone_v2_final.pth")
    print("✅ V2 Training Complete! Saved as smart_drone_v2_final.pth")

if __name__ == "__main__":
    train_v2()