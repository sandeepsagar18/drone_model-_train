import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from PIL import Image
import os

# --- 0. ENVIRONMENT & PERMISSIONS ---
os.environ['TORCH_HOME'] = '/tmp/demo6_work/torch_cache'
os.makedirs('/tmp/demo6_work/torch_cache', exist_ok=True)

# --- 1. ARCHITECTURE V3 (16x16 Grid) ---
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

class SmartDroneNetV3(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.grid_size = 16 
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.SiLU())
        self.layer1 = SmartDroneBlock(64, 128, False)
        self.layer2 = SmartDroneBlock(128, 256, True)
        self.layer3 = SmartDroneBlock(256, 512, True) 
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
            nn.Conv2d(512, 6 + num_classes, 1) 
        )

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = nn.functional.max_pool2d(self.layer2(x), 2)
        x = self.layer3(x)
        return self.detector(x)

# --- 2. DATASET V3 ---
class VisDroneDatasetV3(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'annotations')
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.grid_size = 16
        self.class_map = {i: i-1 for i in range(1, 11)}

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        w_orig, h_orig = image.size
        target = torch.zeros((18, 16, 16))
        
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split(',')
                    if len(data) >= 6:
                        c_id = int(data[5])
                        if c_id in self.class_map:
                            nx, ny = (float(data[0]) + float(data[2])/2)/w_orig, (float(data[1]) + float(data[3])/2)/h_orig
                            nw, nh = float(data[2])/w_orig, float(data[3])/h_orig
                            gx, gy = int(nx * 16), int(ny * 16)
                            if 0 <= gx < 16 and 0 <= gy < 16:
                                target[0, gy, gx] = 1.0
                                target[1:5, gy, gx] = torch.tensor([nx, ny, nw, nh])
                                target[5, gy, gx] = nh
                                target[6 + self.class_map[c_id], gy, gx] = 1.0
        
        if self.transform: image = self.transform(image)
        return image, target

# --- 3. THE "MEMORY-SAFE" TRAINING ENGINE ---
def train_v3():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    
    model = SmartDroneNetV3().to(device)
    
    # 🧠 Load pre-trained DeepLabV3 Teacher
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    teacher = models.segmentation.deeplabv3_resnet50(weights=weights).to(device).eval()
    
    # ⚡ Backbone Weight Transfer
    if os.path.exists("smart_drone_v2_final.pth"):
        print("⚡ Transferring Backbone patterns from V2...")
        v2_weights = torch.load("smart_drone_v2_final.pth")
        model_dict = model.state_dict()
        safe_weights = {k: v for k, v in v2_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(safe_weights)
        model.load_state_dict(model_dict)

    # 📉 LOW MEMORY SETTINGS
    batch_size = 2 
    accumulation_steps = 16 # Effective batch size = 32
    
    loader = DataLoader(VisDroneDatasetV3("/tmp/demo6_work/data/VisDrone2019-DET-train", 
                        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])),
                        batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision Scaler

    print(f"🚀 V3 Training: Goal 50 Epochs | Effective Batch: {batch_size * accumulation_steps}")

    for epoch in range(50):
        total_loss = 0
        optimizer.zero_grad()
        
        for i, (imgs, lbls) in enumerate(loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # Use Mixed Precision to fit into 7GB
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    t_out = teacher(imgs)['out']
                    trees = torch.nn.functional.interpolate(t_out[:, 15:16], size=(16, 16))
                    roads = torch.nn.functional.interpolate(t_out[:, 0:1], size=(16, 16))
                    lbls[:, 16, :, :] = torch.sigmoid(trees).squeeze(1)
                    lbls[:, 17, :, :] = torch.sigmoid(roads).squeeze(1)

                preds = model(imgs)
                # Divide loss by accumulation steps to keep average correct
                loss = criterion(preds, lbls) / accumulation_steps

            # Backward pass (Scaled for FP16)
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache() # Flush memory every update

            total_loss += loss.item() * accumulation_steps
            del imgs, lbls, preds, t_out

        print(f"V3 Epoch {epoch+1} | Combined Loss: {total_loss/len(loader):.5f}")
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"smart_drone_v3_checkpoint_{epoch+1}.pth")

    torch.save(model.state_dict(), "smart_drone_v3_final.pth")
    print("✅ V3 Scene Awareness Training Complete!")

if __name__ == "__main__":
    train_v3()