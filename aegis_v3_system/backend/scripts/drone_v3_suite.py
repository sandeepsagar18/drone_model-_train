import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from PIL import Image
import os

# --- 1. THE V3 ARCHITECTURE (CNN + ATTENTION) ---
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
            nn.Conv2d(512, 6 + num_classes, 1) # 18 Channels Total
        )

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = nn.functional.max_pool2d(self.layer2(x), 2)
        x = self.layer3(x)
        return self.detector(x)

# --- 2. THE BATCH TESTING ENGINE ---
def run_batch_test(video_list, model_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Weights
    model = SmartDroneNetV3().to(device)
    if not os.path.exists(model_weights):
        print(f"❌ Error: Weights file {model_weights} not found!")
        return
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    for vid_name in video_list:
        input_path = f"/tmp/demo6_work/{vid_name}"
        output_path = f"/tmp/demo6_work/v3_result_{vid_name}"
        
        if not os.path.exists(input_path):
            print(f"⚠️ Skipping {vid_name}: File not found.")
            continue

        cap = cv2.VideoCapture(input_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (w, h))

        print(f"🎥 Analyzing: {vid_name}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # AI Inference
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = tf(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(inp)).cpu().numpy()[0]

            # 1. Road Heatmap (Blue "Bone" overlay)
            road_map = cv2.resize(p[17], (w, h))
            road_vis = cv2.applyColorMap(np.uint8(255 * road_map), cv2.COLORMAP_BONE)
            frame = cv2.addWeighted(frame, 0.7, road_vis, 0.3, 0)

            # 2. 16x16 Grid Detections (Green Boxes)
            gw, gh = w // 16, h // 16
            for gy in range(16):
                for gx in range(16):
                    if p[0, gy, gx] > 0.45: # Object Confidence
                        cv2.rectangle(frame, (gx*gw, gy*gh), ((gx+1)*gw, (gy+1)*gh), (0, 255, 0), 1)

            # 3. HUD Display
            cv2.putText(frame, f"V3 INTELLIGENCE | {vid_name}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)

        cap.release(); out.release()
        print(f"✅ Finished: {output_path}")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # Your 3 video mission files
    my_videos = [
        "videoplayback (1).mp4",
        "videoplayback (2).mp4",
        "videoplayback (3).mp4"
    ]
    
    # Use your latest successful brain
    weights_path = "smart_drone_v3_checkpoint_10.pth"
    
    run_batch_test(my_videos, weights_path)