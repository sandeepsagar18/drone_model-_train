import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os

# --- 1. ARCHITECTURE V3 (Consistent for weights) ---
class SmartDroneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.SiLU())
        if use_attention: self.attn = nn.MultiheadAttention(out_channels, 4, batch_first=True)
        self.use_attn = use_attention
    def forward(self, x):
        x = self.conv(x); b, c, h, w = x.shape
        if not self.use_attn: return x
        flat = x.view(b, c, h*w).transpose(1, 2); attn_out, _ = self.attn(flat, flat, flat)
        return x + attn_out.transpose(1, 2).view(b, c, h, w)

class SmartDroneNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.SiLU())
        self.layer1, self.layer2, self.layer3 = SmartDroneBlock(64, 128, False), SmartDroneBlock(128, 256, True), SmartDroneBlock(256, 512, True)
        self.detector = nn.Sequential(nn.AdaptiveAvgPool2d((16, 16)), nn.Conv2d(512, 18, 1))
    def forward(self, x):
        x = self.stem(x); x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = nn.functional.max_pool2d(self.layer2(x), 2); x = self.layer3(x)
        return self.detector(x)

# --- 2. ANALYTICS ENGINE ---
def generate_flight_report(video_path, weights_path, output_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNetV3().to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    danger_timeline = []
    time_steps = []
    
    print(f"📈 Extracting Analytics from {video_path}...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Process every 5th frame to speed up the report generation
        if frame_count % 5 == 0:
            inp = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(inp)).cpu().numpy()[0]
            
            # Metric: Near-Center Obstacle Density (Layer 16)
            danger_val = p[16, 10:, 4:12].mean() # Focusing on the immediate flight path
            danger_timeline.append(danger_val * 100)
            time_steps.append(frame_count / fps)
            
        frame_count += 1
        if frame_count % 500 == 0: print(f"Processed {int(frame_count/fps)} seconds...")

    cap.release()

    # --- 3. PLOTTING THE REPORT ---
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, danger_timeline, color='red', linewidth=2, label='Obstacle Proximity')
    plt.fill_between(time_steps, danger_timeline, color='red', alpha=0.2)
    
    plt.axhline(y=80, color='darkred', linestyle='--', label='Emergency Evasion Threshold')
    plt.title(f"V3 Autonomous Flight Performance: {os.path.basename(video_path)}")
    plt.xlabel("Mission Time (Seconds)")
    plt.ylabel("Danger Level (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(output_image)
    print(f"✅ Performance Graph saved as: {output_image}")

if __name__ == "__main__":
    generate_flight_report(
        "/tmp/demo6_work/videoplayback (3).mp4", 
        "smart_drone_v3_checkpoint_10.pth", 
        "mission_performance_chart.png"
    )