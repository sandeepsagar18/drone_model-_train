import os
# Fix for Matplotlib permission issues on shared servers
os.environ['MPLCONFIGDIR'] = '/tmp/demo6_work/.matplotlib'

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# --- 1. ARCHITECTURE V3 (Stable Baseline) ---
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
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.SiLU())
        self.layer1 = SmartDroneBlock(64, 128, False)
        self.layer2 = SmartDroneBlock(128, 256, True)
        self.layer3 = SmartDroneBlock(256, 512, True)
        self.detector = nn.Sequential(nn.AdaptiveAvgPool2d((16, 16)), nn.Conv2d(512, 18, 1))

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = nn.functional.max_pool2d(self.layer2(x), 2)
        x = self.layer3(x)
        return self.detector(x)

# --- 2. STEALTH TELEMETRY (150x50 Pixel Ghost Graph) ---
def create_stealth_graph(history, width=150, height=50):
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.plot(history, color='#00FF00', linewidth=1) 
    plt.fill_between(range(len(history)), history, color='#00FF00', alpha=0.05)
    plt.ylim(0, 100); plt.axis('off')
    plt.tight_layout(pad=0)
    
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba()) 
    plt.close()
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# --- 3. STEALTH MISSION ENGINE ---
def run_stealth_mission(video_in, video_out, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNetV3().to(device); model.load_state_dict(torch.load(weights)); model.eval()
    
    cap = cv2.VideoCapture(video_in)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    danger_history = [0] * 60 
    print(f"🕵️ Mission Started | Stealth Mode Active | Output: {video_out}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        inp = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(inp)).cpu().numpy()[0]
        
        danger_now = p[16, 11:, 5:11].mean() * 100
        danger_history.append(danger_now); danger_history.pop(0)
        
        # 1. Hairline Obstacle Grid (92% Confidence Threshold)
        for gy in range(16):
            for gx in range(16):
                if p[16, gy, gx] > 0.92: 
                    color = (0, 0, 255) if gy > 11 else (0, 255, 0)
                    cv2.rectangle(frame, (gx*(w//16), gy*(h//16)), ((gx+1)*(w//16), (gy+1)*(h//16)), color, 1)

        # 2. Status Updates (Micro-Text)
        status = "EVADE" if danger_now > 80 else "STABLE"
        color = (0, 0, 255) if danger_now > 80 else (0, 255, 0)
        # Scale 0.4 for micro-visibility
        cv2.putText(frame, status, (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # 3. Ghost Telemetry (Transparent Overlay)
        graph_img = create_stealth_graph(danger_history)
        graph_area = frame[h-65:h-15, w-165:w-15]
        # Low 0.2 Alpha for transparent effect
        cv2.addWeighted(graph_img, 0.2, graph_area, 0.8, 0, graph_area)

        out.write(frame)
        
    cap.release(); out.release()
    print(f"✅ Stealth mission finalized.")

if __name__ == "__main__":
    run_stealth_mission(
        "/tmp/demo6_work/videoplayback (3).mp4", 
        "V3_STEALTH_REPORT.mp4", 
        "smart_drone_v3_checkpoint_10.pth"
    )