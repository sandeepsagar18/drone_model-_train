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

# --- 1. ARCHITECTURE V3 (Attention-Based Backbone) ---
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

# --- 2. THE DYNAMIC GRAPH GENERATOR (Fixed for Matplotlib 3.8+) ---
def create_graph_overlay(history, width=400, height=150):
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.plot(history, color='red', linewidth=2)
    plt.fill_between(range(len(history)), history, color='red', alpha=0.3)
    plt.ylim(0, 100)
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    
    # Updated to use buffer_rgba() for modern Matplotlib compatibility
    img = np.asarray(canvas.buffer_rgba()) 
    plt.close()
    
    # Convert RGBA to BGR for OpenCV
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# --- 3. TRAJECTORY PROJECTION ---
def draw_trajectory(frame, best_idx, danger_lvl):
    h, w = frame.shape[:2]
    if best_idx == 0: # Left
        pts = np.array([[w//2, h-100], [w//3, h-250], [w//5, h-450]], np.int32)
    elif best_idx == 2: # Right
        pts = np.array([[w//2, h-100], [2*w//3, h-250], [4*w//5, h-450]], np.int32)
    else: # Center
        pts = np.array([[w//2, h-100], [w//2, h-300], [w//2, h-500]], np.int32)

    color = (0, 0, 255) if danger_lvl > 80 else (0, 255, 0)
    cv2.polylines(frame, [pts], False, color, 8, cv2.LINE_AA)
    cv2.polylines(frame, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

# --- 4. MASTER ENGINE ---
def run_master_hud(video_in, video_out, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNetV3().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    cap = cv2.VideoCapture(video_in)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    danger_history = [0] * 60 # Scrolling window for 2 seconds of data
    
    print(f"🎬 Processing Master HUD... Generating: {video_out}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        inp = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(inp)).cpu().numpy()[0]
        
        # Scoring Logic
        l_score = p[17, 8:, 0:5].mean() - (p[16, 8:, 0:5].mean() * 2.0)
        c_score = p[17, 8:, 5:11].mean() - (p[16, 8:, 5:11].mean() * 4.0)
        r_score = p[17, 8:, 11:16].mean() - (p[16, 8:, 11:16].mean() * 2.0)
        
        best_idx = np.argmax([l_score, c_score, r_score])
        danger_now = p[16, 10:, 5:11].mean() * 100
        danger_history.append(danger_now)
        danger_history.pop(0)
        
        # Draw Visual Elements
        draw_trajectory(frame, best_idx, danger_now)
        
        # Grid System
        gw, gh = w // 16, h // 16
        for gy in range(16):
            for gx in range(16):
                if p[16, gy, gx] > 0.85:
                    color = (0, 0, 255) if gy > 11 else (0, 255, 0)
                    cv2.rectangle(frame, (gx*gw, gy*gh), ((gx+1)*gw, (gy+1)*gh), color, 1 if gy < 11 else 2)

        # Telemetry Box
        cv2.rectangle(frame, (w-420, h-220), (w-20, h-20), (20, 20, 20), -1)
        graph_img = create_graph_overlay(danger_history)
        frame[h-190:h-40, w-410:w-10] = graph_img
        
        # Text Labels
        status = "CRITICAL: EVASION" if danger_now > 80 else "SYSTEM: STABLE"
        cv2.putText(frame, "LIVE TELEMETRY", (w-410, h-200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"MODE: {status}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if danger_now > 80 else (0, 255, 0), 3)

        out.write(frame)
        
    cap.release(); out.release()
    print(f"✅ Mission Accomplished! File saved as: {video_out}")

if __name__ == "__main__":
    run_master_hud(
        "/tmp/demo6_work/videoplayback (3).mp4", 
        "V3_FINAL_MISSION_REPORT.mp4", 
        "smart_drone_v3_checkpoint_10.pth"
    )