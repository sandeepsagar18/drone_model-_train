import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# --- 1. ARCHITECTURE V3 (Locked for weights compatibility) ---
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

# --- 2. THE PATHFINDING BRAIN (Escape & Survival Logic) ---
def autonomous_brain(p):
    # Layers: p[16] = Obstacles (Red), p[17] = Flight Path (Blue/Safety)
    
    # 🕵️ STEP 1: Scan for immediate threats (The "Crash Zone")
    near_center_danger = p[16, 12:, 5:11].mean() # Bottom-middle
    
    # 🕵️ STEP 2: Calculate Escape Scores for 3 Sectors
    # Score = (Safe Path - Obstacle Penalty)
    l_score = p[17, :, 0:5].mean()   - (p[16, :, 0:5].mean() * 1.8)
    c_score = p[17, :, 5:11].mean()  - (p[16, :, 5:11].mean() * 3.0) # Center is highly penalized
    r_score = p[17, :, 11:16].mean() - (p[16, :, 11:16].mean() * 1.8)

    scores = [l_score, c_score, r_score]
    directions = ["STEER LEFT", "FULL SPEED FORWARD", "STEER RIGHT"]
    best_idx = np.argmax(scores)

    # 🛑 SURVIVAL OVERRIDE: If the chosen path is still too dangerous
    if near_center_danger > 0.85:
        # Check if sides are significantly clearer than the center
        if l_score > c_score + 0.3: return "HARD LEFT EVASION", (0, 0, 255)
        if r_score > c_score + 0.3: return "HARD RIGHT EVASION", (0, 0, 255)
        return "CRITICAL STOP: NO ESCAPE PATH", (0, 0, 255)

    return directions[best_idx], (0, 255, 0)

# --- 3. THE MISSION ENGINE ---
def run_autonomous_suite(video_in, video_out, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNetV3().to(device); model.load_state_dict(torch.load(weights)); model.eval()
    cap = cv2.VideoCapture(video_in); w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        inp = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad(): p = torch.sigmoid(model(inp)).cpu().numpy()[0]

        # 🧠 Get Navigation Decision
        command, color = autonomous_brain(p)

        # 🔵 Overlay Safe Path (Safety Layer)
        path_vis = cv2.applyColorMap(np.uint8(255 * cv2.resize(p[17], (w, h))), cv2.COLORMAP_BONE)
        frame = cv2.addWeighted(frame, 0.75, path_vis, 0.25, 0)

        # 🟥/🟩 Distance-Aware Grid (Red=Near, Green=Far)
        gw, gh = w // 16, h // 16
        for gy in range(16):
            for gx in range(16):
                if p[16, gy, gx] > 0.85:
                    box_color = (0, 0, 255) if gy > 11 else (0, 255, 0)
                    cv2.rectangle(frame, (gx*gw, gy*gh), ((gx+1)*gw, (gy+1)*gh), box_color, 1 if gy < 11 else 2)

        # 🛰️ ENHANCED MISSION HUD
        # Top-Left: Proximity Scanner
        danger_val = p[16, 11:, 5:11].mean()
        cv2.rectangle(frame, (20, 20), (320, 60), (30, 30, 30), -1)
        cv2.rectangle(frame, (20, 20), (20 + int(danger_val*300), 60), (0, 0, 255), -1)
        cv2.putText(frame, "COLLISION SCANNER", (25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Bottom: Autonomous Command
        cv2.rectangle(frame, (0, h-80), (w, h), (0,0,0), -1)
        cv2.putText(frame, f"NAV AI: {command}", (40, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        out.write(frame)
    cap.release(); out.release()
    print(f"✅ Autonomous Log Finalized: {video_out}")

if __name__ == "__main__":
    run_autonomous_suite("/tmp/demo6_work/videoplayback (3).mp4", "v3_final_navigation.mp4", "smart_drone_v3_checkpoint_10.pth")