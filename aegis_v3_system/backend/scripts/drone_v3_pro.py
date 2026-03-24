import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# --- 1. ARCHITECTURE V3 (Consistent for .pth compatibility) ---
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

# --- 2. THE PATH PROJECTION ENGINE ---
def draw_trajectory(frame, best_idx, danger_lvl):
    h, w = frame.shape[:2]
    start_pt = (w // 2, h - 100)
    
    # Path Points based on Decision
    if best_idx == 0: # Left
        pts = np.array([[w//2, h-100], [w//3, h-250], [w//5, h-450]], np.int32)
    elif best_idx == 2: # Right
        pts = np.array([[w//2, h-100], [2*w//3, h-250], [4*w//5, h-450]], np.int32)
    else: # Center
        pts = np.array([[w//2, h-100], [w//2, h-300], [w//2, h-500]], np.int32)

    # Glow Color: Red if danger is high, Green if safe
    color = (0, 0, 255) if danger_lvl > 0.8 else (0, 255, 0)
    
    # Draw a glowing polyline
    cv2.polylines(frame, [pts], False, color, 8, cv2.LINE_AA)
    cv2.polylines(frame, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)

# --- 3. MISSION ENGINE ---
def run_pro_mission(video_in, video_out, weights):
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

        # Calculate Scores
        l_score = p[17, 8:, 0:5].mean() - (p[16, 8:, 0:5].mean() * 2.0)
        c_score = p[17, 8:, 5:11].mean() - (p[16, 8:, 5:11].mean() * 4.0) # High Center Penalty
        r_score = p[17, 8:, 11:16].mean() - (p[16, 8:, 11:16].mean() * 2.0)
        
        scores = [l_score, c_score, r_score]
        best_idx = np.argmax(scores)
        danger_lvl = p[16, 12:, 5:11].mean() # Near Center Danger

        # --- Visual Overlays ---
        # 1. Distance-Aware Grid (Red=Near, Green=Far)
        gw, gh = w // 16, h // 16
        for gy in range(16):
            for gx in range(16):
                if p[16, gy, gx] > 0.85:
                    box_color = (0, 0, 255) if gy > 11 else (0, 255, 0)
                    cv2.rectangle(frame, (gx*gw, gy*gh), ((gx+1)*gw, (gy+1)*gh), box_color, 1 if gy < 12 else 2)

        # 2. Path Projection Line
        draw_trajectory(frame, best_idx, danger_lvl)

        # 🛰️ PROFESSIONAL HUD
        # Top Right: Danger Radar
        cv2.rectangle(frame, (w-320, 20), (w-20, 60), (40, 40, 40), -1)
        cv2.rectangle(frame, (w-320, 20), (w-320 + int(danger_lvl*300), 60), (0, 0, 255), -1)
        cv2.putText(frame, "PROXIMITY SCAN", (w-315, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Bottom Command
        status = "CRITICAL: AVOIDANCE" if danger_lvl > 0.8 else "AUTONOMOUS: ACTIVE"
        cmd_text = ["STEERING LEFT", "MOVING FORWARD", "STEERING RIGHT"][best_idx]
        cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"{status} | COMMAND: {cmd_text}", (40, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if danger_lvl < 0.8 else (0, 0, 255), 2)

        out.write(frame)
    cap.release(); out.release()
    print(f"✅ Pro-Navigator Log Ready: {video_out}")

if __name__ == "__main__":
    run_pro_mission("/tmp/demo6_work/videoplayback (3).mp4", "v3_pro_mission.mp4", "smart_drone_v3_checkpoint_10.pth")