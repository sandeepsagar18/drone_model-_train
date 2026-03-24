import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# --- 1. V3 ARCHITECTURE (Locked for Checkpoint Compatibility) ---
class SmartDroneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.SiLU())
        if use_attention: self.attn = nn.MultiheadAttention(out_channels, 4, batch_first=True)
        self.use_attn = use_attention
    def forward(self, x):
        x = self.conv(x); b, c, h, w = x.shape
        if not self.use_attn: return x
        flat = x.view(b, c, h*w).transpose(1, 2)
        attn_out, _ = self.attn(flat, flat, flat)
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

# --- 2. THE TACTICAL BRAIN (Near vs. Far Weighing) ---
def tactical_decision(p):
    # Split grid into Horizon (Top) and Ground (Bottom)
    near_zone = p[16, 10:, :] # Bottom 6 rows
    far_zone = p[16, :10, :]  # Top 10 rows

    # Calculate Safety Scores (Safety Layer 17)
    left_score   = p[17, 8:, 0:5].mean()   - (p[16, 8:, 0:5].mean() * 2.0)
    center_score = p[17, 8:, 5:11].mean()  - (p[16, 8:, 5:11].mean() * 3.5) # Protect Center Heavily
    right_score  = p[17, 8:, 11:16].mean() - (p[16, 8:, 11:16].mean() * 2.0)

    # 🛑 CRITICAL PROTECTION: Immediate proximity check
    if near_zone.mean() > 0.75:
        return "⚠️ EMERGENCY BRAKE: COLLISION IMMINENT", (0, 0, 255)

    scores = [left_score, center_score, right_score]
    actions = ["STEER LEFT", "STEER CENTER / FORWARD", "STEER RIGHT"]
    return actions[np.argmax(scores)], (0, 255, 0)

# --- 3. THE MISSION ENGINE ---
def run_tactical_mission(video_in, video_out, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNetV3().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()

    cap = cv2.VideoCapture(video_in)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    print(f"📡 Tactical Mission Started: {video_in}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        inp = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(inp)).cpu().numpy()[0]

        # 🧠 Get Navigation Decision
        command, cmd_color = tactical_decision(p)

        # 🟢 Draw Distance-Aware Grid
        gw, gh = w // 16, h // 16
        for gy in range(16):
            for gx in range(16):
                if p[16, gy, gx] > 0.85: # Confidence Threshold
                    # Near (Bottom) = RED | Far (Top) = GREEN
                    color = (0, 0, 255) if gy > 10 else (0, 255, 0)
                    thickness = 2 if gy > 10 else 1
                    cv2.rectangle(frame, (gx*gw, gy*gh), ((gx+1)*gw, (gy+1)*gh), color, thickness)

        # 🛰️ CLEAN HUD - Tactical Layout
        # 1. Top Right Danger Meter
        danger_lvl = np.mean(p[16, 10:, :]) # Near Danger
        meter_w = 200
        cv2.rectangle(frame, (w-230, 30), (w-30, 60), (50, 50, 50), -1) # BG
        cv2.rectangle(frame, (w-230, 30), (w-230 + int(danger_lvl*meter_w), 60), (0, 0, 255), -1) # Fill
        cv2.putText(frame, "PROXIMITY DANGER", (w-230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 2. Bottom Navigation HUD
        cv2.rectangle(frame, (0, h-70), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"NAV AI: {command}", (40, h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cmd_color, 2)

        out.write(frame)

    cap.release(); out.release()
    print(f"✅ Tactical Log Finalized: {video_out}")

if __name__ == "__main__":
    run_tactical_mission("/tmp/demo6_work/videoplayback (3).mp4", "v3_tactical_autonomy.mp4", "smart_drone_v3_checkpoint_10.pth")