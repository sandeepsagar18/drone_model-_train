import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# --- 1. ARCHITECTURE V3: THE ATTENTION BRAIN ---
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

# --- 2. THE PATHFINDING BRAIN (Survival Logic) ---
def calculate_flight_path(p_grid):
    """
    Scans 3 sectors to decide movement.
    p_grid[16] = Obstacles (Red)
    p_grid[17] = Safety/Road (Blue)
    """
    # Focusing on the lower 10 rows (The immediate flight path)
    horizon = p_grid[:, 6:, :] 
    
    # Sector Splitting (Left, Center, Right)
    left_danger   = horizon[16, :, 0:5].mean()
    center_danger = horizon[16, :, 5:11].mean()
    right_danger  = horizon[16, :, 11:16].mean()

    left_safety   = horizon[17, :, 0:5].mean()
    center_safety = horizon[17, :, 5:11].mean()
    right_safety  = horizon[17, :, 11:16].mean()

    # SCORING: Safety minus Weighted Danger
    # Center is weighted higher (1.8) to prevent head-on crashes
    scores = [
        left_safety - (left_danger * 1.5), 
        center_safety - (center_danger * 1.8), 
        right_safety - (right_danger * 1.5)
    ]

    # Survival Threshold: If center is a solid wall
    if center_danger > 0.88: 
        return "⚠️ EMERGENCY REVERSE", (0, 0, 255)
    
    # Path Selection
    best_path = np.argmax(scores)
    actions = ["STEER LEFT (Optimal Path)", "FORWARD (Path Clear)", "STEER RIGHT (Optimal Path)"]
    return actions[best_path], (0, 255, 0)

# --- 3. THE AUTONOMOUS MISSION ENGINE ---
def run_explorer_mission(video_in, video_out, weights, threshold=0.85):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNetV3().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()

    cap = cv2.VideoCapture(video_in)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    print(f"🛰️ Explorer Mode Active | Threshold: {threshold} | Analyzing: {video_in}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        inp = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(inp)).cpu().numpy()[0]

        # 🧠 Brain Decision
        decision, cmd_color = calculate_flight_path(p)

        # 🔵 Visualizing "The Path" (Safety Layer)
        road_vis = cv2.applyColorMap(np.uint8(255 * cv2.resize(p[17], (w, h))), cv2.COLORMAP_BONE)
        frame = cv2.addWeighted(frame, 0.7, road_vis, 0.3, 0)

        # 🔴 Visualizing "The Danger" (16x16 Obstacle Grid)
        gw, gh = w // 16, h // 16
        for gy in range(16):
            for gx in range(16):
                if p[16, gy, gx] > threshold:
                    cv2.rectangle(frame, (gx*gw, gy*gh), ((gx+1)*gw, (gy+1)*gh), (0, 0, 255), 2)

        # 📊 MISSION HUD
        cv2.rectangle(frame, (0, h-90), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"AI PILOT: {decision}", (30, h-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, cmd_color, 3)
        
        # Obstacle Density Meter
        danger_val = np.mean(p[16, 8:, :])
        meter_end = int(30 + (danger_val * 300))
        cv2.rectangle(frame, (w-350, h-45), (w-350 + meter_end, h-35), (0, 0, 255), -1)
        cv2.putText(frame, "DANGER", (w-350, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

    cap.release(); out.release()
    print(f"✅ Mission Report Saved: {video_out}")

if __name__ == "__main__":
    # Test on your most complex video
    run_explorer_mission(
        "/tmp/demo6_work/videoplayback (3).mp4", 
        "v3_explorer_mission_final.mp4", 
        "smart_drone_v3_checkpoint_10.pth"
    )