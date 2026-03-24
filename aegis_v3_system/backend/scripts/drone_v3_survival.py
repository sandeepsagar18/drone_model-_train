import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# --- 1. V3 SURVIVAL ARCHITECTURE ---
class SmartDroneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.SiLU())
        if use_attention: self.attn = nn.MultiheadAttention(out_channels, 4, batch_first=True)
        self.use_attn = use_attention
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
        self.layer1, self.layer2, self.layer3 = SmartDroneBlock(64, 128, False), SmartDroneBlock(128, 256, True), SmartDroneBlock(256, 512, True)
        self.detector = nn.Sequential(nn.AdaptiveAvgPool2d((16, 16)), nn.Conv2d(512, 18, 1))
    def forward(self, x):
        x = self.stem(x); x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = nn.functional.max_pool2d(self.layer2(x), 2); x = self.layer3(x)
        return self.detector(x)

# --- 2. PATH-FINDING BRAIN ---
def calculate_flight_path(p):
    # p[16] = Obstacles (Danger), p[17] = Road/Clear Path (Safety)
    
    # 🛡️ PROTECTION LOGIC: Focus on the bottom half (where the drone is flying into)
    # We divide the horizontal view into Left, Center, and Right
    left_danger   = p[16, 8:, 0:5].mean()
    center_danger = p[16, 8:, 5:11].mean()
    right_danger  = p[16, 8:, 11:16].mean()

    left_safety   = p[17, 8:, 0:5].mean()
    center_safety = p[17, 8:, 5:11].mean()
    right_safety  = p[17, 8:, 11:16].mean()

    # Scores: Higher is better (More Path, Less Obstacle)
    scores = [
        left_safety - (left_danger * 2.5), 
        center_safety - (center_danger * 3.0), # Center is most critical
        right_safety - (right_danger * 2.5)
    ]

    # Decide Action
    if center_danger > 0.8: # Immediate wall in front
        return "⚠️ EMERGENCY REVERSE / HOVER", (0, 0, 255)
    
    best_path = np.argmax(scores)
    actions = ["STEER LEFT (Safe Path Found)", "FULL SPEED FORWARD", "STEER RIGHT (Safe Path Found)"]
    return actions[best_path], (0, 255, 0)

# --- 3. THE AUTONOMOUS MISSION ENGINE ---
def run_survival_mission(video_in, video_out, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNetV3().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()

    cap = cv2.VideoCapture(video_in)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    print(f"🤖 Full Autonomy Engaged: Protecting Drone on {video_in}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        inp = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(inp)).cpu().numpy()[0]

        # 🧠 Brain Decision
        decision, color = calculate_flight_path(p)

        # 🎨 Drawing the AI's "Eyes"
        # Blue = Clean Path, Red = Solid Obstacle
        road_vis = cv2.applyColorMap(np.uint8(255 * cv2.resize(p[17], (w, h))), cv2.COLORMAP_BONE)
        frame = cv2.addWeighted(frame, 0.6, road_vis, 0.4, 0)

        gw, gh = w // 16, h // 16
        for gy in range(16):
            for gx in range(16):
                # Only show Red Boxes for high-confidence obstacles (Buildings/Trees)
                if p[16, gy, gx] > 0.75:
                    cv2.rectangle(frame, (gx*gw, gy*gh), ((gx+1)*gw, (gy+1)*gh), (0, 0, 255), 2)

        # 📊 Survival HUD
        cv2.rectangle(frame, (20, h-100), (600, h-20), (0, 0, 0), -1)
        cv2.putText(frame, f"AI PILOT: {decision}", (40, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Danger Meter
        total_danger = np.mean(p[16, 8:, :]) * 100
        meter_w = int(total_danger * 2)
        cv2.rectangle(frame, (40, h-40), (40+meter_w, h-30), (0, 0, 255), -1)
        cv2.putText(frame, "OBSTACLE DENSITY", (40, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(frame)

    cap.release(); out.release()
    print(f"✅ Mission Log Ready: {video_out}")

if __name__ == "__main__":
    run_survival_mission("/tmp/demo6_work/videoplayback (3).mp4", "v3_full_autonomy.mp4", "smart_drone_v3_checkpoint_10.pth")