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

# --- 1. PHYSICAL CONTROL SYSTEM (PID) ---
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.last_error = 0
        self.integral = 0

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        P = self.kp * error
        I = self.ki * self.integral
        D = self.kd * (error - self.last_error) / dt
        self.last_error = error
        return P + I + D

# --- 2. AI ARCHITECTURE (Attention-Based Backbone) ---
class SmartDroneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.SiLU())
        self.use_attn = use_attention
        if use_attention: self.attn = nn.MultiheadAttention(out_channels, 4, batch_first=True)
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

# --- 3. GHOST TELEMETRY GRAPH ---
def create_stealth_graph(history, width=150, height=50):
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.plot(history, color='#00FF00', linewidth=1)
    plt.fill_between(range(len(history)), history, color='#00FF00', alpha=0.05)
    plt.ylim(0, 100); plt.axis('off'); plt.tight_layout(pad=0)
    canvas = plt.get_current_fig_manager().canvas; canvas.draw()
    img = np.asarray(canvas.buffer_rgba()) 
    plt.close()
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

# --- 4. MASTER INTEGRATED ENGINE ---
def run_integrated_mission(video_in, video_out, weights):
    # Mapping AI commands to physical Tilt Angles (Degrees)
    COMMAND_MAP = {"LEFT": -20.0, "EVADE_L": -40.0, "RIGHT": 20.0, "EVADE_R": 40.0, "STABLE": 0.0}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNetV3().to(device); model.load_state_dict(torch.load(weights)); model.eval()
    cap = cv2.VideoCapture(video_in); w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    # Init Physics & Telemetry
    pid = PIDController(kp=0.7, ki=0.1, kd=0.05)
    current_tilt, danger_history, dt = 0.0, [0]*60, 1/fps
    
    print(f"🛰️ AEGIS-V3 ONLINE | Physics + Stealth HUD Enabled")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        inp = tf(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad(): p = torch.sigmoid(model(inp)).cpu().numpy()[0]
        
        # A. Decision Brain
        l_score = p[17, 8:, 0:5].mean() - (p[16, 8:, 0:5].mean() * 2.0)
        c_score = p[17, 8:, 5:11].mean() - (p[16, 8:, 5:11].mean() * 4.0)
        r_score = p[17, 8:, 11:16].mean() - (p[16, 8:, 11:16].mean() * 2.0)
        
        danger_now = p[16, 11:, 5:11].mean() * 100
        danger_history.append(danger_now); danger_history.pop(0)
        
        # B. Path to Physics Translation
        scores = [l_score, c_score, r_score]
        best = np.argmax(scores)
        
        target = 0.0 # Default STABLE
        if danger_now > 80: # Critical Situation
            target = COMMAND_MAP["EVADE_L"] if scores[0] > scores[2] else COMMAND_MAP["EVADE_R"]
            status = "EVADE"
        else:
            if best == 0: target, status = COMMAND_MAP["LEFT"], "LEFT"
            elif best == 2: target, status = COMMAND_MAP["RIGHT"], "RIGHT"
            else: target, status = COMMAND_MAP["STABLE"], "STABLE"

        # C. Compute Motor Correction (PID)
        correction = pid.compute(target, current_tilt, dt)
        current_tilt += correction * dt # Simulate physical tilt

        # D. Visuals (Stealth HUD)
        for gy in range(16):
            for gx in range(16):
                if p[16, gy, gx] > 0.92:
                    cv2.rectangle(frame, (gx*(w//16), gy*(h//16)), ((gx+1)*(w//16), (gy+1)*(h//16)), (0, 0, 255) if gy > 11 else (0, 255, 0), 1)

        # Status & Physics Readout
        color = (0, 0, 255) if status == "EVADE" else (0, 255, 0)
        cv2.putText(frame, f"{status} | TILT: {current_tilt:.1f} DEG", (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Ghost Graph
        graph_img = create_stealth_graph(danger_history)
        graph_area = frame[h-65:h-15, w-165:w-15]
        cv2.addWeighted(graph_img, 0.2, graph_area, 0.8, 0, graph_area)

        out.write(frame)
        
    cap.release(); out.release()
    print(f"✅ FINAL AEGIS-V3 MISSION LOGGED.")

if __name__ == "__main__":
    run_integrated_mission("/tmp/demo6_work/videoplayback (3).mp4", "AEGIS_V3_FINAL_MISSION.mp4", "smart_drone_v3_checkpoint_10.pth")