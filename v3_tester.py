import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# --- 1. THE V3 BRAIN (Architecture) ---
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
            nn.Conv2d(512, 6 + num_classes, 1) 
        )

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = nn.functional.max_pool2d(self.layer2(x), 2)
        x = self.layer3(x)
        return self.detector(x)

# --- 2. THE VISION ENGINE ---
def test_on_video(video_in, video_out, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = SmartDroneNetV3().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()

    # Video Setup
    cap = cv2.VideoCapture(video_in)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    print(f"🚀 Testing V3 Intelligence on: {video_in}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # AI Inference
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]

        # 16x16 Heatmaps (Resized to video size)
        road_map = cv2.resize(preds[17], (w, h))  # Path index
        obj_map = preds[0]                       # Object confidence grid

        # --- DRAWING OVERLAYS ---
        
        # 🟢 1. Road "Glow" (Blue tint for safe path)
        road_mask = np.uint8(255 * road_map)
        road_color = cv2.applyColorMap(road_mask, cv2.COLORMAP_BONE)
        frame = cv2.addWeighted(frame, 0.7, road_color, 0.3, 0)

        # 🟢 2. Detection Grid (Green Boxes)
        gw, gh = w // 16, h // 16
        for gy in range(16):
            for gx in range(16):
                conf = obj_map[gy, gx]
                if conf > 0.45: # Sensitivity Threshold
                    cv2.rectangle(frame, (gx*gw, gy*gh), ((gx+1)*gw, (gy+1)*gh), (0, 255, 0), 1)

        # 🟢 3. HUD Text
        cv2.putText(frame, f"V3 TEST: VIDEO 2 | EPOCH 10", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Finished! Test video saved as: {video_out}")

if __name__ == "__main__":
    # --- CHANGE THESE IF NEEDED ---
    INPUT_VID = "/tmp/demo6_work/videoplayback (2).mp4"
    OUTPUT_VID = "/tmp/demo6_work/v3_test_result_video2.mp4"
    MODEL_FILE = "smart_drone_v3_checkpoint_10.pth"

    if os.path.exists(INPUT_VID) and os.path.exists(MODEL_FILE):
        test_on_video(INPUT_VID, OUTPUT_VID, MODEL_FILE)
    else:
        print("❌ Error: Check if video path and model file both exist.")