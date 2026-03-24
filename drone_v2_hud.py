import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import os

# --- 1. ARCHITECTURE V2 (11 Output Channels) ---
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

class SmartDroneNetV2(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.SiLU())
        self.layer1 = SmartDroneBlock(64, 128, False)
        self.layer2 = SmartDroneBlock(128, 256, True)
        self.grid_size = 8
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
            nn.Conv2d(256, 6 + num_classes, 1) 
        )

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.max_pool2d(self.layer1(x), 2)
        x = self.layer2(x)
        return self.detector(x)

# --- 2. MULTI-CLASS INFERENCE ENGINE ---
def run_v2_comparison(input_video, output_video):
    device = torch.device("cuda")
    model = SmartDroneNetV2().to(device)
    
    # Load V2 weights
    if os.path.exists("smart_drone_v2_final.pth"):
        model.load_state_dict(torch.load("smart_drone_v2_final.pth"))
        print("🧠 V2 Intelligence Loaded.")
    else:
        print("❌ Error: smart_drone_v2_final.pth not found!")
        return
    
    model.eval()

    # Configuration
    CLASSES = ["PEDESTRIAN", "CAR", "VAN", "BUS", "TRUCK"]
    # Distinct colors for each class (BGR format)
    COLORS = {
        "PEDESTRIAN": (0, 0, 255),    # Bright Red
        "CAR": (0, 255, 0),           # Bright Green
        "VAN": (255, 255, 0),         # Cyan
        "BUS": (0, 165, 255),         # Orange
        "TRUCK": (255, 0, 255)        # Magenta
    }

    cap = cv2.VideoCapture(input_video)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    print(f"🎬 Processing V2 Video... Running on {device}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img_t = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(img_t)).cpu().numpy()[0]

        for row in range(8):
            for col in range(8):
                obj_conf = preds[0, row, col]
                
                # Only show detections above 40% confidence
                if obj_conf > 0.40:
                    nx, ny, nw, nh = preds[1:5, row, col]
                    
                    # Coordinate conversion
                    cx, cy = (col + nx) / 8.0, (row + ny) / 8.0
                    bw, bh = int(abs(nw) * width), int(abs(nh) * height)
                    bx, by = int(cx * width - bw/2), int(cy * height - bh/2)

                    # Class Selection (The "Argmax" logic)
                    class_probs = preds[6:, row, col]
                    class_idx = np.argmax(class_probs)
                    class_name = CLASSES[class_idx]
                    class_conf = class_probs[class_idx]
                    
                    color = COLORS[class_name]

                    # Draw the HUD
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                    
                    # Draw Label Tag
                    label = f"{class_name} {class_conf*100:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (bx, by - 20), (bx + tw + 10, by), color, -1)
                    cv2.putText(frame, label, (bx + 5, by - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✨ V2 Video Generated: {output_video}")

if __name__ == "__main__":
    run_v2_comparison("/tmp/demo6_work/videoplayback.mp4", "drone_v2_analysis.mp4")