import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import os

# --- 1. ARCHITECTURE V3 (16x16 Grid) ---
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

# --- 2. V3 SCENE INFERENCE ENGINE ---
def run_v3_hud(input_path, output_path):
    device = torch.device("cuda")
    model = SmartDroneNetV3().to(device)
    
    # Try loading the final model, or fall back to the checkpoint
    weights = "smart_drone_v3_final.pth" if os.path.exists("smart_drone_v3_final.pth") else "smart_drone_v3_checkpoint_10.pth"
    model.load_state_dict(torch.load(weights))
    model.eval()
    print(f"🧠 Scene Intelligence Loaded from: {weights}")

    CLASSES = ["PED", "PERSON", "BIKE", "CAR", "VAN", "TRUCK", "TRIKE", "AWNING", "BUS", "MOTOR"]
    
    cap = cv2.VideoCapture(input_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img_t = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(img_t)).cpu().numpy()[0]

        # --- Scene Overlay Layer ---
        overlay = frame.copy()
        
        for r in range(16):
            for c in range(16):
                # 1. Road and Tree tinting (Segmentation logic)
                tree_score = preds[16, r, c]
                road_score = preds[17, r, c]
                
                gx, gy = int(c * (w/16)), int(r * (h/16))
                gw, gh = int(w/16), int(h/16)

                if tree_score > 0.5: # Tint Trees GREEN
                    cv2.rectangle(overlay, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), -1)
                elif road_score > 0.5: # Tint Roads BLUE/GRAY
                    cv2.rectangle(overlay, (gx, gy), (gx+gw, gy+gh), (255, 100, 0), -1)

                # 2. Object Detections (Car, Ped, etc.)
                obj_conf = preds[0, r, c]
                if obj_conf > 0.45:
                    nx, ny, nw, nh = preds[1:5, r, c]
                    cx, cy = (c + nx) / 16.0, (r + ny) / 16.0
                    bw, bh = int(abs(nw) * w), int(abs(nh) * h)
                    bx, by = int(cx * w - bw/2), int(cy * h - bh/2)
                    
                    class_idx = np.argmax(preds[6:16, r, c])
                    label = CLASSES[class_idx]
                    
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                    cv2.putText(frame, label, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Blend the tints into the main frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        out.write(frame)

    cap.release()
    out.release()
    print(f"✨ V3 Analysis Complete! Saved as: {output_path}")

if __name__ == "__main__":
    # Note the quotes for the filename with a space
    run_v3_hud("/tmp/demo6_work/videoplayback (1).mp4", "drone_v3_final_analysis.mp4")