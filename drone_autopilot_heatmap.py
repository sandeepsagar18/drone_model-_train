import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from train_drone import SmartDroneNet

def process_autopilot_with_heatmap(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNet().to(device)
    model.load_state_dict(torch.load("smart_drone_final.pth"))
    model.eval()

    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    print(f"📡 System Online. Processing with Heatmap HUD on {device}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. AI Analysis
        img_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            # grid_preds shape is [1, 6, 8, 8]
            grid_preds = torch.sigmoid(model(img_tensor)).cpu().numpy()[0]

        # 2. Steering Logic
        left_sector   = np.sum(grid_preds[0, 4:, 0:3])
        center_sector = np.sum(grid_preds[0, 4:, 3:5])
        right_sector  = np.sum(grid_preds[0, 4:, 5:8])

        command = "✅ FORWARD"
        color = (0, 255, 0)

        if center_sector > 0.8:
            if left_sector < right_sector:
                command = "⬅️ VEER LEFT"
                color = (0, 255, 255)
            else:
                command = "➡️ VEER RIGHT"
                color = (0, 255, 255)
            if center_sector > 2.5:
                command = "⚠️ BRAKE / HOVER"
                color = (0, 0, 255)

        # 3. Create Heatmap Overlay
        # Extract the confidence channel (8x8)
        conf_map = grid_preds[0, :, :]
        # Scale to 0-255 and resize for visibility
        heatmap = cv2.resize(conf_map, (160, 160))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Place heatmap in bottom-right corner with a border
        h_offset, w_offset = height - 180, width - 180
        roi = frame[h_offset:h_offset+160, w_offset:w_offset+160]
        # Blend the heatmap with the original frame pixels (50% transparency)
        merged = cv2.addWeighted(roi, 0.5, heatmap_color, 0.5, 0)
        frame[h_offset:h_offset+160, w_offset:w_offset+160] = merged
        cv2.rectangle(frame, (w_offset, h_offset), (w_offset+160, h_offset+160), (255,255,255), 2)
        cv2.putText(frame, "AI ATTENTION", (w_offset, h_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 4. Standard HUD Overlays (Dividers & Command)
        cv2.line(frame, (int(width*0.375), height), (int(width*0.375), int(height*0.6)), (255,255,255), 1)
        cv2.line(frame, (int(width*0.625), height), (int(width*0.625), int(height*0.6)), (255,255,255), 1)

        # Top Bar Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"NAV: {command}", (30, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        cv2.putText(frame, f"L: {left_sector:.1f} | C: {center_sector:.1f} | R: {right_sector:.1f}", 
                    (width - 400, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"🎬 Mission Accomplished. Video saved: {output_path}")

if __name__ == "__main__":
    process_autopilot_with_heatmap("/tmp/demo6_work/videoplayback.mp4", "/tmp/demo6_work/drone_autopilot_heatmap.mp4")