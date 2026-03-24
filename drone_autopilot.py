import torch
import cv2
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from train_drone import SmartDroneNet

def process_autopilot(input_path, output_path):
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

    print(f"✈️ Initializing Autopilot System on {device}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. AI Analysis
        img_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            grid_preds = torch.sigmoid(model(img_tensor)).cpu().numpy()[0]

        # 2. Navigation Logic (Focus on rows 4-7: The immediate path)
        # We sum the confidence scores in 3 sectors
        left_sector   = np.sum(grid_preds[0, 4:, 0:3])
        center_sector = np.sum(grid_preds[0, 4:, 3:5])
        right_sector  = np.sum(grid_preds[0, 4:, 5:8])

        # Default Flight Command
        command = "✅ FORWARD"
        color = (0, 255, 0) # Green

        # Collision Avoidance Decisions
        if center_sector > 0.8: # If something is directly in front
            if left_sector < right_sector:
                command = "⬅️ VEER LEFT"
                color = (0, 255, 255) # Yellow
            else:
                command = "➡️ VEER RIGHT"
                color = (0, 255, 255)
            
            if center_sector > 2.5: # Extreme Danger
                command = "⚠️ BRAKE / HOVER"
                color = (0, 0, 255) # Red

        # 3. Visual Overlay (The HUD)
        # Draw Sector Dividers
        cv2.line(frame, (int(width*0.375), height), (int(width*0.375), int(height*0.5)), (255,255,255), 1)
        cv2.line(frame, (int(width*0.625), height), (int(width*0.625), int(height*0.5)), (255,255,255), 1)

        # Draw Boxes for Objects
        for row in range(8):
            for col in range(8):
                if grid_preds[0, row, col] > 0.45:
                    nx, ny, nw, nh, _ = grid_preds[1:, row, col]
                    bx, by = int((col+nx)/8 * width - (nw*width)/2), int((row+ny)/8 * height - (nh*height)/2)
                    cv2.rectangle(frame, (bx, by), (bx+int(nw*width), by+int(nh*height)), (255, 0, 0), 2)

        # Draw The Command Center
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"NAV: {command}", (40, 65), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)
        
        # Add "Radar" stats
        cv2.putText(frame, f"L: {left_sector:.1f} | C: {center_sector:.1f} | R: {right_sector:.1f}", 
                    (width - 450, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Mission Complete. Autopilot video saved as {output_path}")

if __name__ == "__main__":
    process_autopilot("/tmp/demo6_work/videoplayback.mp4", "/tmp/demo6_work/drone_autopilot_v1.mp4")