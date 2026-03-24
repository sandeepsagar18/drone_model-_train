import torch
import cv2
import numpy as np
import time
from PIL import Image
from torchvision import transforms
from train_drone import SmartDroneNet

def process_drone_video(input_path, output_path):
    # 1. Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    model = SmartDroneNet().to(device)
    model.load_state_dict(torch.load("smart_drone_final.pth"))
    model.eval()

    # 2. Setup Video Stream
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    print(f"🎬 Processing {total_frames} frames... Please wait.")
    
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A. Pre-process Frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

        # B. AI Inference
        with torch.no_grad():
            logits = model(img_tensor)
            grid_preds = torch.sigmoid(logits).cpu().numpy()[0]

        # C. Draw HUD Overlays
        obstacles_count = 0
        for row in range(8):
            for col in range(8):
                conf = grid_preds[0, row, col]
                
                # Confidence threshold
                if conf > 0.45:
                    nx, ny, nw, nh, _ = grid_preds[1:, row, col]
                    
                    # Coordinate Calculation
                    center_x, center_y = (col + nx) / 8.0, (row + ny) / 8.0
                    bw, bh = int(abs(nw) * width), int(abs(nh) * height)
                    bx, by = int(center_x * width - bw/2), int(center_y * height - bh/2)

                    # Danger Logic (Distance estimation)
                    is_danger = (by + bh) > (height * 0.75) or nh > 0.25
                    color = (0, 0, 255) if is_danger else (0, 255, 0)
                    
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                    obstacles_count += 1

        # D. Add On-Screen Display (OSD)
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(frame, f"AI FPS: {current_fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"OBSTACLES: {obstacles_count}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Write Frame
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"✅ Processed {frame_count}/{total_frames} frames...")

    # Cleanup
    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(f"\n✨ FINISHED!")
    print(f"📁 Processed video saved as: {output_path}")
    print(f"🚀 Average Processing Speed: {frame_count / total_time:.2f} FPS")

if __name__ == "__main__":
    VIDEO_PATH = "/tmp/demo6_work/videoplayback.mp4"
    OUTPUT_PATH = "/tmp/demo6_work/drone_ai_final_output.mp4"
    process_drone_video(VIDEO_PATH, OUTPUT_PATH)