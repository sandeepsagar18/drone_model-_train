import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from train_drone import SmartDroneNet

def process_final_mission(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartDroneNet().to(device)
    model.load_state_dict(torch.load("smart_drone_final.pth"))
    model.eval()

    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    # --- Mission Statistics ---
    stats = {
        "total_obstacles": 0,
        "avoidance_maneuvers": 0,
        "emergency_brakes": 0,
        "max_confidence": 0.0
    }

    print(f"🚀 Launching Final Mission with 100-Epoch Brain...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. AI Inference
        img_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            grid_preds = torch.sigmoid(model(img_tensor)).cpu().numpy()[0]

        # 2. Navigation & Stats Logic
        left_val   = np.sum(grid_preds[0, 4:, 0:3])
        center_val = np.sum(grid_preds[0, 4:, 3:5])
        right_val  = np.sum(grid_preds[0, 4:, 5:8])
        
        frame_obs = np.sum(grid_preds[0] > 0.45)
        stats["total_obstacles"] = max(stats["total_obstacles"], frame_obs)
        stats["max_confidence"] = max(stats["max_confidence"], np.max(grid_preds[0]))

        command = "FORWARD"
        color = (0, 255, 0)

        if center_val > 0.8:
            stats["avoidance_maneuvers"] += 1
            command = "VEER LEFT" if left_val < right_val else "VEER RIGHT"
            color = (0, 255, 255)
            if center_val > 2.5:
                stats["emergency_brakes"] += 1
                command = "EMERGENCY BRAKE"
                color = (0, 0, 255)

        # 3. HUD Overlay
        cv2.rectangle(frame, (20, 20), (450, 110), (0,0,0), -1)
        cv2.putText(frame, f"AI PILOT: {command}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"DANGER LEVEL: {center_val:.2f}", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        out.write(frame)

    # 4. Generate Mission Summary Screen (Last 3 seconds)
    print("📊 Generating Mission Summary...")
    safety_grade = "A+" if stats["emergency_brakes"] < 5 else "B"
    if stats["emergency_brakes"] > 15: safety_grade = "C"

    summary_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(summary_frame, "MISSION SUMMARY", (width//4, height//4), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 3)
    cv2.putText(summary_frame, f"Safety Grade: {safety_grade}", (width//4, height//4 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(summary_frame, f"Max Obstacles in View: {stats['total_obstacles']}", (width//4, height//4 + 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 1)
    cv2.putText(summary_frame, f"Avoidance Actions: {stats['avoidance_maneuvers'] // fps}", (width//4, height//4 + 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 1)
    cv2.putText(summary_frame, f"Peak AI Confidence: {stats['max_confidence']*100:.1f}%", (width//4, height//4 + 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 1)

    for _ in range(fps * 4): # 4 seconds of summary
        out.write(summary_frame)

    cap.release()
    out.release()
    print(f"✅ Full Mission Video with Summary saved as: {output_path}")

if __name__ == "__main__":
    process_final_mission("/tmp/demo6_work/videoplayback.mp4", "final_mission_report.mp4")