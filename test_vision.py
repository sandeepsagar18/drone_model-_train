import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from train_drone import SmartDroneNet 

def test_multi_detection(img_path):
    device = torch.device("cuda")
    
    # 1. Load the "Deep Trained" Brain
    model = SmartDroneNet().to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("smart_drone_final.pth"))
    else:
        model.load_state_dict(torch.load("smart_drone_final.pth", map_location=torch.device('cpu')))
    model.eval()

    # 2. Prepare Image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error: Could not find image at {img_path}")
        return
    
    h_orig, w_orig, _ = img.shape
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Convert BGR to RGB for the model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)

    # 3. Predict
    with torch.no_grad():
        logits = model(img_tensor)
        # Use sigmoid because we used BCEWithLogitsLoss during training
        grid_preds = torch.sigmoid(logits).cpu().numpy()[0] 

    print("🧠 Analyzing scene for obstacles...")

    # 4. Filter and Draw
    for row in range(8):
        for col in range(8):
            conf = grid_preds[0, row, col]
            
            # Raise threshold to 0.45 to be more "picky" and reduce noise
            if conf > 0.45: 
                nx, ny, nw, nh, prox = grid_preds[1:, row, col]
                
                # GRID-CENTRIC SCALING:
                # We calculate the position based on the grid cell [col, row]
                # to prevent all boxes from drifting to the top-left.
                center_x = (col + nx) / 8.0
                center_y = (row + ny) / 8.0
                
                # Rescale to original image size
                bw, bh = int(abs(nw) * w_orig), int(abs(nh) * h_orig)
                bx = int(center_x * w_orig - bw/2)
                by = int(center_y * h_orig - bh/2)

                # DANGER LOGIC: 
                # 1. If box is near bottom of screen (Drone's nose)
                # 2. If box is very large (Taking up > 25% of frame)
                is_danger = (by + bh) > (h_orig * 0.75) or nh > 0.25
                
                color = (0, 0, 255) if is_danger else (0, 255, 0) # BGR
                status = "DANGER" if is_danger else "SAFE"
                
                # Draw Box and Label
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
                label = f"{status} {conf:.2f}"
                cv2.putText(img, label, (bx, by - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 5. Save final HUD
    cv2.imwrite("refined_drone_vision.jpg", img)
    print("✅ Success! Refined vision saved as refined_drone_vision.jpg")

if __name__ == "__main__":
    # Ensure this path matches your unzipped VisDrone folder
    test_path = "/tmp/demo6_work/data/VisDrone2019-DET-train/images/0000002_00005_d_0000014.jpg"
    test_multi_detection(test_path)