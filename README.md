# drone_model_train
# Aegis Drone System v3 🛸

A high-performance drone management and autonomous flight system featuring real-time telemetry, AI-driven object detection, and a custom Heads-Up Display (HUD).

## 🚀 Key Features
- **Autonomous Autopilot**: Intelligent flight path logic and mission planning.
- **Computer Vision**: Integrated object detection using the VisDrone dataset and PyTorch.
- **Real-time Telemetry**: Monitoring of drone health, position, and mission status.
- **Interactive HUD**: A custom web-based interface (`aegis_hud.html`) for pilot visualization.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **AI/ML**: PyTorch (Model Training & Inference)
- **Computer Vision**: OpenCV
- **Frontend**: HTML5/CSS3 (Vite + React integration)
- **Environment**: Linux / SSH Remote Compute

## 📂 Project Structure
```text
├── aegis_v3_system/       # Core backend and frontend logic
│   ├── backend/           # API and flight control scripts
│   └── frontend/          # HUD and UI assets
├── drone_autopilot.py     # Main autonomous flight logic
├── train_drone_v2.py      # AI model training script
├── test_telemetry.py      # System diagnostic tools
└── requirements.txt       # Python dependencies
