import torch
import torch.nn as nn

class SmartDroneBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Local path: Fast feature extraction
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        # Global path: Transformer-style Attention (Simplified)
        self.global_query = nn.Conv2d(out_channels, out_channels, 1)
        self.global_key   = nn.Conv2d(out_channels, out_channels, 1)
        self.global_value = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.local_conv(x)
        
        # Self-Attention logic for 'Smart' context
        b, c, h, w = x.shape
        q = self.global_query(x).view(b, c, -1)
        k = self.global_key(x).view(b, c, -1)
        v = self.global_value(x).view(b, c, -1)
        
        # Calculate context (which parts of the image relate to each other)
        attn = torch.matmul(q.transpose(-2, -1), k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        
        context = torch.matmul(v, attn.transpose(-2, -1))
        context = context.view(b, c, h, w)
        
        return x + context # Combine local + global intelligence
        class SmartDroneNet(nn.Module):
    def __init__(self, num_classes=1): # 1 for 'Obstacle'
        super().__init__()
        # Encoder: Gradually shrinking the image while getting 'smarter'
        self.layer1 = SmartDroneBlock(3, 32)   # 3 color channels (RGB) -> 32
        self.layer2 = SmartDroneBlock(32, 64)  # 64 features
        self.pool = nn.MaxPool2d(2, 2)         # Reduces image size
        
        self.layer3 = SmartDroneBlock(64, 128) # Deep reasoning
        
        # Detection Head: Predicts [x, y, width, height, confidence]
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 5) 
        )

    def forward(self, x):
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.layer3(x)
        return self.detector(x)

# Instantiate and move to your L40S GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmartDroneNet().to(device)
print(f"Model initialized on: {device}")


class SmartDroneNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ... (Keep your existing Stem and SmartDroneBlocks) ...
        
        # New: Grid-based detection head (8x8 grid)
        # For each cell, we predict: [Objectness, x, y, w, h, Distance]
        self.grid_size = 8
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size)),
            nn.Conv2d(256, 128, 1),
            nn.SiLU(),
            nn.Conv2d(128, 6, 1) # 6 outputs: Confidence + 4 coords + Distance
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Output shape: [Batch, 6, 8, 8]
        return self.detector(x)