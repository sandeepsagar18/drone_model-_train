import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class VisDroneDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'annotations')
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt'))
        
        image = Image.open(img_path).convert("RGB")
        
        # VisDrone labels: [xmin, ymin, w, h, score, class, ...]
        # We'll grab the first obstacle as the 'Target' for your self-protection test
        target = torch.zeros(5) 
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                first_line = f.readline().strip().split(',')
                if len(first_line) >= 4:
                    target = torch.tensor([float(x) for x in first_line[:5]])

        if self.transform:
            image = self.transform(image)
            
        return image, target