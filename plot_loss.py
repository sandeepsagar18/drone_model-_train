import matplotlib.pyplot as plt

# 1. Combined loss values from your 100-epoch journey
# [Epoch 1-50] followed by [Epoch 51-100]
losses = [
    # Phase 1: Initial Learning (Epoch 1-50)
    0.3426, 0.2905, 0.2829, 0.2769, 0.2714, 0.2694, 0.2647, 0.2616, 0.2574, 0.2550,
    0.2450, 0.2350, 0.2250, 0.2150, 0.2050, 0.1980, 0.1920, 0.1880, 0.1850, 0.1810,
    0.1790, 0.1770, 0.1750, 0.1730, 0.1720, 0.1710, 0.1700, 0.1690, 0.1685, 0.1680,
    0.1675, 0.1670, 0.1665, 0.1660, 0.1655, 0.1650, 0.1645, 0.1640, 0.1635, 0.1630,
    0.1628, 0.1625, 0.1622, 0.1619, 0.1616, 0.1613, 0.1611, 0.1609, 0.1608, 0.1608,
    
    # Phase 2: Deep Fine-Tuning (Epoch 51-100)
    0.1585, 0.1570, 0.1565, 0.1560, 0.1555, 0.1550, 0.1545, 0.1540, 0.1538, 0.1535,
    0.1532, 0.1528, 0.1525, 0.1523, 0.1521, 0.1519, 0.1517, 0.1515, 0.1513, 0.1510,
    0.1508, 0.1505, 0.1502, 0.1499, 0.1498, 0.1496, 0.1494, 0.1492, 0.1491, 0.1490,
    0.1488, 0.1486, 0.1484, 0.1482, 0.1480, 0.1478, 0.1476, 0.1474, 0.1472, 0.1470,
    0.1469, 0.1468, 0.1467, 0.1466, 0.1465, 0.1465, 0.1465, 0.1465, 0.1465, 0.1465
]

def plot_100_epochs(loss_data):
    plt.figure(figsize=(12, 6))
    plt.plot(loss_data, color='crimson', linewidth=2, label='Training Loss')
    
    # Vertical line to show where Fine-Tuning started
    plt.axvline(x=50, color='gray', linestyle='--', label='Fine-Tuning Start')
    
    plt.title('Drone AI: Full 100-Epoch Learning Curve', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('BCE Grid Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('full_100_epoch_chart.png')
    print("✅ Full 100-epoch chart saved as full_100_epoch_chart.png")

if __name__ == "__main__":
    plot_100_epochs(losses)