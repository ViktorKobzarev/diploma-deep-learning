
import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

diseases_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
                 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
                 'Pneumonia', 'Pneumothorax']

def get_swin_tiny(num_classes=15, dropout_rate=0.4):
    model = models.swin_t(weights=None)

    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(dropout_rate, inplace=True),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(dropout_rate/2, inplace=True),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(dropout_rate/3, inplace=True),
        nn.Linear(256, num_classes)
    )
    return model

# Создаем модель и загружаем веса
model = get_swin_tiny(num_classes=len(diseases_list))
checkpoint = torch.load('swin_tiny_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Перемещаем модель на устройство и переводим в eval режим
model = model.to(device)
model.eval()

# Трансформации для Swin-Tiny (из checkpoint)
image_size = checkpoint.get('image_size', 512)  # Получаем размер из checkpoint
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from PIL import Image
import numpy as np
from typing import List

def get_logits_from_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)

    return logits.cpu().numpy()

def get_probabilities_from_image(image_path: str) -> np.ndarray:
    logits = get_logits_from_image(image_path)
    return 1 / (1 + np.exp(-logits))

def get_logits_batch(image_paths: List[str], batch_size: int = 8) -> np.ndarray:
    all_logits = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            image_tensor = val_transform(image)
            batch_images.append(image_tensor)

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            batch_logits = model(batch_tensor)

        all_logits.append(batch_logits.cpu().numpy())

    return np.vstack(all_logits) if all_logits else np.array([])

print(f"Модель Swin-Tiny загружена на {device}")
print(f"Размер изображения: {image_size}x{image_size}")
print(f"Val Loss: {checkpoint.get('val_loss', 'N/A')}")

