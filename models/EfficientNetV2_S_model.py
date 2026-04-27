
import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

diseases_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
                 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
                 'Pneumonia', 'Pneumothorax']

def get_efficientnetv2_s(num_classes=15):
    model = models.efficientnet_v2_s(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(in_features, 768),
        nn.BatchNorm1d(768),
        nn.SiLU(inplace=True),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(768, 384),
        nn.BatchNorm1d(384),
        nn.SiLU(inplace=True),
        nn.Dropout(0.133, inplace=True),
        nn.Linear(384, num_classes)
    )
    return model

model = get_efficientnetv2_s(num_classes=len(diseases_list))
checkpoint = torch.load('efficientnetv2_s_final.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"Модель загружена на {device}")

from PIL import Image
import numpy as np
from typing import List

def get_logits_from_image(image_path: str) -> np.ndarray:
    """
    Получает логиты для одного изображения
    
    Параметры:
        image_path: str - путь к изображению
    
    Возвращает:
        np.ndarray [1, 15] - логиты для 15 классов
    """
    # Открываем и преобразуем изображение
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0)  # [1, 3, 512, 512]
    
    # Переносим на устройство
    image_tensor = image_tensor.to(device)
    
    # Получаем предсказания (без вычисления градиентов)
    with torch.no_grad():
        logits = model(image_tensor)
    
    # Преобразуем в numpy и возвращаем
    return logits.cpu().numpy()

def get_probabilities_from_image(image_path: str) -> np.ndarray:
    """
    Получает вероятности (после sigmoid) для одного изображения
    
    Параметры:
        image_path: str - путь к изображению
    
    Возвращает:
        np.ndarray [1, 15] - вероятности для 15 классов
    """
    # Получаем логиты
    logits = get_logits_from_image(image_path)
    
    # Применяем сигмоиду для получения вероятностей
    probabilities = 1 / (1 + np.exp(-logits))
    
    return probabilities

def get_logits_batch(image_paths: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Получает логиты для батча изображений
    
    Параметры:
        image_paths: List[str] - список путей к изображениям
        batch_size: int - размер батча (по умолчанию 32)
    
    Возвращает:
        np.ndarray [N, 15] - логиты для N изображений и 15 классов
    """
    all_logits = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Загружаем и преобразуем изображения
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            image_tensor = val_transform(image)
            batch_images.append(image_tensor)
        
        # Создаем батч
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Получаем предсказания
        with torch.no_grad():
            batch_logits = model(batch_tensor)
        
        # Добавляем к результатам
        all_logits.append(batch_logits.cpu().numpy())
    
    # Объединяем все батчи
    return np.vstack(all_logits)