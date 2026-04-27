
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# 1. Определяем Dataset
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, disease_columns, transform=None):
        self.data = pd.read_csv(csv_file)
        self.disease_columns = disease_columns
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['full_path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = self.data.iloc[idx][self.disease_columns].values
        labels = torch.tensor(labels.astype(np.float32))

        return image, labels

# 2. Определяем diseases_list
diseases_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
                 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
                 'Pneumonia', 'Pneumothorax']

# 3. Трансформации для теста
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Создаем val_loader из тестовой выборки (файл test_data.csv)
"""
print("📁 Загружаем тестовые данные...")
val_dataset = ChestXrayDataset(
    csv_file='3//test_data.csv',  # файл с тестовой выборкой
    disease_columns=diseases_list,
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f" Тестовых данных: {len(val_dataset)} изображений")
"""

# 5. Загружаем модель если еще не загружена
#print("\n🧠 Загружаем модель...")
import torchvision.models.densenet
torch.serialization.add_safe_globals([torchvision.models.densenet.DenseNet])

# Укажи правильный путь к файлу модели
model_path = 'full_model_dn_121.pth'  # или 'best_model.pth'
model = torch.load(model_path, weights_only=False)

# 6. Определяем устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}")

# 7. Перемещаем модель на устройство
model = model.to(device)
model.eval()

print(" Модель готова к тестированию")

# 8. Теперь можно запустить код для тестирования
#print("\n" + "=" * 50)
#print(" НАЧИНАЕМ ТЕСТИРОВАНИЕ МОДЕЛИ")
#print("=" * 50)


# функции для получения логитов и вероятностей
def get_logits_from_image(image_path):
    """
    Получает логиты для одного изображения
    
    Args:
        image_path: путь к изображению
    
    Returns:
        np.array [1, 15] - логиты для 15 классов
    """
    # Открываем и преобразуем изображение
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    
    # Переносим на устройство
    image_tensor = image_tensor.to(device)
    
    # Получаем логиты
    with torch.no_grad():
        logits = model(image_tensor)  # [1, 15]
    
    return logits.cpu().numpy()

def get_probabilities_from_image(image_path):
    """
    Получает вероятности (после sigmoid) для одного изображения
    
    Args:
        image_path: путь к изображению
    
    Returns:
        np.array [1, 15] - вероятности для 15 классов
    """
    logits = get_logits_from_image(image_path)
    probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
    return probabilities

#def get_logits_batch(image_paths):
    """
    Получает логиты для нескольких изображений
    
    Args:
        image_paths: список путей к изображениям
    
    Returns:
        np.array [N, 15] - логиты для N изображений
    """
#    batch_tensors = []
    
#    for img_path in image_paths:
#        image = Image.open(img_path).convert('RGB')
#        image_tensor = val_transform(image)
#        batch_tensors.append(image_tensor)
    
    # Собираем в батч
#    batch = torch.stack(batch_tensors).to(device)  # [N, 3, 224, 224]
    
#    with torch.no_grad():
#        logits = model(batch)  # [N, 15]
    
#    return logits.cpu().numpy()


def get_logits_batch(image_paths, batch_size=32):
    """
    Получает логиты для нескольких изображений (обрабатывает батчами)
    
    Args:
        image_paths: список путей к изображениям
        batch_size: размер батча для обработки
    
    Returns:
        np.array [N, 15] - логиты для N изображений
    """
    all_logits = []
    
    # Обрабатываем батчами
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        
        # Загружаем изображения для текущего батча
        for img_path in batch_paths:
            image = Image.open(img_path).convert('RGB')
            image_tensor = val_transform(image)
            batch_tensors.append(image_tensor)
        
        # Собираем батч
        batch = torch.stack(batch_tensors).to(device)
        
        # Получаем логиты
        with torch.no_grad():
            batch_logits = model(batch)
            all_logits.append(batch_logits.cpu().numpy())
        
        # Очищаем память
        del batch, batch_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Прогресс
        if (i // batch_size) % 10 == 0:
            print(f"Обработано {min(i+batch_size, len(image_paths))}/{len(image_paths)} изображений")
    
    # Объединяем все батчи
    return np.vstack(all_logits)
