
import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, List, Tuple

class MetaNeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 45, hidden_size: int = 1024, output_size: int = 15):
        super(MetaNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.output_sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.sigmoid1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.sigmoid2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.output_sigmoid(x)
        return x

class MetaModelPredictor:
    def __init__(self, model_path: str = 'full_meta_model.pth', device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model, self.config = self._load_model(model_path)
        self.model.eval()
        self.thresholds = self.config['optimal_thresholds']
        self.diseases_list = self.config['diseases_list']
        self.no_finding_idx = self.diseases_list.index('No Finding')
    
    def _load_model(self, model_path: str) -> Tuple[nn.Module, Dict]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = MetaNeuralNetwork(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                output_size=config['output_size']
            )
        else:
            config = checkpoint['model_architecture']
            model = MetaNeuralNetwork(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                output_size=config['output_size']
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        model_config = {
            'optimal_thresholds': checkpoint.get('optimal_thresholds', {}),
            'diseases_list': checkpoint.get('diseases_list', []),
            'model_architecture': config
        }
        
        return model, model_config

    # Основной метод для получения предсказаний
    def predict(self, logits_array: np.ndarray, apply_thresholds: bool = True) -> Dict:
        if logits_array.shape[1] != 45:
            raise ValueError(f"Ожидается 45 признаков, получено {logits_array.shape[1]}")
        
        logits_tensor = torch.tensor(logits_array, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            probabilities = self.model(logits_tensor)
            probabilities_np = probabilities.cpu().numpy()
        
        if apply_thresholds:
            binary_predictions = self._apply_thresholds(probabilities_np)
            #binary_predictions = self._apply_no_finding_logic(binary_predictions)
        else:
            binary_predictions = None
        
        return {
            'probabilities': probabilities_np,
            'binary_predictions': binary_predictions,
            'diseases': self.diseases_list
        }
    
    def _apply_thresholds(self, probabilities: np.ndarray) -> np.ndarray:
        binary = np.zeros_like(probabilities)
        for i, disease in enumerate(self.diseases_list):
            threshold = self.thresholds.get(disease, 0.5)
            binary[:, i] = (probabilities[:, i] > threshold).astype(float)
        return binary
    
    def _apply_no_finding_logic(self, predictions: np.ndarray) -> np.ndarray:
        result = predictions.copy()
        no_finding_mask = result[:, self.no_finding_idx] == 1
        result[no_finding_mask, :] = 0
        result[no_finding_mask, self.no_finding_idx] = 1
        return result

    # Предсказание для одного изображения
    def predict_single(self, logits: np.ndarray) -> Dict:
        logits_batch = logits.reshape(1, -1)
        results = self.predict(logits_batch, apply_thresholds=True)
        
        return {
            'probabilities': results['probabilities'][0],
            'binary_predictions': results['binary_predictions'][0],
            'diseases': self.diseases_list,
            'predicted_diseases': [
                self.diseases_list[i] 
                for i in range(len(self.diseases_list)) 
                if results['binary_predictions'][0, i] == 1
            ]
        }

    # Получить оптимальные пороги для каждого класса
    def get_thresholds(self) -> Dict:
        return self.thresholds.copy()

    # Получить список заболеваний в правильном порядке
    def get_diseases_list(self) -> List[str]:
        return self.diseases_list.copy()

#ЗАГРУЗКА МОДЕЛИ
def load_meta_model(model_path: str = 'full_meta_model.pth', device: str = None):
    """Быстрая загрузка мета-модели"""
    return MetaModelPredictor(model_path, device)
