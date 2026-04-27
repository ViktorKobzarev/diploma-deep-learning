
# импортируем модели в ансамбле и дополнительные библиотек
import DenseNet_model
import EfficientNetV2_S_model
import swin_tiny_model

import numpy as np
import pandas as pd

# импортируем мета модель
from meta_model import load_meta_model
predictor = load_meta_model('full_meta_model.pth')

predictor.thresholds['Hernia'] = 0.005 # меняем порог отсечения для класса hernia

# функция для получения истинных заболеваний (если такие данные имеются)
def get_real_diseases(image_path):
    # определяем список с заболеваниями в нужном порядке
    diseases= ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis',
               'Hernia','Infiltration','Mass','No Finding','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']

    train_data = pd.read_csv('3\\train_data.csv')
    teas_data = pd.read_csv('3\\test_data.csv')

    conc = pd.concat([train_data, teas_data], ignore_index=True)
    #print(conc)

    image_ind = image_path[-16:]
    #print(image_ind)

    find_row = conc[conc['Image Index'] == image_ind]
    #print(find_row)
    if find_row.empty:
        return "обьект не найден"

    find_row = find_row.iloc[:, -15:]
    find_row = np.array(find_row)
    #print(find_row)

    real_diseases = []
    for i in range(len(diseases)):
        if find_row[0][i] == 1:
            real_diseases.append(diseases[i])    

    #print(type(find_row))
    return real_diseases



# функция для получения предсказания одного наблюдения
def get_prediction(image_path):
    # получаем логиты с каждой модели в ансамбле
    dn_logits = DenseNet_model.get_logits_from_image(image_path)
    en_logits = EfficientNetV2_S_model.get_logits_from_image(image_path)
    st_logits = swin_tiny_model.get_logits_from_image(image_path)

    # соединяем все логиты в 1 массив
    all_logits = np.concatenate([dn_logits, en_logits, st_logits], axis = 1)

    # прогоняем логиты через мета модель
    result = predictor.predict(all_logits)

    # получаем ответ в формате названия болезней
    predicted_diseases = []
    for i in range(15):
        if result['binary_predictions'][0][i] == 1:
            predicted_diseases.append(result['diseases'][i])

    return [predicted_diseases, result]

# функция для получения предсказаний множества наблюдений
def get_predictions_batch(image_path):
    # получаем логиты с каждой модели в ансамбле
    dn_logits = DenseNet_model.get_logits_batch(image_path)
    en_logits = EfficientNetV2_S_model.get_logits_batch(image_path)
    st_logits = swin_tiny_model.get_logits_batch(image_path)

    # соединяем все логиты в 1 массив
    all_logits = np.concatenate([dn_logits, en_logits, st_logits], axis = 1)

    # прогоняем логиты через мета модель
    result = predictor.predict(all_logits)

    # получаем ответ в формате названия болезней
    predicted_diseases = []

    for row in range(len(result['binary_predictions'])):
        row_diseases = []
        curr_row = result['binary_predictions'][row]
        for dis in range(15):
            if curr_row[dis] == 1:
                row_diseases.append(result['diseases'][dis])

        predicted_diseases.append(row_diseases)
    return [predicted_diseases, result]
