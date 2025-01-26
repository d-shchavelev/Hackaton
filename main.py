import pandas as pd
from perceiver_pytorch import Perceiver
import torch

# Загрузка данных
df = pd.read_csv('data/air-quality-data.csv')

# Вывод доступных параметров
print("Доступные параметры:", df.columns)

# Выбор параметров
features = ['PM10', 'NO2', 'CO', 'SO2']
data = df[features]

print("Форма данных:", data.shape)
print("Первые строки:\n", data.head())
