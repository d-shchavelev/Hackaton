import pandas as pd
from perceiver_pytorch import Perceiver
import torch

# Загрузка данных
df = pd.read_csv('качество воздуха-india.csv')

# Проверка данных
print("Доступные параметры:", df.columns)
features = ['PM10', 'NO2', 'CO', 'SO2']
data = df[features]
print("\nПервые 5 строк:\n", data.head())
