import pandas as pd
from perceiver_pytorch import Perceiver
import torch

# Загрузка данных
df = pd.read_csv('city_day.csv')  # Новое имя файла

# Проверка данных
print("Доступные параметры:", df.columns)

# Выбор нужных параметров
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
data = df[features].fillna(0)  # Заполняем пропуски нулями

print("\nСтатистика по данным:")
print(data.describe())
