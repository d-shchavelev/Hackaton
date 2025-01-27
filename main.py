import pandas as pd
import torch
from perceiver_pytorch import Perceiver
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc

class AirQualityPredictor:
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length #Данные последовательности
        self.features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3'] #Параметры загрязнения 
        self.scaler = StandardScaler() #Инструмент для нормализации данных
        self.model = None
        
    def prepare_data(self, df):
        # Предобработка данных
        df = df.copy()  # Создаем копию для избежания предупреждений
        df = df.tail(1000)  # Берем последние 1000 записей
        
        # Обработка дат
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Заполнение пропущенных значений и удаление бесконечностей
        data = df[self.features].replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Проверка на отсутствие NaN значений
        if data.isnull().any().any():
            raise ValueError("После обработки остались NaN значения")
            
        # Нормализация данных
        normalized = self.scaler.fit_transform(data)
        
        # Создание последовательностей с проверкой размерности
        X, y = [], []
        for i in range(len(normalized) - self.sequence_length):
            seq = normalized[i:(i + self.sequence_length)]
            target = normalized[i + self.sequence_length]
            
            if not np.any(np.isnan(seq)) and not np.any(np.isnan(target)):
                X.append(seq)
                y.append(target)
        
        # Преобразование в numpy массивы перед созданием тензоров
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Проверка данных перед созданием тензоров
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Обнаружены NaN значения в данных")
            
        return torch.from_numpy(X), torch.from_numpy(y)
    
    def create_model(self):
        self.model = Perceiver(
            input_channels = len(self.features),
            input_axis = 1,
            num_freq_bands = 4,
            max_freq = 8,
            depth = 4,
            num_latents = 32, 
            latent_dim = 64,   
            cross_heads = 1,
            latent_heads = 4,
            cross_dim_head = 32,
            latent_dim_head = 32,
            num_classes = len(self.features)
        )
        return self.model
    
    def train(self, X_train, y_train, epochs=30, lr=0.0001, batch_size=16):  # Уменьшили learning rate
        if self.model is None:
            self.create_model()
        
        # Использование оптимизатора с градиентным клиппингом
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        
        # Разбиваем данные на батчи
        num_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Добавляем проверку на NaN в входных данных
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    print(f"NaN обнаружен в батче {i}")
                    continue
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Проверка потерь на NaN
                if torch.isnan(loss):
                    print(f"NaN loss на эпохе {epoch+1}, батч {i}")
                    continue
                
                loss.backward()
                
                # Градиентный клиппинг
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                del outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            avg_loss = total_loss / num_batches
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
            # Остановка обучения при слишком больших потерях
            if avg_loss > 1e6:
                print("Потери слишком велики, остановка обучения")
                break
    
    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.from_numpy(sequence).float()
            if sequence_tensor.dim() == 2:
                sequence_tensor = sequence_tensor.unsqueeze(0)
            
            prediction = self.model(sequence_tensor)
            
            # Проверка на NaN в предсказании
            if torch.isnan(prediction).any():
                print("Предупреждение: NaN в предсказании")
                return {feature: float('nan') for feature in self.features}
            
            prediction_denorm = self.scaler.inverse_transform(prediction.numpy())
            return {feature: value for feature, value in zip(self.features, prediction_denorm[0])}

if __name__ == "__main__":
    try:
        print("Загрузка данных...")
        df = pd.read_csv('city_day.csv')
        
        print("Подготовка данных...")
        predictor = AirQualityPredictor(sequence_length=7)
        
        # Добавляем обработку ошибок при подготовке данных
        try:
            X, y = predictor.prepare_data(df)
        except ValueError as e:
            print(f"Ошибка при подготовке данных: {e}")
            exit(1)
            
        print(f"Размер данных: {X.shape}")
        
        if torch.isnan(X).any() or torch.isnan(y).any():
            print("Ошибка: NaN значения в данных после подготовки")
            exit(1)
        
        print("Разделение данных...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Начало обучения модели...")
        predictor.train(X_train, y_train, epochs=30, batch_size=16)
        
        print("\nПолучение прогноза...")
        latest_sequence = X_test[-1].numpy()
        predictions = predictor.predict(latest_sequence)
        
        print("\nПрогноз на следующие 24 часа:")
        for feature, value in predictions.items():
            if np.isnan(value):
                print(f"{feature}: Нет данных")
            else:
                print(f"{feature}: {value:.2f}")
            
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        
    finally:
        if 'predictor' in locals() and predictor.model is not None:
            del predictor.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
