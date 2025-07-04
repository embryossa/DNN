import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_selected = new_df_with_KPI.xlsx

from sklearn.model_selection import train_test_split
selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl хор.кач-ва", "Частота оплодотворения", "Число Bl",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "Перенесено эмбрионов",
    "KPIScore"
]

# Загрузка данных из DataFrame и разделение на признаки и целевую переменную
X = df_selected[selected_features]
y = df_selected['Исход переноса']

# Разделение на обучающий, валидационный и тестовый наборы
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

# Создание экземпляра класса StandardScaler
scaler = StandardScaler()

# Нормализация данных обучающего и тестового набора
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Изменение формата данных для обучения модели
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

X_train_reshaped = X_train_normalized.reshape((X_train_normalized.shape[0], X_train_normalized.shape[1], 1))
X_val_reshaped = X_val_normalized.reshape((X_val_normalized.shape[0], X_val_normalized.shape[1], 1))
X_test_reshaped = X_test_normalized.reshape((X_test_normalized.shape[0], X_test_normalized.shape[1], 1))
