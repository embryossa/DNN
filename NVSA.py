import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import re

# Загрузка модели нейросети
loaded_neural_network_model = load_model('neural_network_model.h5')

# Загрузка калиброванной модели
loaded_calibrated_model = joblib.load('calibrated_model.pkl')

# Загрузка данных из Excel-файла в DataFrame
new_df = pd.read_excel('сентябрь.xlsx')

# Функция для извлечения числовых значений из строки
def extract_number(s):
    if isinstance(s, str):
        match = re.search(r'\d+', s)
        return int(match.group()) if match else None
    return None

# Преобразование столбцов в числовой формат
columns_to_convert = ['Возраст', 'День переноса']
for col in columns_to_convert:
    new_df[col] = new_df[col].astype(str).apply(extract_number)

# Вычисление дополнительных параметров
new_df['Частота оплодотворения'] = new_df['2 pN'] / new_df['Число инсеминированных']
new_df['Частота дробления'] = new_df['Число дробящихся на 3 день'] / new_df['2 pN']
new_df['Частота формирования бластоцист'] = new_df['Число Bl'] / new_df['2 pN']
new_df['Частота формирования бластоцист хорошего качества'] = new_df['Число Bl хор.кач-ва'] / new_df['2 pN']
new_df['Частота получения ОКК'] = new_df['Число ОКК'] / new_df['Количество фолликулов']
# Создаем функцию для расчета KPIScore
def calculate_kpi_score(row):
    kpi_score = 0
    # Условия для столбца "Возраст"
    if row['Возраст'] >= 40:
        kpi_score += 1
    elif row['Возраст'] <= 36:
        kpi_score += 5
    else:
        kpi_score += 3
    # Условия для столбца "Количество фолликулов"
    if row['Количество фолликулов'] > 15:
        kpi_score += 5
    elif 8 <= row['Количество фолликулов'] <= 15:
        kpi_score += 3
    else:
        kpi_score += 1
    # Условия для столбца "Число инсеминированных"
    if row['Число инсеминированных'] <= 3:
        kpi_score += 1
    elif 4 <= row['Число инсеминированных'] <= 7:
        kpi_score += 3
    else:
        kpi_score += 5
    # Условия для столбца "Частота оплодотворения"
    if row['Частота оплодотворения'] < 0.5:
        kpi_score += 1
    elif row['Частота оплодотворения'] <= 0.65:
        kpi_score += 3
    else:
        kpi_score += 5
    # Условия для столбца "Число Bl хор.кач-ва"
    if row['Число Bl хор.кач-ва'] == 0:
        kpi_score += 1
    elif row['Число Bl хор.кач-ва'] <= 2:
        kpi_score += 3
    else:
        kpi_score += 5
    return kpi_score


new_df['KPIScore'] = new_df.apply(calculate_kpi_score, axis=1)
# Замена отсутствующих значений на 0 в датафрейме new_df
new_df.fillna(0, inplace=True)

# Выбор признаков для прогнозирования
selected_features = [
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl хор.кач-ва", "Частота оплодотворения", "Число Bl",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено бластоцист", "Перенесено эмбрионов",
    "KPIScore"
]

# Создание экземпляра класса StandardScaler
scaler = StandardScaler()
# Нормализация данных
new_df_normalized = scaler.fit_transform(new_df[selected_features])

# Предсказание значений с использованием нейросетевой модели
new_predictions_nn = loaded_neural_network_model.predict(new_df_normalized)

# Функция для расчета KPIScore
def calculate_kpi_score(features):
    age, num_attempts, follicle_count, number_of_okk, inseminated_count, pN_2, num_day_3, good_quality_blastocysts, frequency_of_fertilization, num_bl, frequency_of_division, frequency_of_blastocyst_formation, frequency_of_good_quality_blastocyst, frequency_of_okk, num_day_5, frozen_blastocysts, transferred_embryos, kpi_score = features

    kpi_score = 0
    if age >= 40:
        kpi_score += 1
    elif age <= 36:
        kpi_score += 5
    else:
        kpi_score += 3

    if follicle_count > 15:
        kpi_score += 5
    elif 8 <= follicle_count <= 15:
        kpi_score += 3
    else:
        kpi_score += 1

    if inseminated_count <= 3:
        kpi_score += 1
    elif 4 <= inseminated_count <= 7:
        kpi_score += 3
    else:
        kpi_score += 5

    if frequency_of_fertilization < 0.5:
        kpi_score += 1
    elif frequency_of_fertilization <= 0.65:
        kpi_score += 3
    else:
        kpi_score += 5

    if good_quality_blastocysts == 0:
        kpi_score += 1
    elif good_quality_blastocysts <= 2:
        kpi_score += 3
    else:
        kpi_score += 5

    return kpi_score

# Применение символьных правил к предсказаниям нейросетевой модели
def apply_symbolic_rules(features, predictions):
    results = []
    for feature_set, prediction in zip(features, predictions):
        kpi_score = calculate_kpi_score(feature_set)
        result = {
            'Features': feature_set,
            'Prediction': prediction,
            'KPI Score': kpi_score,
            'Adjusted Prediction': prediction * (kpi_score / 17)  # Пример корректировки предсказания на основе KPI
        }
        results.append(result)
    return pd.DataFrame(results)

# Подготовка данных для символьного анализа
feature_data = new_df[selected_features].values
symbolic_results = apply_symbolic_rules(feature_data, new_predictions_nn)

# Сохранение результатов в Excel-файл
symbolic_results.to_excel('symbolic_results.xlsx', index=False)

# Вывод первых нескольких строк результатов
print(symbolic_results.head())
