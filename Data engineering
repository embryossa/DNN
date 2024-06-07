import pandas as pd
import numpy as np

# Загрузка данных из файлов Excel
df = pd.read_excel('fresh.xlsx')
cryo_df = pd.read_excel('cryo.xlsx')

# Добавление новых столбцов в DataFrame
df['Частота оплодотворения'] = df['2 pN'] / df['Число инсеминированных']
df['Частота дробления'] = df['Число дробящихся на 3 день'] / df['2 pN']
df['Частота формирования бластоцист'] = df['Число Bl'] / df['2 pN']
df['Частота формирования бластоцист хорошего качества'] = df['Число Bl хор.кач-ва'] / df['2 pN']
df['Частота получения ОКК'] = df['Число ОКК'] / df['Количество фолликулов']

# Преобразование столбцов в числовой формат, если они не содержат числовые значения
columns_to_convert = ['День переноса']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Выбор только нужных признаков для объединения
merge_columns = ['Дата пункции', '№ карты', 'ФИО']

# Объединение данных по ключевым столбцам: 'Дата пункции', '№ карты', 'ФИО'
merged_df = pd.merge(df, cryo_df[['Исход переноса'] + merge_columns], on=merge_columns, how='left',
                     suffixes=('_fresh', '_cryo'))

# Замена значений в столбце 'Исход переноса_fresh' значениями из 'Исход переноса_cryo' при условии отсутствия в 'fresh'
mask = (merged_df['Исход переноса_fresh'].isnull()) & (merged_df['Исход переноса_cryo'].notnull())
merged_df.loc[mask, 'Исход переноса_fresh'] = merged_df.loc[mask, 'Исход переноса_cryo']

# Оставить только выбранные признаки
selected_features = [
    "№ карты", "ФИО", "Дата пункции",
    "Возраст", "№ попытки", "Количество фолликулов", "Число ОКК",
    "Число инсеминированных", "2 pN", "Число дробящихся на 3 день",
    "Число Bl", "Число Bl хор.кач-ва", "Частота оплодотворения",
    "Частота дробления", "Частота формирования бластоцист",
    "Частота формирования бластоцист хорошего качества", "Частота получения ОКК",
    "Число эмбрионов 5 дня", "Заморожено эмбрионов", "День переноса", "Перенесено эмбрионов",
    "Исход переноса_fresh"
]

df_selected = merged_df[selected_features]

# Обновление столбца 'Исход переноса' для задания бинарной классификации
df_selected['Исход переноса'] = df_selected['Исход переноса_fresh'].apply(
    lambda x: 1 if x == 'беременность клиническая' else 0)

# Удаление временных столбцов
df_selected.drop(['Исход переноса_fresh', 'Исход переноса_cryo'], axis=1, inplace=True, errors='ignore')


# Создаем функцию для расчета баллов KPIScore
def calculate_kpi_score(row):
    """
    Calculates the KPI score for a given row of data.

    Parameters:
    row (pandas.Series): A row of data containing columns: 'Возраст', 'Количество фолликулов', 'Число инсеминированных', 'Частота оплодотворения', 'Число Bl хор.кач-ва'.

    Returns:
    int: The calculated KPI score.
    """
    kpi_score = 0

    # Conditions for the 'Возраст' column
    if row['Возраст'] >= 40:
        kpi_score += 1
    elif row['Возраст'] <= 36:
        kpi_score += 5
    else:
        kpi_score += 3

    # Conditions for the 'Количество фолликулов' column
    if row['Количество фолликулов'] > 15:
        kpi_score += 5
    elif 8 <= row['Количество фолликулов'] <= 15:
        kpi_score += 3
    else:
        kpi_score += 1

    # Conditions for the 'Число инсеминированных' column
    if row['Число инсеминированных'] <= 3:
        kpi_score += 1
    elif 4 <= row['Число инсеминированных'] <= 7:
        kpi_score += 3
    else:
        kpi_score += 5

    # Conditions for the 'Частота оплодотворения' column
    if row['Частота оплодотворения'] < 0.5:
        kpi_score += 1
    elif row['Частота оплодотворения'] <= 0.65:
        kpi_score += 3
    else:
        kpi_score += 5

    # Conditions for the 'Число Bl хор.кач-ва' column
    if row['Число Bl хор.кач-ва'] == 0:
        kpi_score += 1
    elif row['Число Bl хор.кач-ва'] <= 2:
        kpi_score += 3
    else:
        kpi_score += 5

    return kpi_score


# Применяем функцию для создания столбца "KPIScore"
df_selected['KPIScore'] = df_selected.apply(calculate_kpi_score, axis=1)

# Замена отсутствующих значений на 0 в выбранных признаках
df_selected.fillna(0, inplace=True)

# Заменить значения, вызывающие проблемы, на NaN
df_selected.replace([np.inf, -np.inf], np.nan, inplace=True)

# Удалить строки, содержащие NaN
df_selected.dropna(inplace=True)

# Выводим новый датафрейм с новым столбцом KPIScore
print(df_selected)

# Сохраняем датафрейм с новым столбцом KPIScore
df_selected.to_excel('new_df_with_KPI.xlsx', index=False)
