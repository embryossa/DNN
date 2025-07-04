import matplotlib.pyplot as plt

'''
Plot for predictions spreading
'''

# Построение графика с предсказанными значениями (нейросетевая модель)
plt.figure(figsize=(8, 6))
plt.scatter(range(len(new_predictions_neural_network)), new_predictions_neural_network, label='Предсказанные значения (нейросетевая модель)', marker='o', alpha=0.5)
plt.xlabel('Примеры')
plt.ylabel('Предсказанные значения')
plt.legend()
plt.title('Предсказанные значения для новых данных (нейросетевая модель)')
plt.show()

# Если калиброванная модель была загружена, построение графика с предсказанными значениями (калиброванная модель)
if 'loaded_calibrated_model' in locals():
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(new_predictions_calibrated)), new_predictions_calibrated, label='Предсказанные значения (калиброванная модель)', marker='o', alpha=0.5)
    plt.xlabel('Примеры')
    plt.ylabel('Предсказанные значения')
    plt.legend()
    plt.title('Предсказанные значения для новых данных (калиброванная модель)')
    plt.show()

'''
PR calculations on DNN predictions
'''

new_df['Predicted_Исход переноса_беременность клиническая'] = new_predictions_calibrated
# Порог вероятности для принятия решения
threshold = 0.485

# Замена вероятностей на категории "да" или "нет"
new_df['Predicted_Исход переноса_беременность клиническая'] = new_df['Predicted_Исход переноса_беременность клиническая'].apply(lambda x: "да" if x > threshold else "нет")

# Сохранение датафрейма в файл Excel
new_df.to_excel('предсказания.xlsx', index=False)
pregnancy_yes_count = (new_df['Predicted_Исход переноса_беременность клиническая'] == 'да').sum()
total_count = len(new_df['Predicted_Исход переноса_беременность клиническая'])
pregnancy_frequency = pregnancy_yes_count / total_count
print(f'ЧНБ: {pregnancy_frequency:.2%}')

'''
Plot to compare
'''
import seaborn as sns
import matplotlib.pyplot as plt

x = 0.2222 #predicted
y = 0.25 #actual

plt.figure(figsize=(4, 4))

# Добавляем два полупрозрачных столбца для каждого квартала
plt.bar([0], [x], color='blue', alpha=0.4, label='Predicted')
plt.bar([0.2], [y], color='green', alpha=0.4, label='Actual')

plt.title('Сравнение частоты беременности: Predicted vs Actual')
plt.xticks([])
plt.ylabel('ЧНБ')
plt.legend(loc='lower center')
plt.show()

                                                                                                                                
