import matplotlib.pyplot as plt
import numpy as np

# Построение графика функции потерь
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.axhline(y=test_loss, color='r', linestyle='--', label='test')
plt.legend()
plt.title('Loss')
plt.show()

'''
# Accuracy и График обучения
'''

# Построение графика точности
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='test')
plt.legend()
plt.title('Accuracy')
plt.show()

'''
#ROC-AUC
'''

from sklearn.metrics import roc_curve, auc

# Получение предсказаний модели на тестовых данных
y_pred = model.predict(X_test_reshaped).ravel()

# Вычисление значений FPR, TPR и порогов
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Вычисление площади под ROC-кривой (AUC)
roc_auc = auc(fpr, tpr)

# Нахождение оптимального порога
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Построение ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, label=f'Optimal Threshold: {optimal_threshold:.2f}')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print(f"Optimal Threshold: {optimal_threshold}")

'''
#Precision, Accuracy
'''

from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score, accuracy_score, precision_score, \
    recall_score, average_precision_score

# Получение предсказаний модели на тестовых данных
y_pred = model.predict(X_test_reshaped).ravel()

# Вычисление значений Precision, Recall и порогов
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred)

# Вычисление средневзвешенного значения Precision (AUC-PR)
average_precision = average_precision_score(y_test, y_pred)

# Нахождение оптимального порога
optimal_pr_idx = np.argmax(precision + recall)
optimal_threshold_pr = thresholds_pr[optimal_pr_idx]

# Построение Precision-Recall кривой
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.plot(recall[optimal_pr_idx], precision[optimal_pr_idx], 'ro', markersize=8,
         label=f'Optimal Threshold: {optimal_threshold_pr:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

print(f"Optimal Threshold (Precision-Recall): {optimal_threshold_pr}")

'''
# Metrix
'''

from sklearn.metrics import confusion_matrix

# Получение предсказаний модели
predictions = model.predict(X_test)

# Оценка результатов модели
conf_matrix = confusion_matrix(y_test, (predictions > 0.22))  # Настройте порог по вашему выбору

# Извлечение значений из матрицы ошибок
TN, FP, FN, TP = conf_matrix.ravel()

# Вычисление метрик
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
OA = (TP + TN) / (TP + TN + FP + FN)
OR = (TP * TN) / (FP * FN)

# Вывод результатов
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("PPV:", PPV)
print("NPV:", NPV)
print("FPR:", FPR)
print("FNR:", FNR)
print("Overall Accuracy:", OA)
print("Odds Ratio:", OR)
