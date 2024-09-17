import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Функция для построения графиков бета-распределения
def plot_beta(alpha, beta, label):
    x = np.linspace(0, 1, 100)
    y = stats.beta.pdf(x, alpha, beta)
    plt.plot(x, y, label=label)


# Исходные данные
real_successes = [19, 18, 20, 6, 13, 12, 19, 22, 25]  # клинические беременности всего
real_trials = [43, 45, 65, 18, 26, 31, 47, 49, 58]  # переносы всего
predicted_success_from_nn = 0.2931  # предсказанная вероятность успеха от нейросети для нового цикла

# Предварительное убеждение о частоте успеха (напр., основанное на исторических данных)
prior_alpha = 26  # успехи
prior_beta = 74  # неудачи

# Обновление апостериорного распределения с учетом всех данных
for i in range(len(real_successes)):
    # Обновление апостериорного распределения на основе реальных данных
    posterior_alpha = prior_alpha + real_successes[i]
    posterior_beta = prior_beta + (real_trials[i] - real_successes[i])

    # Обновление параметров для следующих итераций
    prior_alpha, prior_beta = posterior_alpha, posterior_beta

# Добавление предсказанной вероятности от нейросети для нового цикла как нового наблюдения
posterior_alpha = prior_alpha + int(predicted_success_from_nn * 100)
posterior_beta = prior_beta + (100 - int(predicted_success_from_nn * 100))

# Предсказание успеха для следующего цикла IVF
pred_success_rate_next_cycle = posterior_alpha / (posterior_alpha + posterior_beta)

# Вычисление доверительного интервала для предсказания успеха следующего цикла IVF
credible_interval = stats.beta.interval(0.95, posterior_alpha, posterior_beta)

# Визуализация апостериорного распределения для итогового предсказания
plt.figure(figsize=(10, 6))
plot_beta(posterior_alpha, posterior_beta, f"Posterior: {pred_success_rate_next_cycle:.2f}")

# Добавление заголовка и подписей к осям на графике
plt.title("Pregnancy rate forecasting probability")
plt.xlabel("Probability")
plt.ylabel("Density")
plt.legend()

# Добавление доверительного интервала на график
plt.axvline(x=credible_interval[0], color='gray', linestyle='--',
            label=f'95% CI ({credible_interval[0]:.2f}, {credible_interval[1]:.2f})')
plt.axvline(x=credible_interval[1], color='gray', linestyle='--')
plt.legend()
plt.show()

# Вывод предсказания успеха для следующего цикла с доверительным интервалом
print(f"Предсказанная вероятность успеха для следующего цикла: {pred_success_rate_next_cycle:.2f}")
print(f"95% доверительный интервал: ({credible_interval[0]:.2f}, {credible_interval[1]:.2f})")
