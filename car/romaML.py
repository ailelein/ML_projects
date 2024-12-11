import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
# Шаг 1: Загрузка данных
data = pd.read_csv('car_data.csv')
print(data)
# Шаг 2: Преобразование категориальных переменных (Color) в числовые
label_encoder = LabelEncoder()
data['Color'] = label_encoder.fit_transform(data['Color'])  # Преобразуем цвета в числа
# Шаг 3: Выбор признаков (Mileage) и целевой переменной (Price)
X = data[['Mileage']]  # Используем только пробег для построения прямой линии
y = data['Price']
# Шаг 4: Разделение данных на обучающую и тестовую выборки (необязательно)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# Шаг 5: Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)
with open('saved_model.pkl', 'wb') as file:
    pickle.dump(model, file)



# Шаг 6: Предсказание на всем диапазоне данных для построения прямой линии
X_range = np.linspace(X['Mileage'].min(), X['Mileage'].max(), 100).reshape(-1, 1)
y_pred_line = model.predict(X_range)
# Шаг 7: Предсказание цен для новых пробегов
new_mileages = np.array([[15000], [10000], [50000]])  # Пробеги для предсказания
predicted_prices = model.predict(new_mileages)

for mileage, price in zip(new_mileages, predicted_prices):
    print(f'Predicted price for mileage {mileage[0]}: ${price:.2f}')

# Шаг 8: Визуализация данных и предсказаний
plt.figure(figsize=(10, 6))

# Построение точек (реальные данные)
plt.scatter(data['Mileage'], data['Price'], color='blue', label='Actual Price', alpha=0.6, edgecolor='k')

# Построение прямой линии (предсказанная зависимость)
plt.plot(X_range, y_pred_line, color='red', label='Predicted Price (Linear Regression)', linewidth=2)

# Добавление предсказанных цен
plt.scatter(new_mileages, predicted_prices, color='orange', label='Predicted Prices for New Mileages', s=100,
            edgecolor='k')

# Настройки графика
plt.title('Car Price vs Mileage with Linear Regression Line')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.legend()

# Шаг 9: Показ графика
plt.tight_layout()
plt.show()

# Шаг 10: Вывод предсказанных цен в консоль (если нужно)
