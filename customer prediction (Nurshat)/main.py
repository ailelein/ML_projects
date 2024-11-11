
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Вложения в рекламу
y = np.array([50, 60, 65, 80, 90, 100, 105, 110, 115, 130])  # Количество покупателей


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict([[5]])
print(y_pred)

# mse = mean_squared_error(y_test, y_pred)
# print(f"Среднеквадратичная ошибка: {mse:.2f}") # Чем меньше значение, тем лучше модель.


plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Фактические данные')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Линия регрессии')
plt.title('Вложения в рекламу vs Количество покупателей')
plt.xlabel('Вложения в рекламу (тыс. долларов)')
plt.ylabel('Количество покупателей')
plt.legend()
plt.grid(True)
plt.show()
