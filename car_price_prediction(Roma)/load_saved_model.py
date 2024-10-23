import pickle
with open('saved_model.pkl', 'rb') as file:
    load_model = pickle.load(file)
new_data = [[15000], [30000], [70000]]
predicted_prices = load_model.predict(new_data)
for mileage, price in zip(new_data, predicted_prices):
    print(f'Predicted price for mileage {mileage[0]}: ${price:.2f}')

