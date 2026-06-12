import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("train.csv")

# Select Features and Target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict House Prices
y_pred = model.predict(X_test)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 60)
print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
print("=" * 60)

print(f"\nRMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:<15}: {coef:.2f}")

print(f"Intercept       : {model.intercept_:.2f}")

# Actual vs Predicted Prices
results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': np.round(y_pred, 2)
})

print("\nFirst 10 Predictions:")
print(results.head(10))

# Predict a New House
sample_house = [[2000, 3, 2]]  # Area, Bedrooms, Bathrooms
predicted_price = model.predict(sample_house)

print("\nSample Prediction")
print("-" * 30)
print("Square Footage : 2000")
print("Bedrooms       : 3")
print("Bathrooms      : 2")
print(f"Predicted Price: ${predicted_price[0]:,.2f}")


features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
coefficients = model.coef_

plt.figure(figsize=(8,5))
plt.bar(features, coefficients)
plt.title('Impact of Features on House Price')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.show()
actual = y_test.values[:50]
predicted = y_pred[:50]

plt.figure(figsize=(10, 5))
plt.plot(actual, label='Actual Price')
plt.plot(predicted, label='Predicted Price')

plt.title('Actual vs Predicted House Prices')
plt.xlabel('House Samples')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.show()