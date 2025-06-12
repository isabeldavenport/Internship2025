import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Download data for all assets
print("Downloading data...")
btc = yf.download('BTC-USD', period='5y', interval='1d')
gold = yf.download('GC=F', period='5y', interval='1d')  # Gold futures
ftse = yf.download('^FTSE', period='5y', interval='1d')  # FTSE 100
sp500 = yf.download('^GSPC', period='5y', interval='1d')  # S&P 500

# Ensure all data has the same index by reindexing to the Bitcoin data's index
common_index = btc.index
gold = gold.reindex(common_index)
ftse = ftse.reindex(common_index)
sp500 = sp500.reindex(common_index)

# Create a DataFrame with all features
df = pd.DataFrame(index=common_index)
df['BTC_Close'] = btc['Close'].values
df['Gold_Close'] = gold['Close'].values
df['FTSE_Close'] = ftse['Close'].values
df['SP500_Close'] = sp500['Close'].values

# Drop any rows with missing values
df = df.dropna()

# Print dataset information
print(f"\nTotal number of data points: {len(df)}")
print(f"Date range: from {df.index[0]} to {df.index[-1]}")

# Create features (X) and target (y)
X = df[['Gold_Close', 'FTSE_Close', 'SP500_Close']]
y = df['BTC_Close']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print split information
print(f"\nTraining set size: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Testing set size: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
print(f"Training period: from {df.index[0]} to {df.index[len(X_train)-1]}")
print(f"Testing period: from {df.index[len(X_train)]} to {df.index[-1]}")

# Create and train the model
print("\nTraining the model...")
model = DecisionTreeRegressor(
    max_depth=5,  # Limit depth to prevent overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate comprehensive metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

# Calculate additional statistics
residuals = y_test - y_pred
residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals)

print("\n=== Model Performance Summary ===")
print("\nError Metrics:")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared Score: {r2:.4f}")

print("\nResidual Statistics:")
print(f"Mean of Residuals: ${residuals_mean:.2f}")
print(f"Standard Deviation of Residuals: ${residuals_std:.2f}")

print("\nPrice Statistics:")
print(f"Average Bitcoin Price in Test Set: ${y_test.mean():.2f}")
print(f"Standard Deviation of Bitcoin Price: ${y_test.std():.2f}")
print(f"Minimum Bitcoin Price: ${y_test.min():.2f}")
print(f"Maximum Bitcoin Price: ${y_test.max():.2f}")

print("\nPrediction Statistics:")
print(f"Average Predicted Price: ${y_pred.mean():.2f}")
print(f"Standard Deviation of Predictions: ${y_pred.std():.2f}")
print(f"Minimum Predicted Price: ${y_pred.min():.2f}")
print(f"Maximum Predicted Price: ${y_pred.max():.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance in Bitcoin Price Prediction')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1, 1])

# Plot training data
ax1.plot(df.index[:len(X_train)], y_train, label='Training Data', color='blue')
ax1.set_title('Bitcoin Price: Training Data (80%)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price (USD)')
ax1.grid(True)
ax1.legend()

# Plot testing data and predictions
ax2.plot(df.index[-len(y_test):], y_test, label='Actual', color='blue')
ax2.plot(df.index[-len(y_test):], y_pred, label='Predicted', color='red', linestyle='--')
ax2.set_title('Bitcoin Price: Testing Data (20%) and Predictions')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price (USD)')
ax2.grid(True)
ax2.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()

# Print some example predictions
print("\nExample Predictions:")
test_dates = df.index[-len(y_test):]
for i in range(5):
    print(f"\nDate: {test_dates[i]}")
    print(f"Actual Price: ${y_test.iloc[i]:.2f}")
    print(f"Predicted Price: ${y_pred[i]:.2f}")
    print(f"Absolute Error: ${abs(y_test.iloc[i] - y_pred[i]):.2f}")
    print(f"Percentage Error: {abs(y_test.iloc[i] - y_pred[i])/y_test.iloc[i]*100:.2f}%")
    print(f"Features used:")
    print(f"  Gold: ${X_test.iloc[i]['Gold_Close']:.2f}")
    print(f"  FTSE100: {X_test.iloc[i]['FTSE_Close']:.2f}")
    print(f"  S&P500: {X_test.iloc[i]['SP500_Close']:.2f}") 