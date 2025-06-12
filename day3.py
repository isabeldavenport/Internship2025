import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Download data for all assets
print("Downloading data...")
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=5)
btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
gold = yf.download('GC=F', start=start_date, end=end_date, interval='1d')  # Gold futures
ftse = yf.download('^FTSE', start=start_date, end=end_date, interval='1d')  # FTSE 100
sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')  # S&P 500

# Validate and filter data dates
now = pd.Timestamp.now().normalize()
btc = btc[btc.index <= now]
gold = gold[gold.index <= now]
ftse = ftse[ftse.index <= now]
sp500 = sp500[sp500.index <= now]

print("\nLast 5 rows of BTC data after filtering:")
print(btc[['Close']].tail())

# Print detailed data validation
print("\n=== Data Validation ===")
print("\nBitcoin Data:")
print(f"Date range: from {btc.index[0]} to {btc.index[-1]}")
print(f"Number of data points: {len(btc)}")
print(f"Minimum price: ${float(btc['Close'].min()):.2f}")
print(f"Maximum price: ${float(btc['Close'].max()):.2f}")
print(f"Average price: ${float(btc['Close'].mean()):.2f}")
print(f"Latest price: ${float(btc['Close'].iloc[-1]):.2f}")

print("\nChecking for missing values:")
print(f"Bitcoin missing values: {btc['Close'].isnull().sum()}")
print(f"Gold missing values: {gold['Close'].isnull().sum()}")
print(f"FTSE missing values: {ftse['Close'].isnull().sum()}")
print(f"S&P500 missing values: {sp500['Close'].isnull().sum()}")

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
print(f"\nNumber of rows after dropping missing values: {len(df)}")

# Create features (X) and target (y)
X = df[['Gold_Close', 'FTSE_Close', 'SP500_Close']]
y = df['BTC_Close']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData Split Information:")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print(f"Training period: from {df.index[0]} to {df.index[len(X_train)-1]}")
print(f"Testing period: from {df.index[len(X_train)]} to {df.index[-1]}")

# Create and train the model
model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

print("\nPrediction Statistics:")
print(f"Actual price range: ${float(y_test.min()):.2f} to ${float(y_test.max()):.2f}")
print(f"Predicted price range: ${float(y_pred.min()):.2f} to ${float(y_pred.max()):.2f}")



