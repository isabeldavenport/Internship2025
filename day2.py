import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Download Apple stock data with dividend information
df = yf.download('AAPL', period='max', actions=True)

# Create a DataFrame containing only dates with dividend payments
dividend_df = df[df['Dividends'] > 0]

# --- Print Dividend Information ---
print("\nApple Dividend Information:")
print(f"Total number of dividend payments: {len(dividend_df)}")
print("\nRecent dividend payment dates and amounts:")

# Get the last 10 dividend payments to loop through
recent_dividends = dividend_df['Dividends'].tail(10)

# Loop through the recent_dividends Series. 
# .items() gives (date, amount) pairs, where amount is a single number.
for date, amount in recent_dividends.items():
    # If amount is a Series (shouldn't be, but just in case), take the first value
    if isinstance(amount, pd.Series) or isinstance(amount, np.ndarray):
        amount_val = float(amount.iloc[0]) if hasattr(amount, 'iloc') else float(amount[0])
    else:
        amount_val = float(amount)
    # Format date as string
    date_str = str(date)
    print(f"Date: {date_str}, Amount: ${amount_val:.2f}")

# Calculate total dividends paid and ensure it is a single number (float)
total_dividends = dividend_df['Dividends'].sum()
print(f"\nTotal dividends paid: ${float(total_dividends):.2f}")

# Calculate average dividend amount and ensure it is a single number (float)
avg_dividend = dividend_df['Dividends'].mean()
print(f"Average dividend amount: ${float(avg_dividend):.2f}")

# --- Graphing Dividends Over Time ---
print("\nGenerating graph of dividends per year...")

# Group dividends by year and count them
dividends_per_year = dividend_df['Dividends'].groupby(dividend_df.index.year).count()

# Convert the index and values to lists of simple types for plotting
x_years = [int(year) if not isinstance(year, (np.ndarray, list)) else int(year[0]) for year in dividends_per_year.index]
y_counts = [int(count) if not isinstance(count, (np.ndarray, list)) else int(count[0]) for count in dividends_per_year.values]

# Debug prints
print('x_years:', x_years)
print('y_counts:', y_counts)
print('x_years type:', type(x_years[0]))
print('y_counts type:', type(y_counts[0]))

# Create a bar chart
plt.figure(figsize=(12, 7))
bars = plt.bar(x_years, y_counts, color='dodgerblue')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Dividend Payments", fontsize=12)
plt.title("Number of Apple (AAPL) Dividend Payments Per Year", fontsize=14)
plt.xticks(x_years, rotation=45, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add text labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center') # va: vertical alignment

plt.show()


# --- Graphing Stock Price Over Time ---
print("\nGenerating graph of AAPL closing price...")

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Close Price', color='navy')
plt.title("Apple (AAPL) Stock Price Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Stock Price (USD)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show() 