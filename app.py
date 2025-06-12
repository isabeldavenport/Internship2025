import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric label {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stMetric div {
        font-size: 1.5rem;
    }
    .stSubheader {
        padding-top: 2rem;
        padding-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    .stSidebar .sidebar-content {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Bitcoin Price Predictor")
st.markdown("""
    This app uses machine learning to predict Bitcoin prices based on various market indicators.
    Adjust the parameters in the sidebar to see how different features and model settings affect the predictions.
""")

# Sidebar styling
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=100)
    st.markdown("### Model Configuration")
    
    # Model parameters
    st.markdown("#### Model Parameters")
    n_estimators = st.slider('Number of Trees', 50, 500, 100)
    max_depth = st.slider('Maximum Depth', 3, 10, 5)
    learning_rate = st.slider('Learning Rate', 0.01, 0.3, 0.1)
    
    # Feature selection
    st.markdown("#### Feature Selection")
    selected_features = st.multiselect(
        'Select features to include:',
        ['Bitcoin Volume', 'Gold Price', 'FTSE Price', 'S&P 500 Price', 'Bitcoin Volatility', 
         '7-Day Moving Average', '30-Day Moving Average'],
        default=['Bitcoin Volume', 'Gold Price', 'FTSE Price', 'S&P 500 Price', 'Bitcoin Volatility']
    )
    
    # Train-test split
    st.markdown("#### Training Period")
    split_date = st.date_input(
        'Select Training End Date',
        value=pd.Timestamp('2024-01-01'),
        min_value=pd.Timestamp('2020-01-01'),
        max_value=pd.Timestamp('2025-06-30')
    )

# Download data
with st.spinner('Downloading market data...'):
    start_date = '2020-01-01'
    end_date = '2025-06-30'
    btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
    gold = yf.download('GC=F', start=start_date, end=end_date, interval='1d')
    ftse = yf.download('^FTSE', start=start_date, end=end_date, interval='1d')
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')

# Display current market data
st.markdown("### 📊 Current Market Data")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Bitcoin", f"${float(btc['Close'].iloc[-1]):,.2f}")
with col2:
    st.metric("Gold", f"${float(gold['Close'].iloc[-1]):,.2f}")
with col3:
    st.metric("FTSE", f"{float(ftse['Close'].iloc[-1]):,.2f}")
with col4:
    st.metric("S&P 500", f"{float(sp500['Close'].iloc[-1]):,.2f}")

# Prepare dataset
df = pd.DataFrame(index=btc.index)
df['btc_close'] = btc['Close']
df['btc_volume'] = btc['Volume']
df['gold_close'] = gold['Close']
df['ftse_close'] = ftse['Close']
df['sp500_close'] = sp500['Close']
df['btc_volatility'] = df['btc_close'].pct_change().rolling(window=7).std()
# Add moving averages
df['btc_ma7'] = df['btc_close'].rolling(window=7).mean()
df['btc_ma30'] = df['btc_close'].rolling(window=30).mean()
df = df.dropna()

# Map feature names to column names
feature_map = {
    'Bitcoin Volume': 'btc_volume',
    'Gold Price': 'gold_close',
    'FTSE Price': 'ftse_close',
    'S&P 500 Price': 'sp500_close',
    'Bitcoin Volatility': 'btc_volatility',
    '7-Day Moving Average': 'btc_ma7',
    '30-Day Moving Average': 'btc_ma30'
}

# Convert split_date to pandas Timestamp
split_date = pd.Timestamp(split_date)

# Split data
train_data = df[df.index < split_date]
test_data = df[df.index >= split_date]

# Prepare features and target
selected_columns = [feature_map[feature] for feature in selected_features]
X_train = train_data[selected_columns]
X_test = test_data[selected_columns]
y_train = train_data['btc_close']
y_test = test_data['btc_close']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
with st.spinner('Training model...'):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

# Make predictions
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Date': df.index,
    'Actual Price': df['btc_close'],
    'Predicted Price': np.concatenate([train_predictions, test_predictions])
})

# Filter data to show only from 2023 onwards
plot_start_date = pd.Timestamp('2023-01-01')
plot_end_date = pd.Timestamp('2025-06-30')
comparison_df = comparison_df[(comparison_df['Date'] >= plot_start_date) & (comparison_df['Date'] <= plot_end_date)]

# Calculate error metrics
comparison_df['Absolute Error'] = abs(comparison_df['Actual Price'] - comparison_df['Predicted Price'])
comparison_df['Percentage Error'] = (comparison_df['Absolute Error'] / comparison_df['Actual Price']) * 100

# Calculate RMSE and R² score
rmse = np.sqrt(mean_squared_error(comparison_df['Actual Price'], comparison_df['Predicted Price']))
r2 = r2_score(comparison_df['Actual Price'], comparison_df['Predicted Price'])

# Display model performance
st.markdown("### 🎯 Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("RMSE", f"${rmse:.2f}")
with col2:
    st.metric("R² Score", f"{r2:.4f}")
with col3:
    st.metric("Training Size", f"{len(train_data)} days")

# Plot predictions
st.markdown("### 📈 Price Predictions")
fig, ax = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")
ax.plot(comparison_df['Date'], comparison_df['Actual Price'], label='Actual Price', color='#1f77b4', linewidth=2)
ax.plot(comparison_df['Date'], comparison_df['Predicted Price'], label='Predicted Price', color='#ff7f0e', linestyle='--', linewidth=2)
ax.axvline(x=split_date, color='#2ca02c', linestyle='--', label='Train/Test Split')
ax.set_title('Bitcoin Price Predictions', fontsize=14, pad=20)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# Display comparison table
st.markdown("### 📊 Detailed Predictions")
comparison_df['Date'] = comparison_df['Date'].dt.strftime('%Y-%m-%d')
comparison_df['Actual Price'] = comparison_df['Actual Price'].map('${:,.2f}'.format)
comparison_df['Predicted Price'] = comparison_df['Predicted Price'].map('${:,.2f}'.format)
comparison_df['Absolute Error'] = comparison_df['Absolute Error'].map('${:,.2f}'.format)
comparison_df['Percentage Error'] = comparison_df['Percentage Error'].map('{:.2f}%'.format)
st.dataframe(comparison_df, use_container_width=True)

# Feature importance
st.markdown("### 🔍 Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
ax.set_title('Feature Importance', fontsize=14, pad=20)
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
st.pyplot(fig)

# Display raw data
st.subheader("Raw Data")
st.dataframe(df.tail())

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from Yahoo Finance") 