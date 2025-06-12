import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    st.markdown("##### Tree-based Models Parameters")
    n_estimators = st.slider('Number of Trees', 50, 500, 100)
    max_depth = st.slider('Maximum Depth', 3, 10, 5)
    
    st.markdown("##### Gradient Boosting Parameters")
    gb_learning_rate = st.slider('Gradient Boosting Learning Rate', 0.01, 0.3, 0.1)
    
    st.markdown("##### Ensemble Weights")
    lr_weight = st.slider('Linear Regression Weight', 0.0, 1.0, 0.33)
    rf_weight = st.slider('Random Forest Weight', 0.0, 1.0, 0.33)
    gb_weight = 1 - lr_weight - rf_weight
    st.markdown(f"Gradient Boosting Weight: {gb_weight:.2f}")
    
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
    btc_value = float(btc['Close'].iloc[-1])
    st.metric("Bitcoin", f"${btc_value:,.2f}")
with col2:
    gold_value = float(gold['Close'].iloc[-1])
    st.metric("Gold", f"${gold_value:,.2f}")
with col3:
    ftse_value = float(ftse['Close'].iloc[-1])
    st.metric("FTSE", f"{ftse_value:,.2f}")
with col4:
    sp500_value = float(sp500['Close'].iloc[-1])
    st.metric("S&P 500", f"{sp500_value:,.2f}")

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

# Train models
with st.spinner('Training models...'):
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=gb_learning_rate,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)

# Make predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)
gb_train_pred = gb_model.predict(X_train_scaled)
gb_test_pred = gb_model.predict(X_test_scaled)

# Ensemble predictions
train_predictions = (lr_weight * lr_train_pred + rf_weight * rf_train_pred + gb_weight * gb_train_pred)
test_predictions = (lr_weight * lr_test_pred + rf_weight * rf_test_pred + gb_weight * gb_test_pred)

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

# Plot predictions
st.markdown("### 📈 Price Predictions")

# Create prediction dataframes
train_dates = train_data.index
test_dates = test_data.index

# Linear Regression Plot
st.markdown("#### Linear Regression Model")
fig_lr, ax_lr = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")
ax_lr.plot(df.index, df['btc_close'], label='Actual Price', color='#1f77b4', linewidth=2)
ax_lr.plot(train_dates, lr_train_pred, label='Linear Regression Prediction (Train)', color='#ff7f0e', linestyle='--', linewidth=2)
ax_lr.plot(test_dates, lr_test_pred, label='Linear Regression Prediction (Test)', color='#ff7f0e', linewidth=2)
ax_lr.axvline(x=split_date, color='#2ca02c', linestyle='--', label='Train/Test Split')
ax_lr.set_title('Bitcoin Price Predictions (Linear Regression)', fontsize=14, pad=20)
ax_lr.set_xlabel('Date', fontsize=12)
ax_lr.set_ylabel('Price (USD)', fontsize=12)
ax_lr.legend(fontsize=10)
ax_lr.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_lr)

# Random Forest Plot
st.markdown("#### Random Forest Model")
fig_rf, ax_rf = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")
ax_rf.plot(df.index, df['btc_close'], label='Actual Price', color='#1f77b4', linewidth=2)
ax_rf.plot(train_dates, rf_train_pred, label='Random Forest Prediction (Train)', color='#ff7f0e', linestyle='--', linewidth=2)
ax_rf.plot(test_dates, rf_test_pred, label='Random Forest Prediction (Test)', color='#ff7f0e', linewidth=2)
ax_rf.axvline(x=split_date, color='#2ca02c', linestyle='--', label='Train/Test Split')
ax_rf.set_title('Bitcoin Price Predictions (Random Forest)', fontsize=14, pad=20)
ax_rf.set_xlabel('Date', fontsize=12)
ax_rf.set_ylabel('Price (USD)', fontsize=12)
ax_rf.legend(fontsize=10)
ax_rf.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_rf)

# Gradient Boosting Plot
st.markdown("#### Gradient Boosting Model")
fig_gb, ax_gb = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")
ax_gb.plot(df.index, df['btc_close'], label='Actual Price', color='#1f77b4', linewidth=2)
ax_gb.plot(train_dates, gb_train_pred, label='Gradient Boosting Prediction (Train)', color='#ff7f0e', linestyle='--', linewidth=2)
ax_gb.plot(test_dates, gb_test_pred, label='Gradient Boosting Prediction (Test)', color='#ff7f0e', linewidth=2)
ax_gb.axvline(x=split_date, color='#2ca02c', linestyle='--', label='Train/Test Split')
ax_gb.set_title('Bitcoin Price Predictions (Gradient Boosting)', fontsize=14, pad=20)
ax_gb.set_xlabel('Date', fontsize=12)
ax_gb.set_ylabel('Price (USD)', fontsize=12)
ax_gb.legend(fontsize=10)
ax_gb.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_gb)

# Ensemble Plot
st.markdown("#### Ensemble Model")
fig_ensemble, ax_ensemble = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")
ax_ensemble.plot(df.index, df['btc_close'], label='Actual Price', color='#1f77b4', linewidth=2)
ax_ensemble.plot(train_dates, train_predictions, label='Ensemble Prediction (Train)', color='#ff7f0e', linestyle='--', linewidth=2)
ax_ensemble.plot(test_dates, test_predictions, label='Ensemble Prediction (Test)', color='#ff7f0e', linewidth=2)
ax_ensemble.axvline(x=split_date, color='#2ca02c', linestyle='--', label='Train/Test Split')
ax_ensemble.set_title('Bitcoin Price Predictions (Ensemble Model)', fontsize=14, pad=20)
ax_ensemble.set_xlabel('Date', fontsize=12)
ax_ensemble.set_ylabel('Price (USD)', fontsize=12)
ax_ensemble.legend(fontsize=10)
ax_ensemble.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_ensemble)

# Calculate metrics for all models
train_actual = train_data['btc_close']
test_actual = test_data['btc_close']

# Calculate metrics for Linear Regression
lr_train_rmse = np.sqrt(mean_squared_error(train_actual, lr_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(test_actual, lr_test_pred))
lr_train_r2 = r2_score(train_actual, lr_train_pred)
lr_test_r2 = r2_score(test_actual, lr_test_pred)

# Calculate metrics for Random Forest
rf_train_rmse = np.sqrt(mean_squared_error(train_actual, rf_train_pred))
rf_test_rmse = np.sqrt(mean_squared_error(test_actual, rf_test_pred))
rf_train_r2 = r2_score(train_actual, rf_train_pred)
rf_test_r2 = r2_score(test_actual, rf_test_pred)

# Calculate metrics for Gradient Boosting
gb_train_rmse = np.sqrt(mean_squared_error(train_actual, gb_train_pred))
gb_test_rmse = np.sqrt(mean_squared_error(test_actual, gb_test_pred))
gb_train_r2 = r2_score(train_actual, gb_train_pred)
gb_test_r2 = r2_score(test_actual, gb_test_pred)

# Calculate metrics for Ensemble
ensemble_train_rmse = np.sqrt(mean_squared_error(train_actual, train_predictions))
ensemble_test_rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
ensemble_train_r2 = r2_score(train_actual, train_predictions)
ensemble_test_r2 = r2_score(test_actual, test_predictions)

# Display model performance
st.markdown("### 🎯 Model Performance (RMSE)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Linear Regression Train RMSE", f"${lr_train_rmse:.2f}")
with col2:
    st.metric("Random Forest Train RMSE", f"${rf_train_rmse:.2f}")
with col3:
    st.metric("Gradient Boosting Train RMSE", f"${gb_train_rmse:.2f}")
with col4:
    st.metric("Ensemble Train RMSE", f"${ensemble_train_rmse:.2f}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Linear Regression Test RMSE", f"${lr_test_rmse:.2f}")
with col2:
    st.metric("Random Forest Test RMSE", f"${rf_test_rmse:.2f}")
with col3:
    st.metric("Gradient Boosting Test RMSE", f"${gb_test_rmse:.2f}")
with col4:
    st.metric("Ensemble Test RMSE", f"${ensemble_test_rmse:.2f}")

# Display R² scores
st.markdown("### 📊 R² Scores")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Linear Regression Train R²", f"{lr_train_r2:.4f}")
with col2:
    st.metric("Random Forest Train R²", f"{rf_train_r2:.4f}")
with col3:
    st.metric("Gradient Boosting Train R²", f"{gb_train_r2:.4f}")
with col4:
    st.metric("Ensemble Train R²", f"{ensemble_train_r2:.4f}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Linear Regression Test R²", f"{lr_test_r2:.4f}")
with col2:
    st.metric("Random Forest Test R²", f"{rf_test_r2:.4f}")
with col3:
    st.metric("Gradient Boosting Test R²", f"{gb_test_r2:.4f}")
with col4:
    st.metric("Ensemble Test R²", f"{ensemble_test_r2:.4f}")

# Feature importance
st.markdown("### 🔍 Model Coefficients and Feature Importance")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Linear Regression Coefficients")
    lr_importance = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': lr_model.coef_
    })
    lr_importance = lr_importance.sort_values('Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=lr_importance, x='Coefficient', y='Feature', palette='viridis')
    ax.set_title('Linear Regression Coefficients', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("#### Random Forest Feature Importance")
    rf_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': rf_model.feature_importances_
    })
    rf_importance = rf_importance.sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=rf_importance, x='Importance', y='Feature', palette='viridis')
    ax.set_title('Random Forest Feature Importance', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

with col3:
    st.markdown("#### Gradient Boosting Feature Importance")
    gb_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': gb_model.feature_importances_
    })
    gb_importance = gb_importance.sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=gb_importance, x='Importance', y='Feature', palette='viridis')
    ax.set_title('Gradient Boosting Feature Importance', fontsize=12)
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

# Display raw data
st.subheader("Raw Data")
st.dataframe(df.tail())

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from Yahoo Finance") 