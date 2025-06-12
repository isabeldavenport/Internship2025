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
from datetime import datetime, timedelta
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import time

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
with st.spinner("Training models..."):
    # Initialize models
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    sentiment_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    # Train models
    lr_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)
    sentiment_model.fit(X_train_scaled, y_train)

    # Make predictions for training data
    lr_train_pred = lr_model.predict(X_train_scaled)
    rf_train_pred = rf_model.predict(X_train_scaled)
    gb_train_pred = gb_model.predict(X_train_scaled)
    sentiment_train_pred = sentiment_model.predict(X_train_scaled)

    # Make predictions for test data
    lr_test_pred = lr_model.predict(X_test_scaled)
    rf_test_pred = rf_model.predict(X_test_scaled)
    gb_test_pred = gb_model.predict(X_test_scaled)
    sentiment_test_pred = sentiment_model.predict(X_test_scaled)

    # Create comparison dataframe with all predictions
    comparison_df = pd.DataFrame({
        'Date': df.index,
        'Actual': df['btc_close'],
        'Linear Regression': np.concatenate([lr_train_pred, lr_test_pred]),
        'Random Forest': np.concatenate([rf_train_pred, rf_test_pred]),
        'Gradient Boosting': np.concatenate([gb_train_pred, gb_test_pred]),
        'Sentiment Model': np.concatenate([sentiment_train_pred, sentiment_test_pred])
    })

# Filter data to show only from 2023 onwards
plot_start_date = pd.Timestamp('2023-01-01')
plot_end_date = pd.Timestamp('2025-06-30')
comparison_df = comparison_df[(comparison_df['Date'] >= plot_start_date) & (comparison_df['Date'] <= plot_end_date)]

# Calculate error metrics
comparison_df['Absolute Error'] = abs(comparison_df['Actual'] - comparison_df['Linear Regression'])
comparison_df['Percentage Error'] = (comparison_df['Absolute Error'] / comparison_df['Actual']) * 100

# Calculate RMSE and R² score
rmse = np.sqrt(mean_squared_error(comparison_df['Actual'], comparison_df['Linear Regression']))
r2 = r2_score(comparison_df['Actual'], comparison_df['Linear Regression'])

# Plot predictions
st.markdown("### 📈 Price Predictions")

# Create prediction dataframes
train_dates = train_data.index
test_dates = test_data.index

# Plot price prediction models
st.markdown("### 📈 Price Prediction Models")

# Combined plot of all models
st.markdown("### 📈 Combined Price Predictions")
fig = go.Figure()

# Add traces with hover information
fig.add_trace(go.Scatter(
    x=comparison_df['Date'],
    y=comparison_df['Actual'],
    name='Actual Price',
    line=dict(color='black'),
    hovertemplate='Date: %{x}<br>Actual Price: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=comparison_df['Date'][:len(train_dates)],
    y=comparison_df['Linear Regression'][:len(train_dates)],
    name='Linear Regression (Train)',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Linear Regression: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=comparison_df['Date'][len(train_dates):],
    y=comparison_df['Linear Regression'][len(train_dates):],
    name='Linear Regression (Test)',
    line=dict(color='blue', dash='dash'),
    hovertemplate='Date: %{x}<br>Linear Regression: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=comparison_df['Date'][:len(train_dates)],
    y=comparison_df['Random Forest'][:len(train_dates)],
    name='Random Forest (Train)',
    line=dict(color='green'),
    hovertemplate='Date: %{x}<br>Random Forest: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=comparison_df['Date'][len(train_dates):],
    y=comparison_df['Random Forest'][len(train_dates):],
    name='Random Forest (Test)',
    line=dict(color='green', dash='dash'),
    hovertemplate='Date: %{x}<br>Random Forest: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=comparison_df['Date'][:len(train_dates)],
    y=comparison_df['Gradient Boosting'][:len(train_dates)],
    name='Gradient Boosting (Train)',
    line=dict(color='red'),
    hovertemplate='Date: %{x}<br>Gradient Boosting: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=comparison_df['Date'][len(train_dates):],
    y=comparison_df['Gradient Boosting'][len(train_dates):],
    name='Gradient Boosting (Test)',
    line=dict(color='red', dash='dash'),
    hovertemplate='Date: %{x}<br>Gradient Boosting: $%{y:,.2f}<extra></extra>'
))

# Add sentiment model predictions
fig.add_trace(go.Scatter(
    x=comparison_df['Date'][:len(train_dates)],
    y=comparison_df['Sentiment Model'][:len(train_dates)],
    name='Sentiment Model (Train)',
    line=dict(color='purple'),
    hovertemplate='Date: %{x}<br>Sentiment Model: $%{y:,.2f}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=comparison_df['Date'][len(train_dates):],
    y=comparison_df['Sentiment Model'][len(train_dates):],
    name='Sentiment Model (Test)',
    line=dict(color='purple', dash='dash'),
    hovertemplate='Date: %{x}<br>Sentiment Model: $%{y:,.2f}<extra></extra>'
))

# Add vertical line for train/test split
split_date = df.index[len(train_dates)]
fig.add_shape(
    type="line",
    x0=split_date,
    x1=split_date,
    y0=0,
    y1=1,
    yref="paper",
    line=dict(
        color="gray",
        width=2,
        dash="dash",
    )
)

# Add annotation for the split line
fig.add_annotation(
    x=split_date,
    y=1,
    yref="paper",
    text="Train/Test Split",
    showarrow=False,
    yshift=10
)

fig.update_layout(
    title='All Models: Bitcoin Price Predictions with Sentiment',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    yaxis2=dict(
        title='Sentiment Score',
        overlaying='y',
        side='right',
        range=[-1, 1]
    ),
    height=600,
    hovermode='x unified',
    hoverdistance=100,
    spikedistance=1000
)

# Add hover spikes
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)

st.plotly_chart(fig, use_container_width=True)

# Individual model plots
st.markdown("### 📊 Individual Model Predictions")

# Linear Regression Plot
st.markdown("#### Linear Regression Model")
lr_fig = go.Figure()
lr_fig.add_trace(go.Scatter(
    x=df.index,
    y=df['btc_close'],
    name='Actual Price',
    line=dict(color='black'),
    hovertemplate='Date: %{x}<br>Actual Price: $%{y:,.2f}<extra></extra>'
))
lr_fig.add_trace(go.Scatter(
    x=df.index[:len(train_dates)],
    y=lr_model.predict(X_train_scaled),
    name='Train Predictions',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Predicted Price: $%{y:,.2f}<extra></extra>'
))
lr_fig.add_trace(go.Scatter(
    x=df.index[len(train_dates):],
    y=lr_test_pred,
    name='Test Predictions',
    line=dict(color='blue', dash='dash'),
    hovertemplate='Date: %{x}<br>Predicted Price: $%{y:,.2f}<extra></extra>'
))
lr_fig.add_shape(
    type="line",
    x0=split_date,
    x1=split_date,
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color="gray", width=2, dash="dash")
)
lr_fig.update_layout(
    title='Linear Regression Predictions',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    height=400,
    hovermode='x unified',
    showlegend=True
)
st.plotly_chart(lr_fig, use_container_width=True)

# Random Forest Plot
st.markdown("#### Random Forest Model")
rf_fig = go.Figure()
rf_fig.add_trace(go.Scatter(
    x=df.index,
    y=df['btc_close'],
    name='Actual Price',
    line=dict(color='black'),
    hovertemplate='Date: %{x}<br>Actual Price: $%{y:,.2f}<extra></extra>'
))
rf_fig.add_trace(go.Scatter(
    x=df.index[:len(train_dates)],
    y=rf_model.predict(X_train_scaled),
    name='Train Predictions',
    line=dict(color='green'),
    hovertemplate='Date: %{x}<br>Predicted Price: $%{y:,.2f}<extra></extra>'
))
rf_fig.add_trace(go.Scatter(
    x=df.index[len(train_dates):],
    y=rf_test_pred,
    name='Test Predictions',
    line=dict(color='green', dash='dash'),
    hovertemplate='Date: %{x}<br>Predicted Price: $%{y:,.2f}<extra></extra>'
))
rf_fig.add_shape(
    type="line",
    x0=split_date,
    x1=split_date,
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color="gray", width=2, dash="dash")
)
rf_fig.update_layout(
    title='Random Forest Predictions',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    height=400,
    hovermode='x unified',
    showlegend=True
)
st.plotly_chart(rf_fig, use_container_width=True)

# Gradient Boosting Plot
st.markdown("#### Gradient Boosting Model")
gb_fig = go.Figure()
gb_fig.add_trace(go.Scatter(
    x=df.index,
    y=df['btc_close'],
    name='Actual Price',
    line=dict(color='black'),
    hovertemplate='Date: %{x}<br>Actual Price: $%{y:,.2f}<extra></extra>'
))
gb_fig.add_trace(go.Scatter(
    x=df.index[:len(train_dates)],
    y=gb_model.predict(X_train_scaled),
    name='Train Predictions',
    line=dict(color='red'),
    hovertemplate='Date: %{x}<br>Predicted Price: $%{y:,.2f}<extra></extra>'
))
gb_fig.add_trace(go.Scatter(
    x=df.index[len(train_dates):],
    y=gb_test_pred,
    name='Test Predictions',
    line=dict(color='red', dash='dash'),
    hovertemplate='Date: %{x}<br>Predicted Price: $%{y:,.2f}<extra></extra>'
))
gb_fig.add_shape(
    type="line",
    x0=split_date,
    x1=split_date,
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color="gray", width=2, dash="dash")
)
gb_fig.update_layout(
    title='Gradient Boosting Predictions',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    height=400,
    hovermode='x unified',
    showlegend=True
)
st.plotly_chart(gb_fig, use_container_width=True)

# Calculate metrics for each model
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
lr_train_r2 = r2_score(y_train, lr_train_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)

rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)

gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_pred))
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
gb_train_r2 = r2_score(y_train, gb_train_pred)
gb_test_r2 = r2_score(y_test, gb_test_pred)

# Calculate ensemble predictions
ensemble_train_pred = (lr_train_pred + rf_train_pred + gb_train_pred) / 3
ensemble_test_pred = (lr_test_pred + rf_test_pred + gb_test_pred) / 3

# Calculate ensemble metrics
ensemble_train_rmse = np.sqrt(mean_squared_error(y_train, ensemble_train_pred))
ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
ensemble_train_r2 = r2_score(y_train, ensemble_train_pred)
ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)

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
comparison_df['Actual Price'] = comparison_df['Actual'].map('${:,.2f}'.format)
comparison_df['Predicted Price'] = comparison_df['Linear Regression'].map('${:,.2f}'.format)
comparison_df['Absolute Error'] = comparison_df['Absolute Error'].map('${:,.2f}'.format)
comparison_df['Percentage Error'] = comparison_df['Percentage Error'].map('{:.2f}%'.format)
st.dataframe(comparison_df, use_container_width=True)

# Display raw data
st.subheader("Raw Data")
st.dataframe(df.tail())

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from Yahoo Finance")

# Add sentiment analysis section
st.markdown("### 📊 Bitcoin Sentiment Analysis")
st.markdown("Analyze Bitcoin sentiment and integrate it with price predictions.")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Create tabs for different sentiment analysis methods
sentiment_tab1, sentiment_tab2 = st.tabs(["Text Analysis", "Historical Analysis"])

with sentiment_tab1:
    # Text input for sentiment analysis
    text_input = st.text_area("Enter Bitcoin-related text to analyze:", 
                             "Bitcoin is showing strong momentum and positive market sentiment.")
    
    if text_input:
        # Get sentiment scores
        sentiment_scores = analyzer.polarity_scores(text_input)
        
        # Display sentiment scores
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Compound Score", f"{sentiment_scores['compound']:.2f}")
        with col2:
            st.metric("Positive", f"{sentiment_scores['pos']:.2f}")
        with col3:
            st.metric("Neutral", f"{sentiment_scores['neu']:.2f}")
        with col4:
            st.metric("Negative", f"{sentiment_scores['neg']:.2f}")
        
        # Create a gauge chart for compound sentiment
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_scores['compound'],
            title={'text': "Overall Sentiment"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "red"},
                    {'range': [-0.5, 0], 'color': "orange"},
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_scores['compound']
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        st.markdown("#### Interpretation")
        if sentiment_scores['compound'] >= 0.05:
            st.success("The text shows positive sentiment towards Bitcoin.")
        elif sentiment_scores['compound'] <= -0.05:
            st.error("The text shows negative sentiment towards Bitcoin.")
        else:
            st.info("The text shows neutral sentiment towards Bitcoin.")

with sentiment_tab2:
    st.markdown("### Historical Sentiment Analysis")
    
    # Generate synthetic sentiment data for demonstration
    dates = pd.date_range(end=datetime.now(), periods=30)
    sentiment_data = pd.DataFrame({
        'date': pd.to_datetime(dates).date,  # Convert to date objects
        'sentiment': np.random.normal(0.2, 0.3, 30)  # Generate random sentiment scores
    })
    
    # Plot historical sentiment
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sentiment_data['date'],
        y=sentiment_data['sentiment'],
        mode='lines+markers',
        name='Daily Sentiment'
    ))
    fig.update_layout(
        title='Bitcoin Sentiment Over Time',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add sentiment as a feature to the model
    if not sentiment_data.empty:
        # Create a copy of the dataframe with dates
        df_with_dates = df.copy()
        df_with_dates['date'] = df.index.strftime('%Y-%m-%d')
        df_with_dates['date'] = pd.to_datetime(df_with_dates['date']).dt.date
        
        # Merge sentiment data with price data
        df_with_dates = df_with_dates.merge(sentiment_data, on='date', how='left')
        df_with_dates['sentiment'] = df_with_dates['sentiment'].fillna(0)  # Fill missing values with neutral sentiment

# Model Training Section
st.markdown("### 🤖 Model Training")
st.markdown("Train and compare different machine learning models for Bitcoin price prediction.")

if 'df_with_dates' in locals() and not df_with_dates.empty:
    # Add sentiment data to features
    features = ['btc_close', 'btc_volume', 'gold_close', 'ftse_close', 'sp500_close', 'sentiment']
    X = df_with_dates[features].values
    y = df_with_dates['btc_close'].values

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    with st.spinner("Training models..."):
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        sentiment_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        lr_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        sentiment_model.fit(X_train_scaled, y_train)

        # Make predictions for training data
        lr_train_pred = lr_model.predict(X_train_scaled)
        rf_train_pred = rf_model.predict(X_train_scaled)
        gb_train_pred = gb_model.predict(X_train_scaled)
        sentiment_train_pred = sentiment_model.predict(X_train_scaled)

        # Make predictions for test data
        lr_test_pred = lr_model.predict(X_test_scaled)
        rf_test_pred = rf_model.predict(X_test_scaled)
        gb_test_pred = gb_model.predict(X_test_scaled)
        sentiment_test_pred = sentiment_model.predict(X_test_scaled)

        # Create comparison dataframe with all predictions
        comparison_df = pd.DataFrame({
            'Date': df.index,
            'Actual': df['btc_close'],
            'Linear Regression': np.concatenate([lr_train_pred, lr_test_pred]),
            'Random Forest': np.concatenate([rf_train_pred, rf_test_pred]),
            'Gradient Boosting': np.concatenate([gb_train_pred, gb_test_pred]),
            'Sentiment Model': np.concatenate([sentiment_train_pred, sentiment_test_pred])
        })

        # Plot predictions
        fig = go.Figure()
        
        # Add traces with hover information
        fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Actual'],
            name='Actual',
            line=dict(color='black'),
            hovertemplate='Date: %{x}<br>Actual Price: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Linear Regression'],
            name='Linear Regression',
            line=dict(color='blue'),
            hovertemplate='Date: %{x}<br>Linear Regression: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Random Forest'],
            name='Random Forest',
            line=dict(color='green'),
            hovertemplate='Date: %{x}<br>Random Forest: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Gradient Boosting'],
            name='Gradient Boosting',
            line=dict(color='red'),
            hovertemplate='Date: %{x}<br>Gradient Boosting: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Sentiment Model'],
            name='Sentiment Model',
            yaxis='y2',
            line=dict(color='orange', dash='dot'),
            hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title='Bitcoin Price Predictions with Sentiment',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            yaxis2=dict(
                title='Sentiment Score',
                overlaying='y',
                side='right',
                range=[-1, 1]
            ),
            height=600,
            hovermode='x unified',  # Show all values for the same x-coordinate
            hoverdistance=100,      # Distance to show hover info
            spikedistance=1000      # Distance to show spike line
        )

        # Add hover spikes
        fig.update_xaxes(showspikes=True)
        fig.update_yaxes(showspikes=True)

        st.plotly_chart(fig, use_container_width=True)

        # Display metrics
        st.markdown("### 📊 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        try:
            # Convert predictions to numpy arrays with float dtype
            lr_test_pred = np.array(lr_test_pred, dtype=float)
            rf_test_pred = np.array(rf_test_pred, dtype=float)
            gb_test_pred = np.array(gb_test_pred, dtype=float)
            sentiment_test_pred = np.array(sentiment_test_pred, dtype=float)

            # Calculate metrics
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
            lr_r2 = r2_score(y_test, lr_test_pred)

            with col1:
                st.metric("Linear Regression RMSE", f"${lr_rmse:,.2f}")
                st.metric("Linear Regression R²", f"{lr_r2:.3f}")

            with col2:
                rf_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
                rf_r2 = r2_score(y_test, rf_test_pred)
                st.metric("Random Forest RMSE", f"${rf_rmse:,.2f}")
                st.metric("Random Forest R²", f"{rf_r2:.3f}")

            with col3:
                gb_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
                gb_r2 = r2_score(y_test, gb_test_pred)
                st.metric("Gradient Boosting RMSE", f"${gb_rmse:,.2f}")
                st.metric("Gradient Boosting R²", f"{gb_r2:.3f}")

            with col4:
                ensemble_rmse = np.sqrt(mean_squared_error(y_test, (lr_test_pred + rf_test_pred + gb_test_pred) / 3))
                ensemble_r2 = r2_score(y_test, (lr_test_pred + rf_test_pred + gb_test_pred) / 3)
                st.metric("Ensemble RMSE", f"${ensemble_rmse:,.2f}")
                st.metric("Ensemble R²", f"{ensemble_r2:.3f}")

        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            st.info("Please check if the model predictions are valid.")

        # Feature importance plot
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        fig = go.Figure(data=[
            go.Bar(x=feature_importance['Feature'], y=feature_importance['Importance'])
        ])
        fig.update_layout(
            title='Feature Importance (Including Sentiment)',
            xaxis_title='Features',
            yaxis_title='Importance',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add correlation analysis
        st.markdown("### 🔍 Feature Correlations")
        correlation_matrix = df_with_dates[features + ['btc_close']].corr()
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please analyze sentiment data first to enable model training with sentiment features.") 