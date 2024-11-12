import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.model_selection import train_test_split

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        text-align: center; 
        color: #4A90E2; 
        font-size: 45px; 
        font-weight: bold;
    }
    .subtitle {
        text-align: center; 
        color: #4A4A4A; 
        font-size: 24px; 
        font-style: italic;
        margin-top: -10px;
    }
    .section-header {
        color: #E94B3C; 
        font-size: 30px; 
        font-weight: bold;
    }
    .divider {
        border-top: 3px solid #E94B3C; 
        margin-top: 20px; 
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown("<h1 class='title'>🐄 Monthly Milk Production Dashboard 🐄</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Visualize and forecast milk production trends</p>", unsafe_allow_html=True)

# Sidebar for file upload and plot options
st.sidebar.header("📂 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file, index_col='Month', parse_dates=True)
    data.rename(columns={"Monthly milk production (pounds per cow)": "Milk Production"}, inplace=True)

    # Display dataset preview
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header'>Dataset Preview</h3>", unsafe_allow_html=True)
    st.write(data.head())

    # Sidebar options for SMA and EMA selections
    sma_window = st.sidebar.slider("Select SMA Window", 1, 24, 12)
    ema_span = st.sidebar.slider("Select EMA Span", 1, 24, 12)
    custom_ema_alpha = st.sidebar.slider("Select Custom EMA Alpha", 0.1, 0.9, 0.6)

    # Calculations: SMA and EMA
    data['SMA'] = data['Milk Production'].rolling(window=sma_window).mean()
    data['SMA_shifted'] = data['SMA'].shift(1)
    data['EMA'] = data['Milk Production'].ewm(span=ema_span, adjust=False).mean()
    data['EMA_shifted'] = data['EMA'].shift(1)
    data['Custom_EMA'] = data['Milk Production'].ewm(alpha=custom_ema_alpha, adjust=False).mean()
    data['Custom_EMA_shifted'] = data['Custom_EMA'].shift(1)

    # Line chart for Milk Production with Moving Averages
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header'>Milk Production with Moving Averages</h3>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Milk Production'], mode='lines', name='Milk Production', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], mode='lines', name=f'{sma_window}-Month SMA', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_shifted'], mode='lines', name=f'{sma_window}-Month SMA Shifted', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], mode='lines', name=f'{ema_span}-Month EMA', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_shifted'], mode='lines', name=f'{ema_span}-Month EMA Shifted', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['Custom_EMA'], mode='lines', name=f'Custom EMA (alpha={custom_ema_alpha})', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['Custom_EMA_shifted'], mode='lines', name=f'Custom EMA Shifted (alpha={custom_ema_alpha})', line=dict(color='purple', dash='dash')))
    fig.update_layout(
        title='Monthly Milk Production with Moving Averages',
        xaxis_title='Month', 
        yaxis_title='Milk Production (pounds)',
        template='plotly_dark'
    )
    st.plotly_chart(fig)

    # Prepare data for forecasting
    data.index = pd.date_range(start=data.index[0], periods=len(data), freq='MS')
    train, test = train_test_split(data, test_size=0.2, shuffle=False)

    # Simple Exponential Smoothing model
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-header'>Simple Exponential Smoothing Forecast</h3>", unsafe_allow_html=True)
    model = SimpleExpSmoothing(train['Milk Production'])
    fitted_model = model.fit(smoothing_level=0.1, optimized=False)

    # Forecasting
    forecast = fitted_model.forecast(steps=len(test))
    forecast_df = pd.DataFrame({'Actual': test['Milk Production'], 'Forecast': forecast}, index=test.index)

    # Forecasting Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Actual'], mode='lines', name='Actual', line=dict(color='black', width=2)))
    fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='royalblue', dash='dash', width=2)))
    fig2.update_layout(
        title='Milk Production Forecast vs Actual',
        xaxis_title='Month', 
        yaxis_title='Milk Production (pounds)',
        template='plotly_dark'
    )
    st.plotly_chart(fig2)

    # Display forecast data
    st.write("Forecast Data")
    st.write(forecast_df)
else:
    st.write("Please upload a CSV file to view the analysis.")
