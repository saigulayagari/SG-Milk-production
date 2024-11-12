import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.model_selection import train_test_split

# Title and subtitle
st.title("📈 Monthly Milk Production Dashboard")
st.write("Visualize and forecast milk production trends")

# Sidebar for file upload and plot options
st.sidebar.header("📂 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load data
    try:
        data = pd.read_csv(uploaded_file, index_col='Month', parse_dates=True)
        data.rename(columns={"Monthly milk production (pounds per cow)": "Milk Production"}, inplace=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    # Display dataset preview
    st.write("## Dataset Preview")
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
    st.write("## Milk Production with Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Milk Production'], mode='lines', name='Milk Production', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], mode='lines', name=f'{sma_window}-Month SMA', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_shifted'], mode='lines', name=f'{sma_window}-Month SMA Shifted', line=dict(color='red', dash='dash')))
    fig.add_tra
