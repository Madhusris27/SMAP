import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime
import plotly.graph_objects as go
import pickle

# Set page config
st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")

# Handle navigation state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"

def change_page(page_name):
    st.session_state['current_page'] = page_name

pages = ["Home", "Stock Prediction", "Analytics Dashboard", "About Project"]
st.sidebar.title("Navigation")

# Sidebar navigation
selected_page = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state['current_page']))
if selected_page != st.session_state['current_page']:
    st.session_state['current_page'] = selected_page
    st.rerun()

choice = st.session_state['current_page']

# ==========================================
# HOME PAGE
# ==========================================
if choice == "Home":
    st.title("Smart E-Commerce Analytics for Stock Market Prediction")
    st.markdown("#### This application uses machine learning to analyze stock market trends and provide buy recommendations.")
    
    st.markdown("---")
    
    # Navigation Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Stock Prediction", use_container_width=True):
            change_page("Stock Prediction")
            st.rerun()
    with col2:
        if st.button("📊 Analytics Dashboard", use_container_width=True):
            change_page("Analytics Dashboard")
            st.rerun()
    with col3:
        if st.button("ℹ️ About Project", use_container_width=True):
            change_page("About Project")
            st.rerun()
            
    st.markdown("---")
    
    # Small stock market trend chart
    st.subheader("Market Overview (S&P 500)")
    try:
        # Fetch S&P 500 data for an overview chart
        sp500 = yf.Ticker("^GSPC").history(period="3mo")
        st.line_chart(sp500['Close'])
    except Exception as e:
        st.warning("Could not load market overview data.")
        
    # Key features section
    st.markdown("### Key Features")
    st.markdown("""
    - **Machine Learning Prediction**
    - **Stock Market Data Analysis**
    - **Interactive Graphs**
    - **Real-time Stock Data**
    """)

# ==========================================
# STOCK PREDICTION PAGE
# ==========================================
elif choice == "Stock Prediction":
    st.title("Stock Prediction")
    st.write("Predict whether you should BUY or DO NOT BUY a stock based on current market data.")
    
    with st.form("prediction_form"):
        ticker = st.text_input("Stock Ticker Symbol (Example: AAPL, TSLA, MSFT, AMZN)", "AAPL")
        
        col1, col2 = st.columns(2)
        with col1:
            open_price = st.number_input("Open Price", min_value=0.0, format="%.2f", value=150.0)
            high_price = st.number_input("High Price", min_value=0.0, format="%.2f", value=155.0)
            low_price = st.number_input("Low Price", min_value=0.0, format="%.2f", value=149.0)
        with col2:
            close_price = st.number_input("Close Price", min_value=0.0, format="%.2f", value=152.0)
            volume = st.number_input("Volume", min_value=0, step=1000, value=10000000)
            
        submit = st.form_submit_button("Predict")
        
    if submit:
        if not ticker:
            st.error("Please enter a valid stock ticker.")
        else:
            with st.spinner("Fetching historical data and training machine learning model..."):
                try:
                    start_date = "2018-01-01"
                    end_date = datetime.date.today().strftime("%Y-%m-%d")
                    
                    # Fetching historical data
                    stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
                    
                    if stock_data.empty:
                        st.error(f"No data found for symbol '{ticker}'. Please check the ticker symbol and try again.")
                    else:
                        # Preprocessing: Remove missing values
                        stock_data.dropna(inplace=True)
                        
                        # Create target column: 1 (BUY) if next day close > current close, else 0
                        stock_data['Next_Close'] = stock_data['Close'].shift(-1)
                        stock_data['Target'] = (stock_data['Next_Close'] > stock_data['Close']).astype(int)
                        
                        # Drop the last row as it lacks a 'Next_Close' value
                        stock_data.dropna(inplace=True)
                        
                        # Features for model training
                        features = ['Open', 'High', 'Low', 'Close', 'Volume']
                        X = stock_data[features]
                        y = stock_data['Target']
                        
                        # Train test split (80% train, 20% test)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Load Random Forest Classifier from pickle file
                        try:
                            with open("stock_model.pkl", "rb") as f:
                                rf_model = pickle.load(f)
                            st.success("Loaded pre-trained model successfully!")
                        except Exception as e:
                            st.warning(f"Could not load pre-trained model. Training a new one... Error: {e}")
                            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                            rf_model.fit(X_train, y_train)
                        
                        # Evaluate model accuracy
                        y_pred = rf_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Predict based on user input
                        user_input_df = pd.DataFrame({
                            'Open': [open_price],
                            'High': [high_price],
                            'Low': [low_price],
                            'Close': [close_price],
                            'Volume': [volume]
                        })
                        prediction = rf_model.predict(user_input_df)[0]
                        
                        # Display Results matching the EXPECTED OUTPUT EXAMPLE
                        st.markdown("---")
                        st.markdown(f"**Stock:** {ticker.upper()}")
                        st.markdown(f"**Model Accuracy:** {accuracy * 100:.0f}%")
                        
                        if prediction == 1:
                            st.markdown("### Prediction Result: <span style='color:green'>BUY STOCK</span>", unsafe_allow_html=True)
                            st.success("Confidence Message: Based on historical price action and the provided parameters, the ML model anticipates a positive trend. However, always do your own research before investing.")
                            trend_text = "UPWARD"
                        else:
                            st.markdown("### Prediction Result: <span style='color:red'>DO NOT BUY STOCK</span>", unsafe_allow_html=True)
                            st.warning("Confidence Message: The ML model anticipates a downward or sideways movement based on the provided parameters. Proceed with caution.")
                            trend_text = "DOWNWARD"
                            
                        st.markdown(f"**Trend:** {trend_text}")
                        
                        # Display a stock price trend graph below prediction result
                        st.subheader(f"{ticker.upper()} Price Trend (Historical)")
                        st.line_chart(stock_data['Close'])
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

# ==========================================
# ANALYTICS DASHBOARD PAGE
# ==========================================
elif choice == "Analytics Dashboard":
    st.title("Analytics Dashboard")
    st.write("Visualize stock price trends and trading volumes.")
    
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", "AAPL")
    
    if ticker:
        with st.spinner(f"Loading analytics for {ticker.upper()}..."):
            try:
                # Fetch recent data for dashboard
                data = yf.Ticker(ticker).history(period="1y")
                
                if data.empty:
                    st.error("No data found for this ticker.")
                else:
                    # Current vs Previous price for trend indicator
                    current_price = float(data['Close'].iloc[-1])
                    prev_price = float(data['Close'].iloc[-2])
                    
                    if current_price > prev_price:
                        trend_indicator = "UPWARD TREND ⬆️"
                        color_mode = "normal"
                    elif current_price < prev_price:
                        trend_indicator = "DOWNWARD TREND ⬇️"
                        color_mode = "inverse"
                    else:
                        trend_indicator = "FLAT TREND ➖"
                        color_mode = "off"
                        
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Stock Price", f"${current_price:.2f}", f"{(current_price - prev_price):.2f}", delta_color=color_mode)
                    with col2:
                        st.markdown(f"### Trend Indicator: **{trend_indicator}**")
                        
                    # 1. Stock closing price trend graph
                    st.subheader("Stock Closing Price Trend Graph")
                    fig_close = go.Figure()
                    fig_close.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
                    fig_close.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_close, use_container_width=True)
                    
                    # 2. Moving average trend line
                    st.subheader("Moving Average Trend Line (50-Day & 200-Day)")
                    data['MA50'] = data['Close'].rolling(window=50).mean()
                    data['MA200'] = data['Close'].rolling(window=200).mean()
                    
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', opacity=0.3, line=dict(color='gray')))
                    fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='50-Day MA', line=dict(color='orange')))
                    fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA200'], mode='lines', name='200-Day MA', line=dict(color='red')))
                    fig_ma.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_ma, use_container_width=True)
                    
                    # 3. Stock trading volume chart
                    st.subheader("Stock Trading Volume Chart")
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'))
                    fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_vol, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Failed to load dashboard data: {e}")

# ==========================================
# ABOUT PROJECT PAGE
# ==========================================
elif choice == "About Project":
    st.title("About Project")
    
    st.markdown("### Project Objective")
    st.write("The objective of this project is to provide a beginner-friendly, analytical tool for stock market predictions. By leveraging historical stock data and machine learning, this application assists users in identifying market trends and making informed buy/sell decisions. It serves as an optimal solution for a final-year Artificial Intelligence and Machine Learning project.")
    
    st.markdown("### Technology Used")
    st.markdown("""
    - **Frontend / UI**: Streamlit (Python)
    - **Data Manipulation**: Pandas, NumPy
    - **Machine Learning**: Scikit-Learn
    - **Data Visualization**: Plotly, Streamlit Native Charts
    - **Data Source API**: Yahoo Finance API (`yfinance`)
    """)
    
    st.markdown("### Machine Learning Model")
    st.write("We use a **Random Forest Classifier** to predict stock movements. The dataset is split into 80% training data and 20% testing data. Features like Open, High, Low, Close, and Volume are used to predict if the next day's closing price will be higher than the current day's close.")
    
    st.markdown("### Dataset Source")
    st.write("The dataset comes from historical stock market data dynamically fetched using the **Yahoo Finance API** (`yfinance`). The application retrieves data from 2018 up to the present day for accurate, long-term pattern recognition.")
