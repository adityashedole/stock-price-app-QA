import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN optimization for TensorFlow

# Import necessary libraries
import numpy as np
import pandas as pd
import datetime
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import load_model # Not used in the current code, can be removed if not planned for future use
import ta  # Technical analysis library
from textblob import TextBlob  # For sentiment analysis
from fpdf import FPDF
import time
from polygon import RESTClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from forex_python.converter import CurrencyRates, RatesNotAvailableError # Import for live currency conversion

# Set Streamlit page configuration as the very first Streamlit command
st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Polygon client
# Ensure POLYGON_API_KEY is securely managed in a real application (e.g., Streamlit secrets)
POLYGON_API_KEY = "G1m3oosgg3cnxdl9VfEXL2uqj49NzfBk"
client = RESTClient(POLYGON_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CurrencyRates for forex-python
c = CurrencyRates()

@st.cache_data(ttl=3600, show_spinner="Fetching live exchange rates...") # Cache for 1 hour
def get_live_exchange_rates():
    """
    Fetches live exchange rates relative to USD using forex-python.
    Caches the results to avoid frequent API calls.
    """
    try:
        # Get all rates relative to USD
        rates = c.get_rates('USD')
        # Add USD itself to the rates dictionary
        rates['USD'] = 1.0
        logger.info(f"Successfully fetched live exchange rates: {rates.keys()}")
        return rates
    except RatesNotAvailableError:
        logger.error("Live exchange rates are not available. Using fallback rates.")
        return {
            "USD": 1.0,
            "EUR": 0.92,
            "GBP": 0.79,
            "JPY": 156.90,
            "INR": 83.50
        }
    except Exception as e:
        logger.error(f"Error fetching live exchange rates: {e}. Using fallback rates.")
        return {
            "USD": 1.0,
            "EUR": 0.92,
            "GBP": 0.79,
            "JPY": 156.90,
            "INR": 83.50
        }

# Fetch exchange rates once at the start of the app
EXCHANGE_RATES = get_live_exchange_rates()

def get_currency_symbol(currency_code):
    """
    Returns the common symbol for a given currency code.
    """
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "INR": "₹",
        # Add more currency symbols as needed
    }
    return symbols.get(currency_code, "") # Return empty string if symbol not found

# ----------------------
# DATA FETCHING FUNCTIONS (POLYGON.IO)
# ----------------------

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def get_polygon_data(ticker, start_date, end_date, multiplier=1, timespan="day"):
    """
    Fetch stock data from Polygon.io with caching and error handling.
    """
    try:
        # Convert dates to string format for Polygon API
        start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime.date) else start_date
        end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime.date) else end_date
        
        # Fetch aggregates (OHLCV data)
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_str,
            to=end_str,
            adjusted=True,
            sort="asc",
            limit=50000
        )
        
        if not aggs or len(aggs) == 0:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data_list = []
        for agg in aggs:
            data_list.append({
                'Date': pd.to_datetime(agg.timestamp, unit='ms'),
                'Open': agg.open,
                'High': agg.high,
                'Low': agg.low,
                'Close': agg.close,
                'Volume': agg.volume
            })
        
        df = pd.DataFrame(data_list)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Successfully fetched {len(df)} records for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_ticker_details(ticker):
    """
    Get ticker details from Polygon.io.
    """
    try:
        details = client.get_ticker_details(ticker)
        return {
            'name': getattr(details, 'name', ticker),
            'description': getattr(details, 'description', ''),
            'market_cap': getattr(details, 'market_cap', None),
            'primary_exchange': getattr(details, 'primary_exchange', 'Unknown')
        }
    except Exception as e:
        logger.error(f"Error fetching ticker details for {ticker}: {e}")
        return {'name': ticker, 'description': '', 'market_cap': None, 'primary_exchange': 'Unknown'}

@st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
def get_multiple_tickers_data(tickers, start_date, end_date):
    """
    Fetch data for multiple tickers concurrently for faster performance.
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all requests
        future_to_ticker = {
            executor.submit(get_polygon_data, ticker, start_date, end_date): ticker 
            for ticker in tickers
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if not data.empty:
                    results[ticker] = data
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
    
    return results

@st.cache_data(show_spinner=False)
def load_popular_tickers():
    """
    Return list of popular US stock tickers for demo purposes.
    In production, you could fetch this from Polygon's tickers endpoint.
    """
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'SPOT',
        'ZOOM', 'SQ', 'SHOP', 'ROKU', 'TWTR', 'SNAP', 'PINS', 'LYFT'
    ]

def load_stock_data_optimized(stock, start, end):
    """
    Optimized stock data loading with technical indicators.
    """
    # Get the raw stock data
    data = get_polygon_data(stock, start, end)
    
    if not data.empty:
        # Add technical indicators
        data = add_indicators_optimized(data)
    
    return data

# ----------------------
# OPTIMIZED DATA PROCESSING FUNCTIONS
# ----------------------

@st.cache_data(show_spinner=False)
def add_indicators_optimized(df):
    """
    Optimized technical indicators calculation using vectorized operations.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    try:
        # Verify required column exists
        if 'Close' not in df.columns:
            logger.error("No Close price column found")
            return df
        
        # Use pandas rolling for faster calculations
        close_prices = df['Close']
        
        # RSI calculation (optimized)
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages (vectorized)
        df['MA20'] = close_prices.rolling(20).mean()
        df['MA50'] = close_prices.rolling(50).mean()
        df['MA200'] = close_prices.rolling(200).mean()
        
        # MACD calculation (vectorized)
        exp1 = close_prices.ewm(span=12).mean()
        exp2 = close_prices.ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands (vectorized)
        bb_period = 20
        bb_std = 2
        bb_ma = close_prices.rolling(bb_period).mean()
        bb_std_dev = close_prices.rolling(bb_period).std()
        df['BB_Upper'] = bb_ma + (bb_std_dev * bb_std)
        df['BB_Lower'] = bb_ma - (bb_std_dev * bb_std)
        df['BB_Middle'] = bb_ma
        
        # Volume indicators
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price change indicators
        df['Price_Change'] = close_prices.pct_change()
        df['Price_Change_MA'] = df['Price_Change'].rolling(10).mean()
        
        # Volatility (rolling standard deviation)
        df['Volatility'] = close_prices.rolling(20).std()
        
        # Fill NaN values efficiently
        df = df.fillna(method='bfill').fillna(method='ffill')
        
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        st.error(f"Error adding indicators: {e}")
    
    return df

# ----------------------
# OPTIMIZED MODEL FUNCTIONS
# ----------------------

def create_features_optimized(data, lookback=10):
    """
    Optimized feature creation with more predictive features.
    """
    if 'Close' not in data.columns or data.empty:
        return None, None
    
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['Close'] = data['Close']
    features['High'] = data['High']
    features['Low'] = data['Low']
    features['Volume'] = data['Volume'] if 'Volume' in data.columns else 0
    
    # Lagged features (vectorized)
    for i in range(1, lookback + 1):
        features[f'Close_lag_{i}'] = data['Close'].shift(i)
        features[f'Return_lag_{i}'] = data['Close'].pct_change().shift(i)
    
    # Technical indicators
    tech_indicators = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'MA20', 'MA50']
    for indicator in tech_indicators:
        if indicator in data.columns:
            features[indicator] = data[indicator]
    
    # Volatility features
    if 'Volatility' in data.columns:
        features['Volatility'] = data['Volatility']
    
    # Price ratios and relationships
    if all(col in data.columns for col in ['High', 'Low', 'Close']):
        features['HL_Ratio'] = data['High'] / data['Low']
        features['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
    
    # Drop rows with NaN values
    features = features.dropna()
    
    if len(features) < 2:
        return None, None
    
    # Prepare X and y
    X = features.drop('Close', axis=1)
    y = features['Close']
    
    return X, y

def predict_classical_optimized(model, data, test_size=0.2):
    """
    Optimized classical ML prediction with better feature engineering.
    """
    try:
        if 'Close' not in data.columns or data.empty:
            logger.error("No Close price data available")
            return np.array([]), np.array([]), float('inf')
        
        # Create optimized features
        X, y = create_features_optimized(data)
        if X is None or y is None:
            logger.error("Failed to create features")
            return np.array([]), np.array([]), float('inf')
        
        # Split data
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        if len(X_train) < 10 or len(X_test) < 1:
            logger.error(f"Insufficient data: train={len(X_train)}, test={len(X_test)}")
            return np.array([]), np.array([]), float('inf')
        
        # Scale features for better performance
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        
        return y_test.values, y_pred, mse
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return np.array([]), np.array([]), float('inf')

def get_model_optimized(name):
    """
    Return optimized models with better hyperparameters.
    """
    try:
        if name == "Linear Regression":
            return LinearRegression()
        elif name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=50,  # Reduced for speed
                max_depth=10,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        elif name == "XGBoost":
            return xgb.XGBRegressor(
                n_estimators=50,  # Reduced for speed
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        else:
            return LinearRegression()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return LinearRegression()

# ----------------------
# UTILITY FUNCTIONS
# ----------------------

def get_sentiment_score(text):
    """
    Perform sentiment analysis with caching.
    """
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return 0.0

def generate_pdf_optimized(stock, summary, currency_symbol=""):
    """
    Optimized PDF generation with currency symbol.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Stock Report: {stock}", ln=True)
        
        for key, value in summary.items():
            if "Price" in key or "High" in key or "Low" in key or "Volume" in key:
                if isinstance(value, (int, float)):
                    value_str = f"{currency_symbol}{value:,.2f}"
                else:
                    value_str = str(value)
            elif isinstance(value, (int, float)):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            text = f"{key}: {value_str}"
            if len(text) > 80:
                text = text[:77] + "..."
                
            pdf.cell(200, 10, txt=text, ln=True)
            
        pdf_path = f"{stock}_report.pdf"
        pdf.output(pdf_path)
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None

# ----------------------
# STREAMLIT UI FUNCTIONS
# ----------------------

def login():
    """
    Login screen with improved UX.
    """
    st.title("📊 Stock Analysis Platform")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid credentials. Use 'admin' for both username and password.")

def display_portfolio_tracker_optimized(tickers, selected_currency):
    """
    Optimized portfolio tracker with concurrent data fetching and currency conversion.
    """
    st.sidebar.title("📈 Portfolio Tracker")
    
    default_symbols = ["AAPL", "MSFT", "GOOGL"]
    portfolio = st.sidebar.multiselect(
        "Select stocks for watchlist", 
        tickers, 
        default=default_symbols[:3]
    )
    
    currency_symbol = get_currency_symbol(selected_currency)

    if portfolio:
        with st.sidebar.container():
            st.write(f"**Current Prices ({selected_currency})**")
            
            # Fetch data for all portfolio stocks concurrently
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=7)
            
            portfolio_data = get_multiple_tickers_data(portfolio, start_date, end_date)
            
            for symbol in portfolio:
                if symbol in portfolio_data and not portfolio_data[symbol].empty:
                    data = portfolio_data[symbol]
                    if len(data) >= 2:
                        current_price_usd = data['Close'].iloc[-1]
                        prev_price_usd = data['Close'].iloc[-2]

                        # Apply currency conversion
                        conversion_rate = EXCHANGE_RATES.get(selected_currency, 1.0)
                        current_price = current_price_usd * conversion_rate
                        prev_price = prev_price_usd * conversion_rate

                        change = current_price - prev_price
                        pct_change = (change / prev_price) * 100
                        
                        st.sidebar.metric(
                            label=symbol,
                            value=f"{currency_symbol}{current_price:.2f}",
                            delta=f"{pct_change:.2f}%"
                        )
                    else:
                        current_price_usd = data['Close'].iloc[-1]
                        conversion_rate = EXCHANGE_RATES.get(selected_currency, 1.0)
                        current_price = current_price_usd * conversion_rate
                        st.sidebar.metric(
                            label=symbol,
                            value=f"{currency_symbol}{current_price:.2f}"
                        )
                else:
                    st.sidebar.warning(f"{symbol}: No data")

def dashboard():
    """
    Optimized main dashboard with better performance and currency conversion.
    """
    st.title("📊 Advanced Stock Analysis Dashboard")
    
    # Header with logout
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Currency selection
        selected_currency = st.selectbox(
            "Currency",
            list(EXCHANGE_RATES.keys()),
            index=list(EXCHANGE_RATES.keys()).index("USD"), # Default to USD
            help="Select the currency for price display"
        )
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    currency_symbol = get_currency_symbol(selected_currency)
    
    # Load tickers and display portfolio
    tickers = load_popular_tickers()
    display_portfolio_tracker_optimized(tickers, selected_currency)
    
    # Stock selection with search
    st.subheader("🔍 Stock Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        stock_name = st.selectbox(
            'Select Stock Symbol', 
            tickers, 
            index=0,
            help="Choose a stock ticker to analyze"
        )
    with col2:
        start_date = st.date_input(
            "Start Date", 
            value=datetime.date.today() - datetime.timedelta(days=365)
        )
    with col3:
        end_date = st.date_input("End Date", value=datetime.date.today())
    
    if stock_name:
        # Show loading progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Get ticker details
            status_text.text("Fetching company information...")
            progress_bar.progress(20)
            ticker_details = get_ticker_details(stock_name)
            
            # Step 2: Load stock data
            status_text.text("Loading stock data...")
            progress_bar.progress(40)
            data = load_stock_data_optimized(stock_name, start_date, end_date)
            
            if data.empty:
                st.error("No data found for the selected stock and date range.")
                return
            
            progress_bar.progress(60)
            status_text.text("Processing technical indicators...")
            
            # Apply currency conversion to relevant columns
            conversion_rate = EXCHANGE_RATES.get(selected_currency, 1.0)
            data_converted = data.copy()
            for col in ['Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'MA200', 'BB_Upper', 'BB_Lower', 'BB_Middle']:
                if col in data_converted.columns:
                    data_converted[col] = data_converted[col] * conversion_rate

            # Clear progress indicators
            progress_bar.progress(100)
            status_text.text("Data loaded successfully!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Display company info
            st.header(f"📈 {ticker_details['name']} ({stock_name})")
            if ticker_details['description']:
                st.markdown(f"*{ticker_details['description'][:200]}...*")
            
            # Key metrics row
            if not data_converted.empty and 'Close' in data_converted.columns:
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = data_converted['Close'].iloc[-1]
                high_52w = data_converted['High'].max()
                low_52w = data_converted['Low'].min()
                avg_volume = data_converted['Volume'].mean() if 'Volume' in data_converted.columns else 0 # Volume is not currency dependent
                
                with col1:
                    st.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
                with col2:
                    st.metric("52W High", f"{currency_symbol}{high_52w:.2f}")
                with col3:
                    st.metric("52W Low", f"{currency_symbol}{low_52w:.2f}")
                with col4:
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
            
            # Analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Analysis", "🔧 Technical Indicators", "🔮 Predictions", "💭 Sentiment"])
            
            # Tab 1: Price Analysis
            with tab1:
                if 'Close' in data_converted.columns:
                    # Price chart with moving averages
                    st.subheader(f"Price Chart with Moving Averages ({selected_currency})")
                    
                    price_data = pd.DataFrame({
                        'Close': data_converted['Close'],
                        '20-Day MA': data_converted['MA20'] if 'MA20' in data_converted.columns else None,
                        '50-Day MA': data_converted['MA50'] if 'MA50' in data_converted.columns else None,
                        '200-Day MA': data_converted['MA200'] if 'MA200' in data_converted.columns else None
                    }).dropna()
                    
                    st.line_chart(price_data)
                    
                    # Volume analysis
                    if 'Volume' in data_converted.columns:
                        st.subheader("Trading Volume")
                        st.bar_chart(data_converted['Volume'])
                    
                    # Price statistics
                    st.subheader(f"Price Statistics ({selected_currency})")
                    stats_df = data_converted[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
                    st.dataframe(stats_df, use_container_width=True)
            
            # Tab 2: Technical Indicators
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI
                    if 'RSI' in data_converted.columns:
                        st.subheader("RSI (Relative Strength Index)")
                        rsi_data = pd.DataFrame({
                            'RSI': data_converted['RSI'],
                            'Overbought': 70,
                            'Oversold': 30
                        })
                        st.line_chart(rsi_data)
                        
                        current_rsi = data_converted['RSI'].iloc[-1]
                        if current_rsi > 70:
                            st.warning(f"RSI: {current_rsi:.1f} - Potentially Overbought")
                        elif current_rsi < 30:
                            st.success(f"RSI: {current_rsi:.1f} - Potentially Oversold")
                        else:
                            st.info(f"RSI: {current_rsi:.1f} - Neutral")
                
                with col2:
                    # MACD
                    if 'MACD' in data_converted.columns:
                        st.subheader("MACD")
                        macd_data = pd.DataFrame({
                            'MACD': data_converted['MACD'],
                            'Signal': data_converted['MACD_Signal'] if 'MACD_Signal' in data_converted.columns else None
                        }).dropna()
                        st.line_chart(macd_data)
                
                # Bollinger Bands
                if all(col in data_converted.columns for col in ['BB_Upper', 'BB_Lower', 'Close']):
                    st.subheader("Bollinger Bands")
                    bb_data = pd.DataFrame({
                        'Price': data_converted['Close'],
                        'Upper Band': data_converted['BB_Upper'],
                        'Lower Band': data_converted['BB_Lower'],
                        'Middle Band': data_converted['BB_Middle'] if 'BB_Middle' in data_converted.columns else None
                    }).dropna()
                    st.line_chart(bb_data)
            
            # Tab 3: Predictions
            with tab3:
                st.subheader("🔮 Price Prediction Models")
                
                model_choice = st.selectbox(
                    "Select Prediction Model",
                    ["Linear Regression", "Random Forest", "XGBoost"],
                    help="Choose a machine learning model for price prediction"
                )
                if len(data) < 50:
                    st.warning("Need at least 50 days of data for reliable predictions.")
                else:
                    with st.spinner(f"Training {model_choice} model..."):
                        model = get_model_optimized(model_choice)
                        # Pass original data to prediction model, as it expects USD values for training
                        y_actual_usd, y_pred_usd, mse = predict_classical_optimized(model, data)
                        
                        if len(y_actual_usd) > 0:
                            # Convert predictions to selected currency for display
                            y_actual = y_actual_usd * conversion_rate
                            y_pred = y_pred_usd * conversion_rate

                            # Create prediction chart
                            test_dates = data.index[-len(y_actual):]
                            pred_df = pd.DataFrame({
                                'Actual': y_actual,
                                'Predicted': y_pred[:len(y_actual)]
                            }, index=test_dates)
                            
                            st.subheader(f"Actual vs Predicted ({model_choice}) ({selected_currency})")
                            st.line_chart(pred_df)
                            
                            # Model performance metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MSE", f"{mse:.2f}") # MSE is calculated on original scale
                            with col2:
                                rmse = np.sqrt(mse)
                                st.metric("RMSE", f"{currency_symbol}{rmse:.2f}") # RMSE is in the currency unit
                            with col3:
                                mape = np.mean(np.abs((y_actual_usd - y_pred_usd) / y_actual_usd)) * 100
                                st.metric("MAPE", f"{mape:.1f}%")
                        else:
                            st.error("Prediction failed. Please try a different time range.")
            
            # Tab 4: Sentiment Analysis
            with tab4:
                st.subheader("📰 News Sentiment Analysis")
                st.info("This is a demo sentiment analysis using sample headlines.")
                
                # Sample news headlines
                sample_news = [
                    f"{ticker_details['name']} reports strong quarterly earnings",
                    f"Analysts upgrade {stock_name} price target",
                    f"{ticker_details['name']} announces new product launch",
                    f"Market volatility affects {stock_name} trading",
                    f"{ticker_details['name']} expands into new markets"
                ]
                
                sentiment_scores = []
                for headline in sample_news:
                    score = get_sentiment_score(headline)
                    sentiment_scores.append(score)
                    
                    sentiment_label = "Neutral"
                    if score > 0.1:
                        sentiment_label = "Positive 😊"
                    elif score < -0.1:
                        sentiment_label = "Negative 😞"
                    
                    st.write(f"**{headline}**")
                    st.write(f"Sentiment: {sentiment_label} (Score: {score:.2f})")
                    st.write("---")
                
                # Overall sentiment
                avg_sentiment = np.mean(sentiment_scores)
                st.subheader("Overall Sentiment Summary")
                if avg_sentiment > 0.1:
                    st.success(f"Positive sentiment detected (Score: {avg_sentiment:.2f})")
                elif avg_sentiment < -0.1:
                    st.error(f"Negative sentiment detected (Score: {avg_sentiment:.2f})")
                else:
                    st.info(f"Neutral sentiment (Score: {avg_sentiment:.2f})")
            
            # Report generation
            st.subheader("📄 Generate Analysis Report")
            if st.button("Generate PDF Report", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    # Use converted data for the report summary
                    summary = {
                        "Company": ticker_details['name'],
                        "Symbol": stock_name,
                        "Exchange": ticker_details['primary_exchange'],
                        "Analysis Date": datetime.date.today().strftime("%Y-%m-%d"),
                        "Data Range": f"{start_date} to {end_date}",
                        "Currency": selected_currency,
                        "Current Price": data_converted['Close'].iloc[-1],
                        "52-Week High": data_converted['High'].max(),
                        "52-Week Low": data_converted['Low'].min(),
                        "Average Volume": data['Volume'].mean() if 'Volume' in data.columns else 0, # Volume is not currency dependent
                        "Model Used": model_choice,
                        "Prediction Accuracy (RMSE)": np.sqrt(mse) if 'mse' in locals() else "N/A"
                    }
                    
                    if 'RSI' in data.columns:
                        summary["Current RSI"] = data['RSI'].iloc[-1]
                    
                    pdf_path = generate_pdf_optimized(stock_name, summary, currency_symbol)
                    
                    if pdf_path:
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 Download PDF Report",
                                f,
                                file_name=f"{stock_name}_analysis_report_{selected_currency}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    else:
                        st.error("Failed to generate PDF report.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Dashboard error: {e}")
    
    else:
        st.info("👆 Please select a stock ticker from the dropdown to begin analysis.")

# ----------------------
# MAIN APPLICATION
# ----------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    dashboard()
else:
    login()
