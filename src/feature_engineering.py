"""
Feature engineering for stock price prediction
Create technical indicators and features
"""
import pandas as pd
import numpy as np
import ta  # Technical Analysis library


class FeatureEngineer:
    """Create technical indicators and features for ML models"""
    
    def __init__(self, data):
        """
        Initialize with stock data
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data
        """
        self.data = data.copy()
    
    def add_technical_indicators(self):
        """Add common technical indicators"""
        
        # Simple Moving Averages
        self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        self.data['EMA_12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        self.data['RSI'] = ta.momentum.rsi(self.data['Close'], window=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_signal'] = macd.macd_signal()
        self.data['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(self.data['Close'], window=20, window_dev=2)
        self.data['BB_upper'] = bollinger.bollinger_hband()
        self.data['BB_lower'] = bollinger.bollinger_lband()
        self.data['BB_middle'] = bollinger.bollinger_mavg()
        
        # Average True Range (ATR)
        self.data['ATR'] = ta.volatility.average_true_range(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close']
        )
        
        # Volume indicators
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        
        print("✓ Technical indicators added successfully")
        return self.data
    
    def add_price_features(self):
        """Add price-based features"""
        
        # Price changes
        self.data['Price_Change'] = self.data['Close'].diff()
        self.data['Price_Change_Pct'] = self.data['Close'].pct_change() * 100
        
        # High-Low range
        self.data['HL_Range'] = self.data['High'] - self.data['Low']
        self.data['HL_Range_Pct'] = (self.data['HL_Range'] / self.data['Close']) * 100
        
        # Close-Open range
        self.data['CO_Range'] = self.data['Close'] - self.data['Open']
        
        print("✓ Price features added successfully")
        return self.data
    
    def create_target_variable(self, forecast_days=1):
        """
        Create target variable: 1 if price goes up, 0 if price goes down
        
        Args:
            forecast_days (int): Number of days to forecast
        
        Returns:
            pd.DataFrame: Data with target variable
        """
        self.data['Future_Close'] = self.data['Close'].shift(-forecast_days)
        self.data['Target'] = (self.data['Future_Close'] > self.data['Close']).astype(int)
        
        # Remove rows with NaN in Target
        self.data = self.data.dropna()
        
        print(f"✓ Target variable created (predicting {forecast_days} day(s) ahead)")
        print(f"  Up: {self.data['Target'].sum()} | Down: {len(self.data) - self.data['Target'].sum()}")
        
        return self.data
    
    def get_features_for_model(self):
        """Get feature columns for ML model"""
        
        feature_columns = [
            'SMA_10', 'SMA_20', 'SMA_50', 
            'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
            'BB_upper', 'BB_lower', 'BB_middle',
            'ATR', 'Volume_SMA',
            'Price_Change', 'Price_Change_Pct', 
            'HL_Range', 'HL_Range_Pct', 'CO_Range'
        ]
        
        return [col for col in feature_columns if col in self.data.columns]
    
    def get_processed_data(self):
        """Return cleaned and processed data"""
        return self.data.dropna()


# Example usage
if __name__ == "__main__":
    from data_fetcher import StockDataFetcher
    
    # Fetch data
    fetcher = StockDataFetcher()
    data = fetcher.fetch_stock_data('RELIANCE.NS', start_date='2022-01-01')
    
    if data is not None:
        # Create features
        engineer = FeatureEngineer(data)
        engineer.add_technical_indicators()
        engineer.add_price_features()
        processed_data = engineer.create_target_variable()
        
        print("\nProcessed data shape:", processed_data.shape)
        print("\nFeatures available:")
        print(engineer.get_features_for_model())
        print("\nFirst 5 rows with features:")
        print(processed_data.head())
