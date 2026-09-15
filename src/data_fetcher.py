"""
Fetch historical stock data from Indian stock market
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class StockDataFetcher:
    """Fetch stock data for Indian stocks"""
    
    def __init__(self):
        self.data = None
    
    def fetch_stock_data(self, symbol, start_date=None, end_date=None, interval='1d'):
        """
        Fetch historical stock data for Indian stocks
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NS', 'INFY.NS', 'TCS.NS')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d' for daily, '1h' for hourly, etc.)
        
        Returns:
            pd.DataFrame: DataFrame with OHLC data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            self.data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            self.data = self.data.reset_index()
            print(f"✓ Successfully fetched data for {symbol}")
            print(f"  Date range: {start_date} to {end_date}")
            print(f"  Total records: {len(self.data)}")
            return self.data
        except Exception as e:
            print(f"✗ Error fetching data: {str(e)}")
            return None
    
    def get_latest_price(self, symbol):
        """Get the latest closing price"""
        try:
            data = yf.download(symbol, period='1d')
            return data['Close'].iloc[-1]
        except Exception as e:
            print(f"Error getting latest price: {str(e)}")
            return None
    
    def save_data(self, filepath):
        """Save fetched data to CSV"""
        if self.data is not None:
            self.data.to_csv(filepath, index=False)
            print(f"✓ Data saved to {filepath}")
        else:
            print("✗ No data to save")


# Example usage
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    
    # Fetch data for Reliance Industries
    data = fetcher.fetch_stock_data('RELIANCE.NS', 
                                     start_date='2022-01-01', 
                                     end_date='2024-01-01')
    
    if data is not None:
        print("\nFirst 5 rows:")
        print(data.head())
        print("\nData shape:", data.shape)
        
        # Save data
        fetcher.save_data('reliance_data.csv')
