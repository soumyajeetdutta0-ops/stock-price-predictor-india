# 📈 Stock Price Predictor - Indian Market

A Python machine learning application to predict when stock prices in the Indian market will **rise** or **fall**.

## 🎯 Features

- **Real-time Data Fetching**: Download historical stock data from Indian markets (NSE)
- **Technical Analysis**: 15+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **Machine Learning Models**: Random Forest, Logistic Regression, SVM classifiers
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score metrics
- **Prediction**: Predict price direction (UP/DOWN) for any Indian stock

## 📊 Supported Indian Stocks

All NSE stocks with `.NS` suffix:
- **RELIANCE.NS** - Reliance Industries
- **INFY.NS** - Infosys
- **TCS.NS** - Tata Consultancy Services
- **WIPRO.NS** - Wipro
- **HDFC.NS** - HDFC Bank
- And many more...

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/soumyajeetdutta0-ops/stock-price-predictor-india.git
cd stock-price-predictor-india
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from src.data_fetcher import StockDataFetcher
from src.feature_engineering import FeatureEngineer
from src.model import StockPricePredictor

# Step 1: Fetch data
fetcher = StockDataFetcher()
data = fetcher.fetch_stock_data('RELIANCE.NS', start_date='2022-01-01')

# Step 2: Create features
engineer = FeatureEngineer(data)
engineer.add_technical_indicators()
engineer.add_price_features()
processed_data = engineer.create_target_variable(forecast_days=1)

# Step 3: Train model
predictor = StockPricePredictor(model_type='random_forest')
features = engineer.get_features_for_model()
predictor.prepare_data(processed_data, features)
predictor.train()

# Step 4: Evaluate
results = predictor.evaluate()

# Step 5: Make predictions
latest_features = processed_data.iloc[-1][features].values
prediction = predictor.predict(latest_features)
print(f"Stock will go: {prediction['prediction']}")
```

## 📁 Project Structure

```
stock-price-predictor-india/
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── src/
│   ├── data_fetcher.py       # Fetch stock data from Yahoo Finance
│   ├── feature_engineering.py # Create technical indicators & features
│   ├── model.py              # ML models for prediction
│   └── predictor.py          # Main prediction script (coming soon)
├── notebooks/
│   └── exploratory.ipynb     # Jupyter notebook for exploration
└── data/
    └── (Stock data will be saved here)
```

## 🔧 Modules

### 1. **data_fetcher.py**
Fetches historical stock data using yfinance.

```python
from src.data_fetcher import StockDataFetcher

fetcher = StockDataFetcher()
data = fetcher.fetch_stock_data('RELIANCE.NS', 
                                 start_date='2022-01-01',
                                 end_date='2024-01-01')
```

### 2. **feature_engineering.py**
Creates technical indicators and features for ML models.

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(data)
engineer.add_technical_indicators()  # SMA, EMA, RSI, MACD, etc.
engineer.add_price_features()        # Price changes, ranges
processed_data = engineer.create_target_variable(forecast_days=1)
```

**Available Indicators:**
- Moving Averages: SMA (10, 20, 50), EMA (12, 26)
- Momentum: RSI, MACD
- Volatility: Bollinger Bands, ATR
- Volume indicators

### 3. **model.py**
Machine learning models for classification.

```python
from src.model import StockPricePredictor

predictor = StockPricePredictor(model_type='random_forest')
predictor.prepare_data(processed_data, features)
predictor.train()
results = predictor.evaluate()
prediction = predictor.predict(new_features)
```

**Supported Models:**
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

## 📈 How It Works

1. **Data Collection**: Download historical OHLC data
2. **Feature Engineering**: Calculate 15+ technical indicators
3. **Data Preparation**: Split into train/test sets (80/20)
4. **Model Training**: Fit ML classifier on training data
5. **Evaluation**: Test accuracy, precision, recall, F1-score
6. **Prediction**: Predict UP or DOWN for next day

## 📊 Example Output

```
✓ Successfully fetched data for RELIANCE.NS
  Date range: 2022-01-01 to 2024-01-01
  Total records: 504

✓ Technical indicators added successfully
✓ Price features added successfully
✓ Target variable created (predicting 1 day(s) ahead)
  Up: 252 | Down: 252

✓ Model trained successfully

==================================================
MODEL EVALUATION RESULTS
==================================================
Accuracy:  0.6234
Precision: 0.6145
Recall:    0.6321
F1-Score:  0.6231

Top 10 Important Features:
 1. RSI                   : 0.1245
 2. MACD_diff             : 0.0987
 3. SMA_50                : 0.0876
...
```

## ⚠️ Disclaimer

**This is a prediction tool, not financial advice.**
- Stock market predictions are never 100% accurate
- Past performance does not guarantee future results
- Use this tool for educational purposes and research
- Always consult financial advisors before making investment decisions
- Do your own due diligence

## 🛠️ Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn, tensorflow
- yfinance
- ta (Technical Analysis)
- matplotlib, seaborn, plotly

See `requirements.txt` for all dependencies.

## 📚 Upcoming Features

- [ ] LSTM/RNN deep learning models
- [ ] Real-time alerts and notifications
- [ ] Web dashboard with Streamlit
- [ ] Multiple stock portfolio analysis
- [ ] Sentiment analysis integration
- [ ] Options trading prediction

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## 📄 License

MIT License - see LICENSE file for details

## 📞 Support

For questions or issues, please create an issue on GitHub.

---

**Happy Predicting! 📈**
