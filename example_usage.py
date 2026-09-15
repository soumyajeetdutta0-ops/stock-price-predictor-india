"""
Complete example of using the Stock Price Predictor
"""
from src.data_fetcher import StockDataFetcher
from src.feature_engineering import FeatureEngineer
from src.model import StockPricePredictor


def main():
    print("\n" + "="*60)
    print("INDIAN STOCK PRICE PREDICTOR - COMPLETE EXAMPLE")
    print("="*60 + "\n")
    
    # ========== STEP 1: FETCH DATA ==========
    print("STEP 1: Fetching stock data...")
    print("-" * 60)
    
    fetcher = StockDataFetcher()
    
    # Change these parameters to predict different stocks
    stock_symbol = 'RELIANCE.NS'  # Try: 'INFY.NS', 'TCS.NS', 'WIPRO.NS'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    data = fetcher.fetch_stock_data(
        symbol=stock_symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    if data is None:
        print("Failed to fetch data. Exiting...")
        return
    
    # ========== STEP 2: FEATURE ENGINEERING ==========
    print("\n" + "="*60)
    print("STEP 2: Creating technical indicators and features...")
    print("-" * 60)
    
    engineer = FeatureEngineer(data)
    
    # Add technical indicators
    engineer.add_technical_indicators()
    
    # Add price features
    engineer.add_price_features()
    
    # Create target variable (predict 1 day ahead)
    processed_data = engineer.create_target_variable(forecast_days=1)
    
    print(f"\nData shape after feature engineering: {processed_data.shape}")
    print(f"Features created: {len(engineer.get_features_for_model())}")
    
    # ========== STEP 3: PREPARE DATA ==========
    print("\n" + "="*60)
    print("STEP 3: Preparing data for model training...")
    print("-" * 60)
    
    feature_columns = engineer.get_features_for_model()
    
    # ========== STEP 4: TRAIN MODEL ==========
    print("\n" + "="*60)
    print("STEP 4: Training the model...")
    print("-" * 60)
    
    # Choose model: 'random_forest', 'logistic_regression', or 'svm'
    predictor = StockPricePredictor(model_type='random_forest')
    
    predictor.prepare_data(
        data=processed_data,
        feature_columns=feature_columns,
        test_size=0.2
    )
    
    predictor.train()
    
    # ========== STEP 5: EVALUATE MODEL ==========
    print("\n" + "="*60)
    print("STEP 5: Evaluating model performance...")
    print("-" * 60)
    
    results = predictor.evaluate()
    
    # ========== STEP 6: FEATURE IMPORTANCE ==========
    print("\n" + "="*60)
    print("STEP 6: Feature Importance Analysis")
    print("-" * 60)
    
    predictor.feature_importance()
    
    # ========== STEP 7: MAKE PREDICTIONS ==========
    print("\n" + "="*60)
    print("STEP 7: Making predictions on new data...")
    print("-" * 60)
    
    # Get the last row of features
    latest_features = processed_data.iloc[-1][feature_columns].values
    
    print(f"\nPredicting for: {stock_symbol}")
    print(f"Latest Close Price: ₹{processed_data.iloc[-1]['Close']:.2f}")
    
    prediction = predictor.predict(latest_features)
    
    print(f"\n🔮 PREDICTION: Stock will go {prediction['prediction']}")
    if prediction['probability']:
        confidence = prediction['probability'] * 100
        print(f"   Confidence: {confidence:.2f}%")
    
    # ========== STEP 8: SAVE MODEL ==========
    print("\n" + "="*60)
    print("STEP 8: Saving trained model...")
    print("-" * 60)
    
    model_filename = f"models/{stock_symbol.replace('.', '_')}_model.pkl"
    predictor.save_model(model_filename)
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Stock Symbol:     {stock_symbol}")
    print(f"Data Period:      {start_date} to {end_date}")
    print(f"Total Records:    {len(processed_data)}")
    print(f"Features Used:    {len(feature_columns)}")
    print(f"Model Type:       Random Forest")
    print(f"Accuracy:         {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Prediction:       {prediction['prediction']}")
    print("="*60 + "\n")
    
    return predictor, engineer, processed_data


if __name__ == "__main__":
    main()
