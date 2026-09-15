"""
Machine Learning models for stock price prediction
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle


class StockPricePredictor:
    """ML model for predicting stock price movements"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize predictor with specified model
        
        Args:
            model_type (str): Type of model ('random_forest', 'logistic_regression', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42, probability=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"✓ Model initialized: {self.model_type}")
    
    def prepare_data(self, data, feature_columns, test_size=0.2, random_state=42):
        """
        Prepare and split data for training
        
        Args:
            data (pd.DataFrame): Processed data with features and target
            feature_columns (list): List of feature column names
            test_size (float): Proportion of test data
            random_state (int): Random seed
        
        Returns:
            dict: Training results
        """
        self.feature_names = feature_columns
        
        X = data[feature_columns].values
        y = data['Target'].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Data prepared successfully")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        print(f"  Features: {len(feature_columns)}")
        
        return {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'n_features': len(feature_columns)
        }
    
    def train(self):
        """Train the model"""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print(f"Training {self.model_type} model...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model trained successfully")
    
    def evaluate(self):
        """Evaluate model performance"""
        if self.model is None or self.X_test is None:
            raise ValueError("Model not trained or data not prepared.")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")
        print("="*50 + "\n")
        
        return results
    
    def predict(self, features):
        """
        Predict for new data
        
        Args:
            features (np.array): Feature array
        
        Returns:
            dict: Prediction result
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            return {
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'probability': probabilities[prediction]
            }
        else:
            return {
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'probability': None
            }
    
    def feature_importance(self):
        """Get feature importance (for tree-based models)"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            print("\nTop 10 Important Features:")
            for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
                print(f"{i:2d}. {feature:20s} : {importance:.4f}")
            
            return dict(sorted_importance)
        else:
            print("Feature importance not available for this model type")
            return None
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            print("✗ No model to save")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return model_data
