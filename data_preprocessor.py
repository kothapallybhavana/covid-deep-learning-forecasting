"""
Data Preprocessing Module for COVID-19 Forecasting
Handles feature engineering, normalization, and sequence creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.feature_selector = None
        self.selected_features = None
        
    def load_data(self, file_path=None):
        """Load processed data"""
        if file_path is None:
            file_path = self.config.PROCESSED_DATA_PATH
            
        try:
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'])
            print(f"Loaded data: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_features(self, data):
        """Create additional features from existing data"""
        print("Creating additional features...")
        
        # Make a copy
        df = data.copy()
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # COVID-related derived features
        if 'total_cases' in df.columns:
            df['cases_growth_rate'] = df['total_cases'].pct_change().fillna(0)
            df['cases_7day_avg'] = df['total_cases'].rolling(window=7).mean().fillna(df['total_cases'])
            df['cases_14day_avg'] = df['total_cases'].rolling(window=14).mean().fillna(df['total_cases'])
        
        if 'new_cases' in df.columns:
            df['new_cases_7day_avg'] = df['new_cases'].rolling(window=7).mean().fillna(df['new_cases'])
            df['new_cases_trend'] = df['new_cases'].diff().fillna(0)
        
        if 'total_deaths' in df.columns:
            df['death_rate'] = np.where(df['total_cases'] > 0, 
                                       df['total_deaths'] / df['total_cases'], 0)
        
        # Weather-related derived features
        weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for col in weather_cols:
            if col in df.columns:
                df[f'{col}_7day_avg'] = df[col].rolling(window=7).mean().fillna(df[col])
                df[f'{col}_change'] = df[col].diff().fillna(0)
        
        # Twitter sentiment features
        if 'avg_sentiment' in df.columns:
            df['sentiment_7day_avg'] = df['avg_sentiment'].rolling(window=7).mean().fillna(df['avg_sentiment'])
            df['sentiment_volatility'] = df['avg_sentiment'].rolling(window=7).std().fillna(0)
        
        # Lag features
        lag_features = ['new_cases', 'total_cases', 'avg_sentiment', 'temperature']
        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 3, 7, 14]:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag).fillna(0)
        
        print(f"Created features. New shape: {df.shape}")
        return df
    
    def handle_outliers(self, data, method='iqr', threshold=3):
        """Handle outliers in the data"""
        print("Handling outliers...")
        
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'date':
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(df[col]))
                    df = df[z_scores < threshold]
        
        print(f"After outlier handling: {df.shape}")
        return df
    
    def normalize_features(self, data, method='minmax'):
        """Normalize/standardize features"""
        print("Normalizing features...")
        
        df = data.copy()
        
        # Separate date column
        date_col = df['date'] if 'date' in df.columns else None
        
        # Get numeric columns (excluding date)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            print("Unknown normalization method. Using MinMax.")
            scaler = MinMaxScaler()
        
        # Fit and transform
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Store scaler for later use
        self.scalers['main_scaler'] = scaler
        
        print("Features normalized")
        return df
    
    def select_features(self, data, target_col='new_cases', k=20):
        """Select best features using statistical methods"""
        print("Selecting best features...")
        
        df = data.copy()
        
        # Separate features and target
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found. Using all features.")
            return df
        
        # Get feature columns (excluding date and target)
        feature_cols = [col for col in df.columns if col not in ['date', target_col]]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Select k best features
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
        
        # Create new dataframe with selected features
        result_df = df[['date', target_col] + selected_features].copy()
        
        # Store for later use
        self.feature_selector = selector
        self.selected_features = selected_features
        
        print(f"Selected {len(selected_features)} features: {selected_features}")
        return result_df
    
    def create_sequences(self, data, target_col='new_cases'):
        """Create sequences for time series prediction"""
        print("Creating sequences for time series modeling...")
        
        df = data.copy().sort_values('date')
        
        # Remove date column for sequence creation
        feature_cols = [col for col in df.columns if col != 'date']
        values = df[feature_cols].values.astype('float32')
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.config.SEQUENCE_LENGTH, len(values) - self.config.FORECAST_HORIZON + 1):
            # Input sequence
            X.append(values[i-self.config.SEQUENCE_LENGTH:i])
            
            # Target (next FORECAST_HORIZON values of target column)
            target_idx = feature_cols.index(target_col)
            y.append(values[i:i+self.config.FORECAST_HORIZON, target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y, feature_cols
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        print("Splitting data...")
        
        n_samples = X.shape[0]
        train_size = int(n_samples * self.config.TRAIN_RATIO)
        val_size = int(n_samples * self.config.VAL_RATIO)
        
        # Chronological split (important for time series)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def plot_data_analysis(self, data):
        """Plot data analysis and statistics"""
        print("Creating data analysis plots...")
        
        import os
        os.makedirs(self.config.PLOTS_DIR, exist_ok=True)
        
        # Time series plot
        plt.figure(figsize=(15, 10))
        
        # COVID cases over time
        plt.subplot(2, 2, 1)
        if 'new_cases' in data.columns:
            plt.plot(data['date'], data['new_cases'])
            plt.title('New COVID-19 Cases Over Time')
            plt.xlabel('Date')
            plt.ylabel('New Cases')
            plt.xticks(rotation=45)
        
        # Weather features
        plt.subplot(2, 2, 2)
        weather_cols = ['temperature', 'humidity']
        for col in weather_cols:
            if col in data.columns:
                plt.plot(data['date'], data[col], label=col)
        plt.title('Weather Features Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Twitter sentiment
        plt.subplot(2, 2, 3)
        if 'avg_sentiment' in data.columns:
            plt.plot(data['date'], data['avg_sentiment'])
            plt.title('Average Twitter Sentiment Over Time')
            plt.xlabel('Date')
            plt.ylabel('Sentiment')
            plt.xticks(rotation=45)
        
        # Correlation heatmap
        plt.subplot(2, 2, 4)
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]  # Top 10 features
        correlation_matrix = data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig(f'{self.config.PLOTS_DIR}/data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot if available
        if self.feature_selector is not None:
            plt.figure(figsize=(10, 8))
            scores = self.feature_selector.scores_
            features = self.selected_features
            
            # Sort by importance
            importance_data = list(zip(features, scores))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            features_sorted = [x[0] for x in importance_data]
            scores_sorted = [x[1] for x in importance_data]
            
            plt.barh(range(len(features_sorted)), scores_sorted)
            plt.yticks(range(len(features_sorted)), features_sorted)
            plt.xlabel('Feature Importance Score')
            plt.title('Top Selected Features')
            plt.tight_layout()
            plt.savefig(f'{self.config.PLOTS_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def preprocess_pipeline(self, data_path=None, target_col='new_cases'):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Load data
        data = self.load_data(data_path)
        if data is None:
            return None
        
        # Create additional features
        data = self.create_features(data)
        
        # Handle outliers
        data = self.handle_outliers(data)
        
        # Plot initial analysis
        self.plot_data_analysis(data)
        
        # Select best features
        data = self.select_features(data, target_col)
        
        # Normalize features
        data = self.normalize_features(data)
        
        # Create sequences
        X, y, feature_names = self.create_sequences(data, target_col)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Save preprocessed data
        import os
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        
        np.save(f'{self.config.DATA_DIR}/X_train.npy', X_train)
        np.save(f'{self.config.DATA_DIR}/X_val.npy', X_val)
        np.save(f'{self.config.DATA_DIR}/X_test.npy', X_test)
        np.save(f'{self.config.DATA_DIR}/y_train.npy', y_train)
        np.save(f'{self.config.DATA_DIR}/y_val.npy', y_val)
        np.save(f'{self.config.DATA_DIR}/y_test.npy', y_test)
        
        # Save feature names and other metadata
        import pickle
        metadata = {
            'feature_names': feature_names,
            'selected_features': self.selected_features,
            'scalers': self.scalers,
            'sequence_length': self.config.SEQUENCE_LENGTH,
            'forecast_horizon': self.config.FORECAST_HORIZON
        }
        
        with open(f'{self.config.DATA_DIR}/preprocessing_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("Preprocessing pipeline completed successfully!")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': feature_names,
            'metadata': metadata
        }

if __name__ == "__main__":
    from config import Config
    
    config = Config()
    preprocessor = DataPreprocessor(config)
    
    # Run preprocessing pipeline
    result = preprocessor.preprocess_pipeline()
    
    if result is not None:
        print("\nPreprocessing completed successfully!")
        print(f"Training data shape: {result['X_train'].shape}")
        print(f"Feature names: {result['feature_names']}")
    else:
        print("Preprocessing failed!")