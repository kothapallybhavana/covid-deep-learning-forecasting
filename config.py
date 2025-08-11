"""
Configuration file for COVID-19 Forecasting Pipeline
"""
import os
from dataclasses import dataclass

@dataclass
class Config:
    # API Keys
    TWITTER_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAI3L2QEAAAAAo3LAhCkOPv2ouGm8qKV%2FA9WYM04%3DUt0sPDNtAOPdc4B11jeBf42YyVych65f5q2pNcSZEu100qk1ke'
    NEWS_API_KEY = '1b6125d0311a4351994f2cb04c2ff887'
    WEATHER_API_KEY = 'bd5e378503939ddaee76f12ad7a97608'
    
    # File paths
    COVID_DATA_PATH = 'data/owid-covid-data.csv'
    WEATHER_DATA_PATH = 'data/weather_data.csv'
    TWITTER_DATA_PATH = 'data/twitter_data.csv'
    PROCESSED_DATA_PATH = 'data/processed_data.csv'
    
    # Model parameters
    SEQUENCE_LENGTH = 30  # Days to look back
    FORECAST_HORIZON = 7  # Days to predict ahead
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 10
    
    # Data collection parameters
    INDIA_LOCATION = "India"
    INDIA_ISO_CODE = "IND"
    WEATHER_CITIES = ["New Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
    
    # Twitter search parameters
    COVID_KEYWORDS = ["covid", "corona", "virus", "pandemic", "lockdown", 
                     "vaccine", "cases", "death", "hospital", "oxygen"]
    TWITTER_LANG = "en"
    TWEET_COUNT = 100
    
    # Feature columns from COVID data
    COVID_FEATURES = [
        'total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths',
        'new_deaths', 'new_deaths_smoothed', 'reproduction_rate',
        'icu_patients', 'hosp_patients', 'total_tests', 'new_tests',
        'positive_rate', 'total_vaccinations', 'people_vaccinated',
        'stringency_index'
    ]
    
    # Model save paths
    LSTM_MODEL_PATH = 'models/lstm_model.h5'
    BPNN_MODEL_PATH = 'models/bpnn_model.h5'
    ELMAN_MODEL_PATH = 'models/elman_model.h5'
    ANFIS_MODEL_PATH = 'models/anfis_model.pkl'
    BERT_MODEL_PATH = 'models/bert_model'
    ENSEMBLE_MODEL_PATH = 'models/ensemble_model.pkl'
    
    # Directories
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    PLOTS_DIR = 'plots'