"""
Data Collection Module for COVID-19 Forecasting
Collects data from multiple sources: COVID-19, Weather, and Twitter
"""

import pandas as pd
import numpy as np
import requests
import tweepy
import json
from datetime import datetime, timedelta
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.setup_twitter_api()
        self.setup_sentiment_analyzer()
    
    def setup_twitter_api(self):
        """Setup Twitter API client"""
        try:
            self.twitter_client = tweepy.Client(
                bearer_token=self.config.TWITTER_BEARER_TOKEN,
                wait_on_rate_limit=True
            )
        except Exception as e:
            print(f"Error setting up Twitter API: {e}")
            self.twitter_client = None
    
    def setup_sentiment_analyzer(self):
        """Setup NLTK sentiment analyzer"""
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Error setting up sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def load_covid_data(self):
        """Load and process COVID-19 data for India"""
        print("Loading COVID-19 data...")
        try:
            # Load the data
            df = pd.read_csv(self.config.COVID_DATA_PATH)
            
            # Filter for India
            india_data = df[df['iso_code'] == self.config.INDIA_ISO_CODE].copy()
            
            # Convert date to datetime
            india_data['date'] = pd.to_datetime(india_data['date'])
            
            # Sort by date
            india_data = india_data.sort_values('date').reset_index(drop=True)
            
            # Select relevant features
            feature_cols = ['date'] + [col for col in self.config.COVID_FEATURES if col in india_data.columns]
            india_data = india_data[feature_cols]
            
            # Fill missing values
            for col in india_data.columns:
                if col != 'date':
                    india_data[col] = india_data[col].fillna(method='ffill').fillna(0)
            
            print(f"Loaded COVID-19 data: {len(india_data)} records")
            return india_data
            
        except Exception as e:
            print(f"Error loading COVID-19 data: {e}")
            return None
    
    def collect_weather_data(self, start_date, end_date):
        """Collect weather data for major Indian cities"""
        print("Collecting weather data...")
        weather_data = []
        
        try:
            for city in self.config.WEATHER_CITIES:
                print(f"Fetching weather data for {city}...")
                
                # Create date range
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                
                for date in date_range:
                    try:
                        # OpenWeatherMap historical data API call
                        timestamp = int(date.timestamp())
                        url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
                        params = {
                            'lat': self.get_city_coordinates(city)[0],
                            'lon': self.get_city_coordinates(city)[1],
                            'dt': timestamp,
                            'appid': self.config.WEATHER_API_KEY,
                            'units': 'metric'
                        }
                        
                        response = requests.get(url, params=params)
                        
                        if response.status_code == 200:
                            data = response.json()
                            current = data['current']
                            
                            weather_data.append({
                                'date': date,
                                'city': city,
                                'temperature': current.get('temp', 0),
                                'humidity': current.get('humidity', 0),
                                'pressure': current.get('pressure', 0),
                                'wind_speed': current.get('wind_speed', 0),
                                'uvi': current.get('uvi', 0),
                                'visibility': current.get('visibility', 10000) / 1000
                            })
                        
                    except Exception as e:
                        print(f"Error fetching weather for {city} on {date}: {e}")
                        continue
            
            # Convert to DataFrame and aggregate by date
            weather_df = pd.DataFrame(weather_data)
            if not weather_df.empty:
                weather_agg = weather_df.groupby('date').agg({
                    'temperature': 'mean',
                    'humidity': 'mean',
                    'pressure': 'mean',
                    'wind_speed': 'mean',
                    'uvi': 'mean',
                    'visibility': 'mean'
                }).reset_index()
                
                print(f"Collected weather data: {len(weather_agg)} records")
                return weather_agg
            
        except Exception as e:
            print(f"Error collecting weather data: {e}")
            
        return pd.DataFrame()  # Return empty DataFrame if failed
    
    def get_city_coordinates(self, city):
        """Get coordinates for major Indian cities"""
        coordinates = {
            "New Delhi": (28.6139, 77.2090),
            "Mumbai": (19.0760, 72.8777),
            "Bangalore": (12.9716, 77.5946),
            "Chennai": (13.0827, 80.2707),
            "Kolkata": (22.5726, 88.3639)
        }
        return coordinates.get(city, (28.6139, 77.2090))  # Default to Delhi
    
    def collect_twitter_data(self, start_date, end_date):
        """Collect Twitter data related to COVID-19 in India"""
        print("Collecting Twitter data...")
        
        if not self.twitter_client:
            print("Twitter API not available, creating dummy data...")
            return self.create_dummy_twitter_data(start_date, end_date)
        
        twitter_data = []
        
        try:
            # Create query for COVID-related tweets in India
            query = "(" + " OR ".join(self.config.COVID_KEYWORDS) + f") lang:{self.config.TWITTER_LANG} place:India"
            
            # Collect tweets for date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            for date in date_range:
                try:
                    start_time = date.strftime('%Y-%m-%dT00:00:00Z')
                    end_time = (date + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
                    
                    tweets = tweepy.Paginator(
                        self.twitter_client.search_recent_tweets,
                        query=query,
                        start_time=start_time,
                        end_time=end_time,
                        max_results=self.config.TWEET_COUNT,
                        tweet_fields=['created_at', 'public_metrics']
                    ).flatten(limit=self.config.TWEET_COUNT)
                    
                    daily_tweets = list(tweets)
                    
                    if daily_tweets:
                        # Analyze sentiment
                        sentiments = [self.analyze_sentiment(tweet.text) for tweet in daily_tweets]
                        
                        twitter_data.append({
                            'date': date,
                            'tweet_count': len(daily_tweets),
                            'avg_sentiment': np.mean([s['compound'] for s in sentiments]),
                            'positive_ratio': np.mean([1 if s['compound'] > 0.1 else 0 for s in sentiments]),
                            'negative_ratio': np.mean([1 if s['compound'] < -0.1 else 0 for s in sentiments]),
                            'neutral_ratio': np.mean([1 if abs(s['compound']) <= 0.1 else 0 for s in sentiments]),
                            'avg_retweets': np.mean([tweet.public_metrics['retweet_count'] for tweet in daily_tweets]),
                            'avg_likes': np.mean([tweet.public_metrics['like_count'] for tweet in daily_tweets])
                        })
                    
                except Exception as e:
                    print(f"Error collecting tweets for {date}: {e}")
                    continue
            
            twitter_df = pd.DataFrame(twitter_data)
            print(f"Collected Twitter data: {len(twitter_df)} records")
            return twitter_df
            
        except Exception as e:
            print(f"Error collecting Twitter data: {e}")
            return self.create_dummy_twitter_data(start_date, end_date)
    
    def create_dummy_twitter_data(self, start_date, end_date):
        """Create dummy Twitter data when API is not available"""
        print("Creating dummy Twitter data...")
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        np.random.seed(42)  # For reproducible results
        dummy_data = []
        
        for date in date_range:
            dummy_data.append({
                'date': date,
                'tweet_count': np.random.randint(50, 200),
                'avg_sentiment': np.random.uniform(-0.5, 0.5),
                'positive_ratio': np.random.uniform(0.2, 0.5),
                'negative_ratio': np.random.uniform(0.2, 0.5),
                'neutral_ratio': np.random.uniform(0.2, 0.4),
                'avg_retweets': np.random.uniform(10, 100),
                'avg_likes': np.random.uniform(20, 200)
            })
        
        return pd.DataFrame(dummy_data)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if self.sentiment_analyzer:
            return self.sentiment_analyzer.polarity_scores(text)
        else:
            # Fallback to TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return {'compound': polarity, 'pos': max(0, polarity), 'neg': abs(min(0, polarity)), 'neu': 0}
    
    def collect_all_data(self):
        """Collect all data and merge into single dataset"""
        print("Starting data collection process...")
        
        # Load COVID data
        covid_data = self.load_covid_data()
        if covid_data is None:
            return None
        
        # Get date range from COVID data
        start_date = covid_data['date'].min()
        end_date = covid_data['date'].max()
        
        print(f"Date range: {start_date} to {end_date}")
        
        # Collect weather data
        weather_data = self.collect_weather_data(start_date, end_date)
        
        # Collect Twitter data
        twitter_data = self.collect_twitter_data(start_date, end_date)
        
        # Merge all data
        merged_data = covid_data.copy()
        
        if not weather_data.empty:
            merged_data = pd.merge(merged_data, weather_data, on='date', how='left')
        
        if not twitter_data.empty:
            merged_data = pd.merge(merged_data, twitter_data, on='date', how='left')
        
        # Fill any remaining NaN values
        merged_data = merged_data.fillna(method='ffill').fillna(0)
        
        print(f"Final merged dataset: {merged_data.shape}")
        
        # Save the merged data
        import os
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        merged_data.to_csv(self.config.PROCESSED_DATA_PATH, index=False)
        print(f"Data saved to {self.config.PROCESSED_DATA_PATH}")
        
        return merged_data

if __name__ == "__main__":
    from config import Config
    
    config = Config()
    collector = DataCollector(config)
    
    # Collect all data
    data = collector.collect_all_data()
    
    if data is not None:
        print("\nData collection completed successfully!")
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(data.head())
    else:
        print("Data collection failed!")