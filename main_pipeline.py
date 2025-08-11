"""
Main Pipeline for COVID-19 Forecasting Project
Orchestrates the complete workflow: data collection, preprocessing, training, and evaluation
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import Config
from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from models import EnsembleModel, evaluate_model
from visualization import CovidVisualizer

class CovidForecastingPipeline:
    def __init__(self):
        self.config = Config()
        self.setup_directories()
        
        # Initialize components
        self.data_collector = DataCollector(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.visualizer = CovidVisualizer(self.config)
        
        # Results storage
        self.results = {}
        self.data = None
        self.preprocessed_data = None
        self.trained_models = None
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.DATA_DIR,
            self.config.MODELS_DIR,
            self.config.RESULTS_DIR,
            self.config.PLOTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("✓ Directories created successfully")
    
    def run_data_collection(self):
        """Step 1: Collect data from all sources"""
        print("\n" + "="*60)
        print("STEP 1: DATA COLLECTION")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Check if processed data already exists
            if os.path.exists(self.config.PROCESSED_DATA_PATH):
                print("Processed data already exists. Loading...")
                self.data = pd.read_csv(self.config.PROCESSED_DATA_PATH)
                self.data['date'] = pd.to_datetime(self.data['date'])
                print(f"✓ Loaded existing data: {self.data.shape}")
            else:
                print("Collecting data from multiple sources...")
                self.data = self.data_collector.collect_all_data()
                
                if self.data is None:
                    raise Exception("Data collection failed")
                
                print(f"✓ Data collection completed: {self.data.shape}")
            
            # Store data info for reporting
            self.results['data_info'] = {
                'total_records': len(self.data),
                'date_range': f"{self.data['date'].min()} to {self.data['date'].max()}",
                'num_features': len(self.data.columns) - 1,  # Exclude date
                'train_period': f"{self.config.TRAIN_RATIO*100:.0f}% of data"
            }
            
            duration = time.time() - start_time
            print(f"✓ Data collection completed in {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Data collection failed: {e}")
            return False
    
    def run_preprocessing(self):
        """Step 2: Preprocess and prepare data for modeling"""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Check if preprocessed data already exists
            if (os.path.exists(f'{self.config.DATA_DIR}/X_train.npy') and 
                os.path.exists(f'{self.config.DATA_DIR}/y_train.npy')):
                
                print("Preprocessed data already exists. Loading...")
                self.preprocessed_data = {
                    'X_train': np.load(f'{self.config.DATA_DIR}/X_train.npy'),
                    'X_val': np.load(f'{self.config.DATA_DIR}/X_val.npy'),
                    'X_test': np.load(f'{self.config.DATA_DIR}/X_test.npy'),
                    'y_train': np.load(f'{self.config.DATA_DIR}/y_train.npy'),
                    'y_val': np.load(f'{self.config.DATA_DIR}/y_val.npy'),
                    'y_test': np.load(f'{self.config.DATA_DIR}/y_test.npy')
                }
                
                # Load metadata
                import pickle
                with open(f'{self.config.DATA_DIR}/preprocessing_metadata.pkl', 'rb') as f:
                    metadata = pickle.load(f)
                
                self.preprocessed_data.update(metadata)
                print(f"✓ Loaded preprocessed data: {self.preprocessed_data['X_train'].shape}")
                
            else:
                print("Running preprocessing pipeline...")
                self.preprocessed_data = self.preprocessor.preprocess_pipeline()
                
                if self.preprocessed_data is None:
                    raise Exception("Preprocessing failed")
                
                print(f"✓ Preprocessing completed: {self.preprocessed_data['X_train'].shape}")
            
            duration = time.time() - start_time
            print(f"✓ Preprocessing completed in {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Preprocessing failed: {e}")
            return False
    
    def run_model_training(self):
        """Step 3: Train all models"""
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Extract data
            X_train = self.preprocessed_data['X_train']
            X_val = self.preprocessed_data['X_val']
            X_test = self.preprocessed_data['X_test']
            y_train = self.preprocessed_data['y_train']
            y_val = self.preprocessed_data['y_val']
            y_test = self.preprocessed_data['y_test']
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Validation data shape: {X_val.shape}")
            print(f"Test data shape: {X_test.shape}")
            
            # Initialize ensemble model
            self.trained_models = EnsembleModel(self.config, X_train.shape)
            
            # Train all models
            print("Training ensemble of models...")
            self.trained_models.train_all(X_train, y_train, X_val, y_val)
            
            # Save models
            print("Saving trained models...")
            self.trained_models.save_all()
            
            duration = time.time() - start_time
            print(f"✓ Model training completed in {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Model training failed: {e}")
            return False
    
    def run_evaluation(self):
        """Step 4: Evaluate models and generate results"""
        print("\n" + "="*60)
        print("STEP 4: MODEL EVALUATION")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Extract test data
            X_test = self.preprocessed_data['X_test']
            y_test = self.preprocessed_data['y_test']
            
            # Make predictions
            print("Generating predictions...")
            ensemble_pred, individual_preds = self.trained_models.predict(X_test)
            
            # Evaluate models
            print("Evaluating model performance...")
            results = []
            
            # Evaluate ensemble
            ensemble_result = evaluate_model(y_test, ensemble_pred, "Ensemble")
            results.append(ensemble_result)
            
            # Evaluate individual models
            for name, pred in individual_preds.items():
                result = evaluate_model(y_test, pred, name.upper())
                results.append(result)
            
            # Create results DataFrame
            self.results['model_results'] = pd.DataFrame(results)
            
            # Save results
            self.results['model_results'].to_csv(
                f'{self.config.RESULTS_DIR}/model_results.csv', index=False
            )
            
            # Store predictions for visualization
            self.results['predictions'] = {
                'ensemble': ensemble_pred,
                'individual': individual_preds,
                'y_true': y_test,
                'y_pred_ensemble': ensemble_pred
            }
            
            # Print summary
            print("\n" + "="*50)
            print("MODEL PERFORMANCE SUMMARY")
            print("="*50)
            print(self.results['model_results'].round(4))
            
            # Find best model
            best_model_idx = self.results['model_results']['rmse'].idxmin()
            best_model = self.results['model_results'].iloc[best_model_idx]
            
            print(f"\n🏆 Best Model: {best_model['model']}")
            print(f"   RMSE: {best_model['rmse']:.4f}")
            print(f"   MAE: {best_model['mae']:.4f}")
            print(f"   R²: {best_model['r2']:.4f}")
            print(f"   MAPE: {best_model['mape']:.2f}%")
            
            duration = time.time() - start_time
            print(f"\n✓ Evaluation completed in {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Model evaluation failed: {e}")
            return False
    
    def run_visualization(self):
        """Step 5: Generate visualizations and reports"""
        print("\n" + "="*60)
        print("STEP 5: VISUALIZATION & REPORTING")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Get training histories (if available)
            training_histories = {}
            for name, model in self.trained_models.models.items():
                if hasattr(model, 'history') and model.history is not None:
                    training_histories[name] = model.history
            
            # Create visualizations
            print("Creating training history plots...")
            if training_histories:
                self.visualizer.plot_training_history(
                    training_histories, 
                    list(training_histories.keys())
                )
            
            print("Creating prediction comparison plots...")
            self.visualizer.plot_predictions_comparison(
                self.results['predictions']['y_true'],
                self.results['predictions']['individual']
            )
            
            print("Creating model performance plots...")
            self.visualizer.plot_model_performance(self.results['model_results'])
            
            print("Creating residuals analysis...")
            for name, pred in self.results['predictions']['individual'].items():
                self.visualizer.plot_residuals_analysis(
                    self.results['predictions']['y_true'], 
                    pred, 
                    name
                )
            
            print("Creating forecast horizon plots...")
            self.visualizer.plot_forecast_horizon(
                self.results['predictions']['y_true'],
                self.results['predictions']['y_pred_ensemble'],
                'Ensemble'
            )
            
            print("Creating feature correlation plot...")
            if self.data is not None:
                self.visualizer.plot_feature_correlation(self.data)
            
            print("Creating time series decomposition...")
            if self.data is not None:
                self.visualizer.plot_time_series_decomposition(self.data)
            
            print("Creating interactive dashboard...")
            if self.data is not None:
                self.visualizer.create_interactive_dashboard(
                    self.data,
                    self.results['predictions']['individual'],
                    self.results['model_results']
                )
            
            print("Creating summary report...")
            self.visualizer.create_summary_report(
                self.results['model_results'],
                self.results['data_info']
            )
            
            duration = time.time() - start_time
            print(f"✓ Visualization completed in {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("🚀 STARTING COVID-19 FORECASTING PIPELINE")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        pipeline_start = time.time()
        
        steps = [
            ("Data Collection", self.run_data_collection),
            ("Data Preprocessing", self.run_preprocessing),
            ("Model Training", self.run_model_training),
            ("Model Evaluation", self.run_evaluation),
            ("Visualization & Reporting", self.run_visualization)
        ]
        
        completed_steps = 0
        
        for step_name, step_function in steps:
            try:
                success = step_function()
                if success:
                    completed_steps += 1
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"❌ {step_name} failed")
                    break
            except Exception as e:
                print(f"❌ {step_name} failed with error: {e}")
                break
        
        # Pipeline summary
        total_duration = time.time() - pipeline_start
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Completed Steps: {completed_steps}/{len(steps)}")
        print(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if completed_steps == len(steps):
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("\nGenerated Files:")
            print(f"📊 Results: {self.config.RESULTS_DIR}/")
            print(f"🤖 Models: {self.config.MODELS_DIR}/")
            print(f"📈 Plots: {self.config.PLOTS_DIR}/")
            print(f"💾 Data: {self.config.DATA_DIR}/")
            
            print("\nNext Steps:")
            print("1. Review model performance in results/model_results.csv")
            print("2. Examine visualizations in plots/ directory")
            print("3. Open plots/interactive_dashboard.html for interactive analysis")
            print("4. Check plots/summary_report.png for comprehensive overview")
            
        else:
            print("⚠️ PIPELINE INCOMPLETE")
            print("Please check the error messages above and resolve issues.")
        
        print("=" * 80)
        
        return completed_steps == len(steps)
    
    def run_quick_prediction(self, days_ahead=7):
        """Make a quick prediction for the next few days"""
        print(f"\n🔮 QUICK PREDICTION FOR NEXT {days_ahead} DAYS")
        print("=" * 50)
        
        try:
            if self.trained_models is None:
                print("❌ Models not trained yet. Please run the complete pipeline first.")
                return None
            
            # Use the last sequence from test data
            X_test = self.preprocessed_data['X_test']
            last_sequence = X_test[-1:]  # Get last sample
            
            # Make prediction
            prediction, _ = self.trained_models.predict(last_sequence)
            
            # Create prediction summary
            pred_days = min(days_ahead, self.config.FORECAST_HORIZON)
            forecast = prediction[0][:pred_days]
            
            print("Forecast Summary:")
            print("-" * 30)
            for i, pred_cases in enumerate(forecast, 1):
                print(f"Day +{i}: {pred_cases:.0f} cases")
            
            print(f"\nAverage daily cases (next {pred_days} days): {forecast.mean():.0f}")
            print(f"Peak day: Day +{np.argmax(forecast)+1} ({forecast.max():.0f} cases)")
            print(f"Lowest day: Day +{np.argmin(forecast)+1} ({forecast.min():.0f} cases)")
            
            return forecast
            
        except Exception as e:
            print(f"❌ Quick prediction failed: {e}")
            return None

def main():
    """Main function to run the pipeline"""
    pipeline = CovidForecastingPipeline()
    
    # Check command line arguments for specific operations
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "collect":
            pipeline.run_data_collection()
        elif command == "preprocess":
            pipeline.run_data_collection()
            pipeline.run_preprocessing()
        elif command == "train":
            pipeline.run_data_collection()
            pipeline.run_preprocessing()
            pipeline.run_model_training()
        elif command == "evaluate":
            pipeline.run_data_collection()
            pipeline.run_preprocessing()
            pipeline.run_model_training()
            pipeline.run_evaluation()
        elif command == "predict":
            pipeline.run_data_collection()
            pipeline.run_preprocessing()
            pipeline.run_model_training()
            pipeline.run_quick_prediction()
        elif command == "full" or command == "all":
            pipeline.run_complete_pipeline()
        else:
            print("Unknown command. Available commands:")
            print("  collect    - Run data collection only")
            print("  preprocess - Run data collection and preprocessing")
            print("  train      - Run up to model training")
            print("  evaluate   - Run up to evaluation")
            print("  predict    - Run pipeline and make quick prediction")
            print("  full/all   - Run complete pipeline")
    else:
        # Run complete pipeline by default
        pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()