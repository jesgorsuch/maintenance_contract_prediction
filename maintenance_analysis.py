import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import shap
from datetime import datetime

class MaintenanceAnalyzer:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """
        Load the maintenance and financial data from a CSV file
        Expected columns:
        - client_id: unique identifier for each client
        - contract_id: identifier for service agreement
        - contract_start_date: when the service agreement began
        - contract_end_date: when the service agreement ends
        - instrument_id: unique identifier for each instrument
        - instrument_type: type of scientific instrument
        - instrument_age: age in years
        - contract_revenue: revenue from service agreement
        - phone_consultations: number of phone calls for expert consultation
        - onsite_visits: number of service calls requiring onsite visit
        - days_onsite: total days spent on site
        - parts_cost: cost of parts used in repairs
        - repair_hours: technician hours for covered repairs
        - pm_visits_scheduled: number of scheduled preventative maintenance visits
        - pm_visits_completed: number of completed preventative maintenance visits
        - client_errors: number of issues caused by client misuse
        - total_repair_cost: total cost of repairs (target variable)
        """
        self.data = pd.read_csv(file_path)
        print("Data shape:", self.data.shape)
        print("\nData overview:")
        print(self.data.head())
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
    def calculate_derived_metrics(self):
        """
        Calculate additional metrics from the raw data
        """
        # Contract duration in days
        self.data['contract_start_date'] = pd.to_datetime(self.data['contract_start_date'])
        self.data['contract_end_date'] = pd.to_datetime(self.data['contract_end_date'])
        self.data['contract_duration'] = (self.data['contract_end_date'] - self.data['contract_start_date']).dt.days
        
        # PM compliance rate
        self.data['pm_compliance_rate'] = self.data['pm_visits_completed'] / self.data['pm_visits_scheduled']
        
        # Cost metrics
        self.data['cost_per_day_onsite'] = self.data['total_repair_cost'] / self.data['days_onsite'].replace(0, 1)
        self.data['revenue_per_visit'] = self.data['contract_revenue'] / (self.data['onsite_visits'] + 1)
        
        # Service intensity
        self.data['service_intensity'] = (self.data['phone_consultations'] + 
                                        self.data['onsite_visits'] * 3 +  # Weighting onsite visits more heavily
                                        self.data['client_errors'] * 2)   # Weighting client errors
        
        # Profit margin
        self.data['estimated_profit'] = (self.data['contract_revenue'] - 
                                       self.data['parts_cost'] - 
                                       (self.data['repair_hours'] * 150) -  # Assuming $150/hour labor cost
                                       (self.data['days_onsite'] * 500))    # Assuming $500/day overhead
        
    def preprocess_data(self):
        """
        Preprocess the data including:
        - Handling missing values
        - Encoding categorical variables
        - Feature scaling
        - Calculate derived metrics
        """
        # Calculate derived metrics first
        self.calculate_derived_metrics()

        # Encode categorical variables
        categorical_columns = ['client_id', 'contract_id', 'instrument_id', 'instrument_type']
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col].fillna('Unknown'))

        # Handle missing numerical values
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_columns] = self.data[numerical_columns].fillna(self.data[numerical_columns].mean())

    def prepare_features(self, target_column='total_repair_cost'):
        """
        Prepare features for model training
        """
        # Remove target column and non-feature columns from features
        exclude_columns = [target_column, 'contract_start_date', 'contract_end_date']
        feature_columns = [col for col in self.data.columns if col not in exclude_columns]
        
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def train_model(self, model_type='random_forest'):
        """
        Train the predictive model
        """
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            )
            
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate_model(self):
        """
        Evaluate the model's performance and generate insights
        """
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        print("\nModel Performance:")
        print(f"Root Mean Squared Error: ${rmse:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance analysis
        self.plot_feature_importance()
        self.plot_shap_values()
        self.generate_financial_analysis()
        
    def plot_feature_importance(self):
        """
        Create and save feature importance visualization
        """
        feature_importance = pd.DataFrame({
            'feature': [col for col in self.data.columns if col not in ['total_repair_cost', 'contract_start_date', 'contract_end_date']],
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
    def plot_shap_values(self):
        """
        Create and save SHAP value analysis
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title('SHAP Value Analysis')
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
        
    def generate_financial_analysis(self):
        """
        Generate financial insights and metrics
        """
        financial_metrics = {
            'avg_contract_revenue': self.data['contract_revenue'].mean(),
            'avg_repair_cost': self.data['total_repair_cost'].mean(),
            'avg_profit_margin': (self.data['estimated_profit'] / self.data['contract_revenue']).mean() * 100,
            'avg_pm_compliance': self.data['pm_compliance_rate'].mean() * 100,
            'total_service_calls': self.data['onsite_visits'].sum(),
            'avg_parts_cost': self.data['parts_cost'].mean()
        }
        
        # Save financial metrics
        with open('financial_analysis.txt', 'w') as f:
            f.write("Financial Analysis Summary\n")
            f.write("=========================\n\n")
            f.write(f"Average Contract Revenue: ${financial_metrics['avg_contract_revenue']:,.2f}\n")
            f.write(f"Average Repair Cost: ${financial_metrics['avg_repair_cost']:,.2f}\n")
            f.write(f"Average Profit Margin: {financial_metrics['avg_profit_margin']:.1f}%\n")
            f.write(f"Average PM Compliance Rate: {financial_metrics['avg_pm_compliance']:.1f}%\n")
            f.write(f"Total Service Calls: {financial_metrics['total_service_calls']:,}\n")
            f.write(f"Average Parts Cost: ${financial_metrics['avg_parts_cost']:,.2f}\n")
        
    def predict_maintenance(self, new_data):
        """
        Make predictions on new data
        """
        # Preprocess new data similarly to training data
        for col in self.label_encoders:
            if col in new_data:
                new_data[col] = self.label_encoders[col].transform(new_data[col])
        
        # Scale the features
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make prediction
        prediction = self.model.predict(new_data_scaled)
        return prediction

def main():
    # Initialize the analyzer
    analyzer = MaintenanceAnalyzer()
    
    # Load and prepare data
    analyzer.load_data('maintenance_data.csv')
    analyzer.preprocess_data()
    analyzer.prepare_features()
    
    # Train and evaluate model
    analyzer.train_model(model_type='random_forest')
    analyzer.evaluate_model()

if __name__ == "__main__":
    main() 