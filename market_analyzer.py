import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import fred
import json
import os

class MarketDataAnalyzer:
    def __init__(self):
        self.market_data = {}
        self.equipment_benchmarks = {}
        self.economic_indicators = {}
        
    def load_equipment_benchmarks(self, file_path=None):
        """
        Load or fetch equipment market data including:
        - Average prices for new equipment
        - Typical service contract rates
        - Standard maintenance intervals
        - Expected lifespans
        """
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.equipment_benchmarks = json.load(f)
        else:
            # Default benchmark data (can be updated with real market data)
            self.equipment_benchmarks = {
                'mass_spectrometer': {
                    'avg_new_price': 250000,
                    'avg_service_rate_yearly': 25000,
                    'maintenance_interval_months': 6,
                    'typical_lifespan_years': 10,
                    'labor_rate_per_hour': 150
                },
                'gas_chromatograph': {
                    'avg_new_price': 75000,
                    'avg_service_rate_yearly': 8000,
                    'maintenance_interval_months': 4,
                    'typical_lifespan_years': 12,
                    'labor_rate_per_hour': 125
                },
                'auto_system_xl': {
                    'avg_new_price': 120000,
                    'avg_service_rate_yearly': 12000,
                    'maintenance_interval_months': 3,
                    'typical_lifespan_years': 8,
                    'labor_rate_per_hour': 135
                }
            }
    
    def fetch_economic_indicators(self):
        """
        Fetch relevant economic indicators from FRED API
        Requires FRED API key set as environment variable FRED_API_KEY
        """
        try:
            fred_client = fred.Fred(api_key=os.getenv('FRED_API_KEY'))
            
            # Fetch relevant economic indicators
            self.economic_indicators = {
                'inflation': fred_client.get_series('CPIAUCSL')[-12:],  # Last 12 months of CPI
                'ppi': fred_client.get_series('PPIACO')[-12:],  # Producer Price Index
                'healthcare_spending': fred_client.get_series('HLTHSCPD')[-12:],  # Healthcare spending
                'research_development': fred_client.get_series('Y694RC1Q027SBEA')[-4:]  # R&D spending
            }
        except Exception as e:
            print(f"Warning: Could not fetch FRED data: {e}")
            # Provide some default trend data
            self.economic_indicators = {
                'inflation_rate': 0.032,  # 3.2% annual inflation
                'healthcare_growth': 0.045,  # 4.5% healthcare spending growth
                'rd_growth': 0.038  # 3.8% R&D spending growth
            }

    def analyze_market_trends(self, maintenance_data):
        """
        Analyze maintenance data in context of market trends
        """
        market_analysis = {
            'cost_efficiency': {},
            'service_benchmarks': {},
            'market_positioning': {}
        }
        
        # Calculate cost efficiency metrics
        for instrument_type in self.equipment_benchmarks:
            benchmark = self.equipment_benchmarks[instrument_type]
            relevant_data = maintenance_data[
                maintenance_data['instrument_type'] == instrument_type
            ]
            
            if len(relevant_data) > 0:
                avg_service_cost = relevant_data['total_repair_cost'].mean()
                benchmark_cost = benchmark['avg_service_rate_yearly']
                
                market_analysis['cost_efficiency'][instrument_type] = {
                    'actual_vs_benchmark': avg_service_cost / benchmark_cost,
                    'cost_per_value': avg_service_cost / benchmark['avg_new_price'],
                    'lifecycle_position': relevant_data['instrument_age'].mean() / benchmark['typical_lifespan_years']
                }
        
        # Service level benchmarking
        market_analysis['service_benchmarks'] = {
            'labor_rate_competitiveness': {},
            'maintenance_interval_compliance': {},
            'lifecycle_cost_ratio': {}
        }
        
        # Market positioning analysis
        market_analysis['market_positioning'] = {
            'price_positioning': self._analyze_price_positioning(maintenance_data),
            'service_quality_metrics': self._analyze_service_quality(maintenance_data),
            'market_share_indicators': self._analyze_market_share(maintenance_data)
        }
        
        return market_analysis
    
    def _analyze_price_positioning(self, maintenance_data):
        """
        Analyze price positioning relative to market benchmarks
        """
        price_analysis = {}
        
        for instrument_type in self.equipment_benchmarks:
            relevant_data = maintenance_data[
                maintenance_data['instrument_type'] == instrument_type
            ]
            
            if len(relevant_data) > 0:
                benchmark_rate = self.equipment_benchmarks[instrument_type]['avg_service_rate_yearly']
                actual_rate = relevant_data['contract_revenue'].mean()
                
                price_analysis[instrument_type] = {
                    'price_position': (actual_rate - benchmark_rate) / benchmark_rate,
                    'price_competitiveness': 'premium' if actual_rate > benchmark_rate * 1.1 else 
                                          'competitive' if actual_rate > benchmark_rate * 0.9 else 
                                          'discount'
                }
        
        return price_analysis
    
    def _analyze_service_quality(self, maintenance_data):
        """
        Analyze service quality metrics
        """
        return {
            'pm_compliance': maintenance_data['pm_visits_completed'].sum() / 
                           maintenance_data['pm_visits_scheduled'].sum(),
            'response_efficiency': self._calculate_response_efficiency(maintenance_data),
            'client_satisfaction_proxy': self._calculate_satisfaction_proxy(maintenance_data)
        }
    
    def _analyze_market_share(self, maintenance_data):
        """
        Analyze market share indicators
        """
        total_instruments = len(maintenance_data['instrument_id'].unique())
        
        return {
            'total_instruments_serviced': total_instruments,
            'instrument_type_distribution': maintenance_data['instrument_type'].value_counts().to_dict(),
            'revenue_concentration': self._calculate_revenue_concentration(maintenance_data)
        }
    
    def _calculate_response_efficiency(self, maintenance_data):
        """
        Calculate service response efficiency metrics
        """
        return {
            'avg_response_time': maintenance_data['days_onsite'].mean(),
            'first_time_resolution': len(maintenance_data[maintenance_data['onsite_visits'] == 1]) / 
                                   len(maintenance_data)
        }
    
    def _calculate_satisfaction_proxy(self, maintenance_data):
        """
        Calculate proxy metrics for client satisfaction
        """
        return {
            'contract_renewal_rate': 0.85,  # Placeholder - would need actual renewal data
            'error_rate': maintenance_data['client_errors'].mean(),
            'consultation_ratio': maintenance_data['phone_consultations'].sum() / 
                                maintenance_data['onsite_visits'].sum()
        }
    
    def _calculate_revenue_concentration(self, maintenance_data):
        """
        Calculate revenue concentration metrics
        """
        client_revenue = maintenance_data.groupby('client_id')['contract_revenue'].sum()
        total_revenue = client_revenue.sum()
        
        return {
            'top_client_share': client_revenue.max() / total_revenue,
            'top_5_share': client_revenue.nlargest(5).sum() / total_revenue,
            'herfindahl_index': ((client_revenue / total_revenue) ** 2).sum()
        }
    
    def generate_market_report(self, analysis_results):
        """
        Generate a detailed market analysis report
        """
        report = []
        report.append("Market Analysis Report")
        report.append("===================\n")
        
        # Cost Efficiency Analysis
        report.append("Cost Efficiency Analysis")
        report.append("-----------------------")
        for instrument_type, metrics in analysis_results['cost_efficiency'].items():
            report.append(f"\n{instrument_type.replace('_', ' ').title()}:")
            report.append(f"- Cost vs Benchmark: {metrics['actual_vs_benchmark']:.2%}")
            report.append(f"- Cost per Value: {metrics['cost_per_value']:.2%}")
            report.append(f"- Lifecycle Position: {metrics['lifecycle_position']:.1%}")
        
        # Market Positioning
        report.append("\nMarket Positioning")
        report.append("----------------")
        for instrument_type, position in analysis_results['market_positioning']['price_positioning'].items():
            report.append(f"\n{instrument_type.replace('_', ' ').title()}:")
            report.append(f"- Price Position: {position['price_position']:.2%}")
            report.append(f"- Competitive Position: {position['price_competitiveness']}")
        
        # Service Quality
        report.append("\nService Quality Metrics")
        report.append("--------------------")
        quality = analysis_results['market_positioning']['service_quality_metrics']
        report.append(f"- PM Compliance Rate: {quality['pm_compliance']:.2%}")
        report.append(f"- First Time Resolution: {quality['response_efficiency']['first_time_resolution']:.2%}")
        report.append(f"- Consultation Ratio: {quality['consultation_ratio']:.2f}")
        
        # Market Share
        report.append("\nMarket Share Analysis")
        report.append("-------------------")
        share = analysis_results['market_positioning']['market_share_indicators']
        report.append(f"- Total Instruments Serviced: {share['total_instruments_serviced']}")
        report.append(f"- Revenue Concentration:")
        report.append(f"  * Top Client Share: {share['revenue_concentration']['top_client_share']:.2%}")
        report.append(f"  * Top 5 Clients Share: {share['revenue_concentration']['top_5_share']:.2%}")
        
        # Save report
        with open('market_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

def main():
    # Initialize analyzer
    analyzer = MarketDataAnalyzer()
    
    # Load benchmark data
    analyzer.load_equipment_benchmarks()
    
    # Fetch economic indicators
    analyzer.fetch_economic_indicators()
    
    # Example usage with maintenance data
    maintenance_data = pd.read_csv('maintenance_data.csv')
    
    # Perform market analysis
    analysis_results = analyzer.analyze_market_trends(maintenance_data)
    
    # Generate report
    report = analyzer.generate_market_report(analysis_results)
    print(report)

if __name__ == "__main__":
    main() 