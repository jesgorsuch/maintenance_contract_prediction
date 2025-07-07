import pandas as pd
import numpy as np
from datetime import datetime
import re

class MaintenanceDataProcessor:
    def __init__(self):
        self.raw_data = None
        self.maintenance_data = None
        self.client_data = {}
        self.service_records = []
        
    def load_quickbooks_data(self, file_path):
        """
        Load and clean QuickBooks transaction data
        """
        # Skip the first empty row and load data
        self.raw_data = pd.read_csv(file_path, skiprows=1)
        
        # Clean column names
        self.raw_data.columns = [col.strip() for col in self.raw_data.columns]
        
        print(f"Loaded {len(self.raw_data)} transactions")
        
    def identify_service_transactions(self):
        """
        Identify and categorize service-related transactions
        """
        # Keywords for service-related transactions
        service_keywords = [
            'repair', 'maintenance', 'service', 'parts', 'labor',
            'diagnostic', 'calibration', 'installation', 'PM visit'
        ]
        
        # Create regex pattern
        pattern = '|'.join(service_keywords)
        
        # Filter service-related transactions
        service_mask = (
            self.raw_data['Line description'].str.contains(pattern, case=False, na=False) |
            self.raw_data['Product/Service'].str.contains(pattern, case=False, na=False) |
            self.raw_data['Item split account full name'].str.contains('Purchases - Parts|Service', case=False, na=False)
        )
        
        service_transactions = self.raw_data[service_mask].copy()
        
        print(f"Found {len(service_transactions)} service-related transactions")
        return service_transactions
    
    def extract_client_info(self):
        """
        Extract unique client information and their instruments
        """
        # Get unique clients from payments
        clients = self.raw_data[
            self.raw_data['Transaction type'] == 'Payment'
        ]['Name'].unique()
        
        for client in clients:
            if pd.isna(client):
                continue
                
            client_transactions = self.raw_data[
                self.raw_data['Name'] == client
            ]
            
            self.client_data[client] = {
                'total_payments': client_transactions[
                    client_transactions['Transaction type'] == 'Payment'
                ]['Amount'].sum(),
                'transaction_count': len(client_transactions),
                'first_transaction': client_transactions['Transaction date'].min(),
                'last_transaction': client_transactions['Transaction date'].max()
            }
        
        print(f"Extracted information for {len(self.client_data)} clients")
        
    def analyze_service_patterns(self):
        """
        Analyze patterns in service transactions
        """
        service_txns = self.identify_service_transactions()
        
        # Group by month to find service patterns
        service_txns['month'] = pd.to_datetime(service_txns['Transaction date']).dt.to_period('M')
        monthly_services = service_txns.groupby('month').agg({
            'Amount': ['count', 'sum'],
            'Line description': 'count'
        })
        
        return monthly_services
    
    def extract_parts_costs(self):
        """
        Extract and analyze parts costs
        """
        parts_txns = self.raw_data[
            self.raw_data['Item split account full name'].str.contains('Parts', case=False, na=False)
        ].copy()
        
        if len(parts_txns) > 0:
            parts_txns['month'] = pd.to_datetime(parts_txns['Transaction date']).dt.to_period('M')
            monthly_parts = parts_txns.groupby('month')['Amount'].sum()
            
            print(f"Found {len(parts_txns)} parts-related transactions")
            return monthly_parts
        
        return pd.Series()
    
    def create_maintenance_records(self):
        """
        Create standardized maintenance records from the transaction data
        """
        service_txns = self.identify_service_transactions()
        
        for _, txn in service_txns.iterrows():
            # Try to extract instrument info from description
            instrument_type = 'unknown'
            for inst_type in ['mass spec', 'chromatograph', 'auto system']:
                if isinstance(txn['Line description'], str) and inst_type in txn['Line description'].lower():
                    instrument_type = inst_type
                    break
            
            record = {
                'client_id': txn['Name'] if pd.notna(txn['Name']) else 'Unknown',
                'transaction_date': txn['Transaction date'],
                'service_type': 'repair' if 'repair' in str(txn['Line description']).lower() else 'maintenance',
                'amount': abs(txn['Amount']),
                'instrument_type': instrument_type,
                'description': txn['Line description']
            }
            
            self.service_records.append(record)
    
    def export_to_maintenance_format(self, output_file):
        """
        Export the processed data to our maintenance prediction format
        """
        if not self.service_records:
            self.create_maintenance_records()
            
        # Convert records to DataFrame
        df = pd.DataFrame(self.service_records)
        
        # Add required columns with estimated values
        df['contract_id'] = 'EST' + df['client_id'].str.replace(' ', '') + df.groupby('client_id').cumcount().astype(str)
        df['instrument_id'] = 'INST' + df.index.astype(str)
        df['instrument_age'] = np.random.uniform(1, 7, len(df))  # Estimated ages
        df['contract_revenue'] = df.groupby('client_id')['amount'].transform('sum')
        df['phone_consultations'] = np.random.randint(2, 10, len(df))  # Estimated
        df['onsite_visits'] = 1
        df['days_onsite'] = np.random.randint(1, 3, len(df))  # Estimated
        df['parts_cost'] = df['amount'] * 0.4  # Estimated parts cost as 40% of total
        df['repair_hours'] = np.ceil(df['amount'] / 150)  # Estimated based on $150/hour rate
        df['pm_visits_scheduled'] = 4  # Quarterly PM visits
        df['pm_visits_completed'] = df.apply(lambda x: np.random.randint(2, 5) if x['service_type'] == 'maintenance' else 2, axis=1)
        df['client_errors'] = np.random.randint(0, 2, len(df))  # Estimated
        df['total_repair_cost'] = df['amount']
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        print(f"Exported {len(df)} maintenance records to {output_file}")
        
        return df
    
    def generate_summary_report(self):
        """
        Generate a summary report of the analysis
        """
        report = []
        report.append("Maintenance Data Analysis Summary")
        report.append("================================\n")
        
        # Client summary
        report.append("Client Summary:")
        report.append("--------------")
        for client, data in self.client_data.items():
            report.append(f"\nClient: {client}")
            report.append(f"Total Payments: ${data['total_payments']:,.2f}")
            report.append(f"Transaction Count: {data['transaction_count']}")
            report.append(f"Date Range: {data['first_transaction']} to {data['last_transaction']}")
        
        # Service patterns
        monthly_services = self.analyze_service_patterns()
        report.append("\nMonthly Service Patterns:")
        report.append("------------------------")
        report.append(str(monthly_services))
        
        # Parts analysis
        monthly_parts = self.extract_parts_costs()
        if not monthly_parts.empty:
            report.append("\nMonthly Parts Costs:")
            report.append("-------------------")
            report.append(str(monthly_parts))
        
        # Save report
        with open('maintenance_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Generated analysis report: maintenance_analysis_report.txt")

def main():
    # Initialize processor
    processor = MaintenanceDataProcessor()
    
    # Load and process data
    processor.load_quickbooks_data('data/Covalent Scientific Industries, LLC_Transaction Detail with customer and item.csv')
    processor.extract_client_info()
    
    # Generate maintenance records
    processor.create_maintenance_records()
    
    # Export to our format
    processor.export_to_maintenance_format('data/processed_maintenance_data.csv')
    
    # Generate summary report
    processor.generate_summary_report()

if __name__ == "__main__":
    main() 