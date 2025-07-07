# Maintenance Data Templates

This directory contains templates and examples for maintenance data collection.

## Files

1. `maintenance_data_template.csv`: Empty template with headers and column descriptions
2. `maintenance_data_sample.csv`: Sample data file with example records

## Data Format Description

### Required Columns

#### Client and Contract Information
- `client_id`: Unique identifier for each client (e.g., CL001)
- `contract_id`: Service agreement identifier (e.g., CNT2023001)
- `contract_start_date`: Contract start date (YYYY-MM-DD)
- `contract_end_date`: Contract end date (YYYY-MM-DD)
- `contract_revenue`: Annual contract value in dollars

#### Instrument Information
- `instrument_id`: Unique identifier for each instrument (e.g., INST001)
- `instrument_type`: Type of scientific instrument
  - Valid values: mass_spectrometer, gas_chromatograph, auto_system_xl
- `instrument_age`: Age of the instrument in years (can use decimals)

#### Service Metrics
- `phone_consultations`: Number of phone support calls
- `onsite_visits`: Number of onsite service visits required
- `days_onsite`: Total days spent onsite for service
- `repair_hours`: Total technician hours spent on repairs
- `pm_visits_scheduled`: Number of preventative maintenance visits scheduled
- `pm_visits_completed`: Number of preventative maintenance visits completed
- `client_errors`: Number of issues caused by client misuse

#### Financial Metrics
- `parts_cost`: Total cost of parts used in repairs (dollars)
- `total_repair_cost`: Total cost of all repairs (dollars)

### Data Guidelines

1. **Dates**: Use YYYY-MM-DD format for all dates
2. **Monetary Values**: Enter as whole numbers without currency symbols or commas
3. **Instrument Types**: Use exact spelling as shown in examples
4. **Numerical Values**: Use whole numbers for counts, decimals allowed for ages
5. **Missing Values**: Leave empty if unknown (don't use 0 unless it's actually zero)

### Example Record
```csv
CL001,CNT2023001,2023-01-01,2023-12-31,INST001,mass_spectrometer,3.5,25000,8,4,6,3500,24,4,4,1,8500
```

This record shows:
- Client CL001 has a contract (CNT2023001) for the year 2023
- They have a mass spectrometer (INST001) that is 3.5 years old
- Contract revenue is $25,000
- Had 8 phone consultations and 4 onsite visits
- Spent 6 days onsite and 24 hours on repairs
- Parts cost $3,500
- All 4 scheduled PM visits were completed
- Had 1 client error
- Total repair cost was $8,500

### Data Collection Best Practices

1. **Consistency**: Use the same format for each record
2. **Completeness**: Try to fill all fields when possible
3. **Accuracy**: Double-check numerical values and dates
4. **Updates**: Update records as new information becomes available
5. **Validation**: Ensure:
   - Dates are valid
   - PM visits completed ≤ PM visits scheduled
   - Total repair cost includes parts cost
   - Contract dates are logical (end date after start date) 