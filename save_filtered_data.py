import pandas as pd

def save_filtered_data(input_file, output_file):
    # Load the data from the input file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Filter the DataFrame to include only the last 90 days
    current_date = pd.Timestamp.now()
    start_date = current_date - pd.Timedelta(days=90)
    filtered_df = df[df['date_column'] >= start_date]
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    # Specify the paths to your input and output files
    input_file = 'path/to/your/israel-alerts.txt'
    output_file = 'path/to/your/israel-alerts_filtered.csv'
    
    # Call the function to save the filtered data
    save_filtered_data(input_file, output_file)
