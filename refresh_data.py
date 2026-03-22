import pandas as pd

def load_and_print_stats(file_path):
    # Load the data from the file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Filter the DataFrame to include only the last 90 days
    current_date = pd.Timestamp.now()
    start_date = current_date - pd.Timedelta(days=90)
    filtered_df = df[df['date_column'] >= start_date]
    
    # Print basic statistics of the filtered DataFrame
    print("High-Level Stats:")
    print(filtered_df.info())
    print("\nSummary Statistics:")
    print(filtered_df.describe())

if __name__ == '__main__':
    # Specify the path to your data file
    file_path = 'path/to/your/israel-alerts.txt'
    
    # Call the function to load and print stats
    load_and_print_stats(file_path)
