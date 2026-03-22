import pandas as pd

def load_and_print_stats(file_path):
    # Load the data from the file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Print basic statistics of the DataFrame
    print("High-Level Stats:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

if __name__ == '__main__':
    # Specify the path to your data file
    file_path = 'path/to/your/israel-alerts.txt'
    
    # Call the function to load and print stats
    load_and_print_stats(file_path)
