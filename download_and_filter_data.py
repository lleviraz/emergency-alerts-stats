import requests
from datetime import datetime, timedelta
import pandas as pd

def download_csv(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open('israel-alerts.csv', 'wb') as file:
            file.write(response.content)
        print("CSV file downloaded successfully.")
    else:
        print(f"Failed to download CSV file. Status code: {response.status_code}")

def filter_last_90_days(file_path):
    df = pd.read_csv(file_path)
    today = datetime.now()
    ninety_days_ago = today - timedelta(days=90)
    
    filtered_df = df[df['date'] >= ninety_days_ago.strftime('%Y-%m-%d')]
    return filtered_df

def save_filtered_data(filtered_df, output_file):
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")

if __name__ == '__main__':
    url = "https://github.com/dleshem/israel-alerts-data/blob/main/israel-alerts.csv"
    download_csv(url)
    
    filtered_data = filter_last_90_days('israel-alerts.csv')
    save_filtered_data(filtered_data, 'last_90_days_alerts.csv')
