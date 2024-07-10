import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset with the specified encoding
data = pd.read_csv('data/IMDB_Top250_Tvshows.csv', encoding='ISO-8859-1')

# Display the first few rows of the dataset to ensure it's loaded correctly
print("First few rows of the dataset:")
print(data.head())

# Display the column names to verify they are correct
print("\nColumn names in the dataset:")
print(data.columns)

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop rows with missing values (or handle them appropriately)
data = data.dropna()

# Clean the 'Year' column to ensure it contains numeric values
def extract_start_year(year):
    # Replace the non-standard dash character with a standard hyphen
    year = year.replace('\x96', '-')
    if '-' in year:
        return int(year.split('-')[0])
    else:
        return int(year)

data['Year'] = data['Year'].apply(extract_start_year)

# Clean the 'Total_episodes' column to extract numeric values
def extract_episodes(episodes):
    return int(episodes.split()[0])

data['Total_episodes'] = data['Total_episodes'].apply(extract_episodes)

# Handle 'Age' column - converting age ratings to numeric values if necessary
def convert_age(age):
    age_dict = {
        'G': 0,
        'PG': 1,
        'PG-13': 2,
        'R': 3,
        'NC-17': 4
    }
    return age_dict.get(age, -1)  # Return -1 for unknown ratings

data['Age'] = data['Age'].apply(convert_age)

# Normalize numerical features (e.g., 'Year', 'Rating', 'Total_episodes', 'Age')
scaler = MinMaxScaler()
data[['Year', 'Rating', 'Total_episodes', 'Age']] = scaler.fit_transform(data[['Year', 'Rating', 'Total_episodes', 'Age']])

# Display the preprocessed data
print("\nPreprocessed data:")
print(data.head())

# Save the preprocessed data
data.to_csv('data/preprocessed_data.csv', index=False)