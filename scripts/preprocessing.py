import pandas as pd

def load_and_preprocess_data():
    # Load the original data
    data = pd.read_csv('D:/Osiri University/ML/Final Project/BathSoap_Clustering_Project/data/DM_Sheet.csv')

    # Drop rows with missing values
    data_cleaned = data.dropna()

    # Save the cleaned data
    data_cleaned.to_csv('D:/Osiri University/ML/Final Project/BathSoap_Clustering_Project/data/DM_Sheet_cleaned.csv', index=False)

    # Select relevant features for clustering
    features = [
        'SEC', 'FEH', 'MT', 'AGE', 'HS', 'Affluence Index',
        'Total Volume', 'No. of Brands', 'Value', 'Vol/Tran'
    ]

    # Create a new DataFrame with the selected features
    data_features = data_cleaned[features]

    # Return the preprocessed features
    return data_features

# If you want to test this function in the script, you can use the following lines:
if __name__ == "__main__":
    data_features = load_and_preprocess_data()
    print("Selected Features for Clustering:")
    print(data_features.head())
