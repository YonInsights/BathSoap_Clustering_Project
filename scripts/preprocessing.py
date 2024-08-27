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

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    return data_scaled

def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

# Function to save the cluster summary and means
def save_cluster_summary(df, clusters, output_path):
    df['Cluster'] = clusters
    cluster_summary = df.groupby('Cluster').mean()
    cluster_summary.to_csv(output_path)
    return cluster_summary
