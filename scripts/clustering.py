import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data():
    # Load the cleaned data
    data_features = pd.read_csv('D:/Osiri University/ML/Final Project/BathSoap_Clustering_Project/data/DM_Sheet_cleaned.csv')
    return data_features

def clean_percentage_columns(data_features):
    # Identify columns that contain percentages and clean them
    for col in data_features.columns:
        if data_features[col].dtype == object and data_features[col].str.contains('%').any():
            data_features[col] = data_features[col].str.replace('%', '').astype(float)
    return data_features

def scale_features(data_features):
    # Clean percentage columns
    data_features = clean_percentage_columns(data_features)
    
    # Scale the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)
    return data_scaled

def find_optimal_clusters(data_scaled):
    # Use the Elbow Method to find the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot the Elbow graph
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

if __name__ == "__main__":
    # Load and scale the data
    data_features = load_data()
    data_scaled = scale_features(data_features)
    
    # Find the optimal number of clusters
    find_optimal_clusters(data_scaled)
from sklearn.cluster import KMeans

# Define the number of clusters (you need to set this)
num_clusters = num_clusters = 4

# Initialize K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model
kmeans.fit(data_scaled)



