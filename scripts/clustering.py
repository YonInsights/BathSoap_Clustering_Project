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
from sklearn.cluster import KMeans

# Set the optimal number of clusters
num_clusters = 4  # the number determined from the graph

# Initialize K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model to your scaled data
kmeans.fit(data_scaled)

from sklearn.cluster import KMeans
import pandas as pd

# Assuming data_scaled is your scaled dataset and data_features is the original data
# Perform K-Means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(data_scaled)

# Add the cluster labels to the original data
data_features['Cluster'] = cluster_labels

# Save the clustered data to a CSV file
output_file_path = 'D:/Osiri University/ML/Final Project/BathSoap_Clustering_Project/data/DM_Sheet_clustered.csv'
data_features.to_csv(output_file_path, index=False)

# Print the first few rows to confirm
print(data_features.head())

# Confirm the file has been saved
import os
if os.path.exists(output_file_path):
    print(f"File saved successfully at {output_file_path}")
else:
    print("File not found, please check the file path and permissions.")

from sklearn.cluster import KMeans
import pandas as pd

# Assuming data_scaled is your scaled dataset
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(data_scaled)

# Add the cluster labels to your original data
data_features['Cluster'] = cluster_labels

# Save the clustered data
data_features.to_csv('D:\Osiri University\ML\Final Project\BathSoap_Clustering_Project\data\DM_Sheet_clustered.csv', index=False)

# Print the first few rows to confirm
print(data_features.head())
# Based on your decision from the Elbow method
num_clusters = 4  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
# Assuming data_features is my DataFrame with features
data_clustered = data_features.copy()  
data_clustered['Cluster'] = clusters

cluster_means = data_clustered.groupby('Cluster').mean(numeric_only=True)
print(cluster_means)
#Visualize the Clusters with a Pair Plot
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(data_clustered, hue='Cluster', palette='Set1')
plt.title('Pair Plot of Features by Cluster')
plt.show()
#Create a Summary of Cluster Characteristics
cluster_summary = clusters.describe().transpose()
print(cluster_summary)

# Summarize the characteristics of each cluster
cluster_summary = clusters.describe().transpose()

# Print the cluster summary
print("Cluster Summary:")
print(cluster_summary)

# Save the summary as a CSV file for later reference
cluster_summary.to_csv('D:/Osiri University/ML/Final Project/BathSoap_Clustering_Project/reports/Cluster_Summary.csv')

import matplotlib.pyplot as plt

# Assuming 'cluster_means' is DataFrame containing the mean values for each cluster.
plt.figure(figsize=(10, 6))
cluster_means.T.plot(kind='bar')
plt.title('Cluster Means of Features')
plt.xlabel('Features')
plt.ylabel('Mean Values')
plt.legend(title='Cluster')
plt.tight_layout()

# Save the plot to the report folder
plt.savefig(r"D:\Osiri University\ML\Final Project\BathSoap_Clustering_Project\reports\cluster_means.png")
plt.show()


