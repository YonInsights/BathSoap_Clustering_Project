import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
def load_data():
    data_features = pd.read_csv('D:/Osiri University/ML/Final Project/BathSoap_Clustering_Project/data/DM_Sheet_cleaned.csv')
    return data_features

# Clean Percentage Columns
def clean_percentage_columns(data_features):
    for col in data_features.columns:
        if data_features[col].dtype == object and data_features[col].str.contains('%').any():
            data_features[col] = data_features[col].str.replace('%', '').astype(float)
    return data_features

# Scale Features
def scale_features(data_features):
    data_features = clean_percentage_columns(data_features)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)
    return data_scaled, data_features

# Find Optimal Clusters (Elbow Method)
def find_optimal_clusters(data_scaled):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot the Elbow Graph
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# Perform K-Means Clustering
def perform_clustering(data_scaled, data_features, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    data_features['Cluster'] = clusters
    return data_features, clusters

# Plot Pair Plot (Choose Specific Features to Reduce Complexity)
def plot_pairplot(data_clustered, selected_features):
    sns.pairplot(data_clustered[selected_features + ['Cluster']], hue='Cluster', palette='Set1')
    plt.title('Pair Plot of Selected Features by Cluster')
    plt.show()

# Summarize Cluster Characteristics
def summarize_clusters(data_clustered):
    cluster_means = data_clustered.groupby('Cluster').mean(numeric_only=True)
    print("Cluster Means:\n", cluster_means)

    # Save Summary
    summary_path = 'D:/Osiri University/ML/Final Project/BathSoap_Clustering_Project/reports/Cluster_Summary.csv'
    cluster_means.to_csv(summary_path)
    print(f"Cluster summary saved to {summary_path}")

    return cluster_means

# Plot Cluster Means
def plot_cluster_means(cluster_means):
    plt.figure(figsize=(12, 8))
    cluster_means.T.plot(kind='bar')
    plt.title('Cluster Means of Features')
    plt.xlabel('Features')
    plt.ylabel('Mean Values')
    plt.legend(title='Cluster')
    plt.tight_layout()

    # Save Plot
    plot_path = r"D:\Osiri University\ML\Final Project\BathSoap_Clustering_Project\reports\cluster_means.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    # Load and scale data
    data_features = load_data()
    data_scaled, data_features = scale_features(data_features)

    # Find optimal clusters
    find_optimal_clusters(data_scaled)

    # Perform clustering
    data_clustered, clusters = perform_clustering(data_scaled, data_features)

    # Summarize and visualize clusters
    selected_features = ['SEC', 'FEH', 'MT', 'SEX', 'AGE'] 
     # Select specific features to visualize
    plot_pairplot(data_clustered, selected_features)

    # Summarize clusters
    cluster_means = summarize_clusters(data_clustered)

    # Plot cluster means
    plot_cluster_means(cluster_means)
