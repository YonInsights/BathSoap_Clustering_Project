import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(df, features, output_dir):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=df)
        plt.title(f'Distribution of {feature} by Cluster')
        plt.savefig(f'{output_dir}/{feature}_distribution_by_cluster.png')
        plt.close()

def plot_cluster_means(cluster_summary, output_path):
    cluster_summary.transpose().plot(kind='bar')
    plt.title('Cluster Means')
    plt.savefig(output_path)
    plt.close()
