# src/profiling.py
def create_cluster_profiles(cluster_summary, output_path):
    profile_report = ''
    for cluster_id in cluster_summary.columns:
        profile_report += f"Cluster {cluster_id} Profile:\n"
        profile_report += "-" * 30 + "\n"
        for feature, value in cluster_summary[cluster_id].items():
            profile_report += f"{feature}: {value:.2f}\n"
        profile_report += "\n"
    
    # Save the profile report
    with open(output_path, 'w') as file:
        file.write(profile_report)
    
    return profile_report
