#!/usr/bin/env python3
"""
Clustering result Analysis

"""

import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ClimateVideoAnalyzer:
    def __init__(self, project_root="/work/shixu/climate_project"):
        self.project_root = Path(project_root)
        self.features_dir = self.project_root / "features"
        self.results_dir = self.project_root / "results"
        self.analysis_base_dir = self.project_root / "analysis"
        self.graph_prep_dir = self.project_root / "MP4VisualFrameDetection" / "graph_prep" / "results"
        
        # Create base analysis directory
        self.analysis_base_dir.mkdir(exist_ok=True)
        
        print(f"Climate Video Analyzer initialized")
        print(f"Features: {self.features_dir}")
        print(f"Results: {self.results_dir}")
        print(f"Analysis base: {self.analysis_base_dir}")
    
    def convert_numpy_to_python(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_to_python(item) for item in obj)
        else:
            return obj
    
    def get_analysis_dir(self, dataset_name, model, method):
        """Create and return specific analysis directory"""
        analysis_dir = self.analysis_base_dir / f"{dataset_name}_{model}_{method}"
        analysis_dir.mkdir(exist_ok=True)
        return analysis_dir
    
    def compute_clustering_quality_metrics(self, embeddings_dict, labels):
        """Compute clustering quality metrics"""
        print("Computing clustering quality metrics...")
        
        # Convert embeddings dict to matrix
        video_ids = list(embeddings_dict.keys())
        embeddings_list = []
        
        for vid in video_ids:
            emb = embeddings_dict[vid]
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            embeddings_list.append(emb.flatten())
        
        embeddings_matrix = np.array(embeddings_list)
        
        # Remove singletons for some metrics
        unique_labels, counts = np.unique(labels, return_counts=True)
        non_singleton_mask = np.isin(labels, unique_labels[counts > 1])
        
        metrics = {}
        
        if np.sum(non_singleton_mask) < 2:
            metrics['note'] = 'Too few non-singleton clusters for quality metrics'
            return metrics
        
        # Silhouette Score
        try:
            sample_size = min(5000, np.sum(non_singleton_mask))
            if sample_size > 1:
                metrics['silhouette_score'] = float(silhouette_score(
                    embeddings_matrix[non_singleton_mask], 
                    labels[non_singleton_mask],
                    sample_size=sample_size
                ))
        except Exception as e:
            print(f"Could not compute silhouette score: {e}")
            metrics['silhouette_score'] = None
        
        # Davies-Bouldin Score
        try:
            if len(unique_labels[counts > 1]) > 1:
                metrics['davies_bouldin_score'] = float(davies_bouldin_score(
                    embeddings_matrix[non_singleton_mask], 
                    labels[non_singleton_mask]
                ))
        except Exception as e:
            print(f"Could not compute Davies-Bouldin score: {e}")
            metrics['davies_bouldin_score'] = None
        
        # Calinski-Harabasz Score
        try:
            if len(unique_labels[counts > 1]) > 1:
                metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(
                    embeddings_matrix[non_singleton_mask], 
                    labels[non_singleton_mask]
                ))
        except Exception as e:
            print(f"Could not compute Calinski-Harabasz score: {e}")
            metrics['calinski_harabasz_score'] = None
        
        print(f"Quality metrics computed")
        return metrics
    
    def analyze_single_result(self, dataset_name, model, method):
        """Analyze a single clustering result"""
        print(f"\nAnalyzing: {dataset_name}_{model}_{method}")
        
        # Create specific analysis directory
        analysis_dir = self.get_analysis_dir(dataset_name, model, method)
        print(f"Output directory: {analysis_dir}")
        
        # Load embeddings
        embedding_file = self.features_dir / f"{dataset_name}_{model}_{method}.torch"
        if not embedding_file.exists():
            embedding_file = self.features_dir / f"{dataset_name}_{model}.torch"
            if not embedding_file.exists():
                print(f"Embedding file not found: {embedding_file}")
                return None
        
        embeddings = torch.load(embedding_file, weights_only=True)
        print(f"Loaded {len(embeddings)} embeddings")
        
        # Load clustering results
        cluster_file = self.results_dir / f"{dataset_name}_{model}_{method}_clusters.h5"
        if not cluster_file.exists():
            cluster_file = self.results_dir / f"{dataset_name}_{model}_clusters.h5"
            if not cluster_file.exists():
                print(f"Cluster file not found: {cluster_file}")
                return None
        
        with h5py.File(cluster_file, 'r') as f:
            labels = f['labels'][:]
        
        # Load key mapping
        key_mapping_file = self.graph_prep_dir / f"key_mapping_{dataset_name}_{model}_{method}.torch"
        if not key_mapping_file.exists():
            key_mapping_file = self.graph_prep_dir / f"key_mapping_{dataset_name}_{model}.torch"
        
        reverse_mapping = None
        if key_mapping_file.exists():
            key_mapping = torch.load(key_mapping_file, weights_only=True)
            reverse_mapping = {v: k for k, v in key_mapping.items()}
        else:
            print(f"Key mapping not found, using node indices")
            reverse_mapping = {i: f"video_{i}" for i in range(len(labels))}
        
        # Basic cluster analysis
        unique_labels = np.unique(labels)
        cluster_sizes = Counter(labels)
        
        # Create cluster statistics
        cluster_info = []
        for label in unique_labels:
            size = cluster_sizes[label]
            
            # Get sample videos
            sample_videos = []
            if reverse_mapping:
                node_ids = np.where(labels == label)[0][:10]
                sample_videos = [reverse_mapping.get(nid, f"node_{nid}") for nid in node_ids]
            
            cluster_info.append({
                'cluster_id': int(label),
                'size': int(size),
                'sample_videos': sample_videos
            })
        
        # Sort by size
        cluster_info = sorted(cluster_info, key=lambda x: x['size'], reverse=True)
        
        # Compute quality metrics
        quality_metrics = self.compute_clustering_quality_metrics(embeddings, labels)
        
        # Calculate basic statistics
        cluster_sizes_list = [info['size'] for info in cluster_info]
        
        # Statistics
        stats = {
            'dataset': dataset_name,
            'model': model,
            'method': method,
            'num_videos': len(labels),
            'num_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes_list,
            'avg_cluster_size': float(np.mean(cluster_sizes_list)),
            'median_cluster_size': float(np.median(cluster_sizes_list)),
            'max_cluster_size': int(max(cluster_sizes_list)),
            'min_cluster_size': int(min(cluster_sizes_list)),
            'singleton_clusters': sum(1 for info in cluster_info if info['size'] == 1),
            'large_clusters_10plus': sum(1 for info in cluster_info if info['size'] >= 10),
            'large_clusters_50plus': sum(1 for info in cluster_info if info['size'] >= 50),
            'large_clusters_100plus': sum(1 for info in cluster_info if info['size'] >= 100),
            'top_20_clusters': cluster_info[:20],
            'quality_metrics': quality_metrics
        }
        
        # Convert numpy types before saving
        stats_json_safe = self.convert_numpy_to_python(stats)
        
        # Save statistics
        stats_file = analysis_dir / "clustering_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_json_safe, f, indent=2)
        
        # Save cluster details
        cluster_details_file = analysis_dir / "all_clusters.json"
        with open(cluster_details_file, 'w') as f:
            json.dump(self.convert_numpy_to_python(cluster_info), f, indent=2)
        
        # Save top clusters for inspection
        top_clusters_df = pd.DataFrame(cluster_info[:50])
        top_clusters_df.to_csv(analysis_dir / "top_50_clusters.csv", index=False)
        
        # Create visualizations
        self.create_single_result_visualizations(stats, cluster_info, analysis_dir)
        
        print(f"Analysis complete for {dataset_name}_{model}_{method}")
        
        return stats
    
    def create_single_result_visualizations(self, stats, cluster_info, output_dir):
        """Create visualizations for a single result"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster size distribution (histogram)
        cluster_sizes = [c['size'] for c in cluster_info]
        axes[0,0].hist(cluster_sizes, bins=50, edgecolor='black', alpha=0.7)
        axes[0,0].set_title('Cluster Size Distribution', fontsize=14)
        axes[0,0].set_xlabel('Cluster Size')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_yscale('log')
        
        # Add quality metric to title if available
        if stats.get('quality_metrics', {}).get('silhouette_score') is not None:
            silhouette = stats['quality_metrics']['silhouette_score']
            axes[0,0].text(0.95, 0.95, f'Silhouette: {silhouette:.3f}', 
                          transform=axes[0,0].transAxes, ha='right', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Top 20 clusters
        top_20 = cluster_info[:20]
        x_pos = np.arange(len(top_20))
        axes[0,1].bar(x_pos, [c['size'] for c in top_20])
        axes[0,1].set_title('Top 20 Largest Clusters', fontsize=14)
        axes[0,1].set_xlabel('Cluster Rank')
        axes[0,1].set_ylabel('Number of Videos')
        axes[0,1].set_xticks(x_pos[::2])
        axes[0,1].set_xticklabels([str(i+1) for i in x_pos[::2]])
        
        # 3. Cluster size categories (pie chart)
        categories = {
            'Singletons': stats['singleton_clusters'],
            'Small (2-9)': sum(1 for c in cluster_info if 2 <= c['size'] <= 9),
            'Medium (10-49)': sum(1 for c in cluster_info if 10 <= c['size'] <= 49),
            'Large (50-99)': sum(1 for c in cluster_info if 50 <= c['size'] <= 99),
            'Very Large (100+)': stats['large_clusters_100plus']
        }
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v > 0}
        
        if categories:
            axes[1,0].pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('Distribution of Cluster Sizes', fontsize=14)
        else:
            axes[1,0].text(0.5, 0.5, 'No cluster data available', 
                         ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Distribution of Cluster Sizes', fontsize=14)
        
        # 4. Cumulative coverage
        sorted_sizes = sorted([c['size'] for c in cluster_info], reverse=True)
        cumulative = np.cumsum(sorted_sizes)
        total_videos = sum(sorted_sizes)
        cumulative_pct = (cumulative / total_videos) * 100
        
        axes[1,1].plot(range(1, len(cumulative) + 1), cumulative_pct, linewidth=2)
        axes[1,1].set_title('Cumulative Video Coverage by Cluster Rank', fontsize=14)
        axes[1,1].set_xlabel('Number of Clusters')
        axes[1,1].set_ylabel('Cumulative % of Videos')
        axes[1,1].grid(True)
        axes[1,1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
        axes[1,1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80%')
        axes[1,1].legend()
        
        # Find how many clusters cover 80% of videos
        clusters_for_80pct = np.argmax(cumulative_pct >= 80) + 1
        axes[1,1].axvline(x=clusters_for_80pct, color='g', linestyle='--', alpha=0.5)
        axes[1,1].text(clusters_for_80pct + 5, 70, f'{clusters_for_80pct} clusters\ncover 80%', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "clustering_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "clustering_analysis.pdf", bbox_inches='tight')
        plt.close()
        
        # Additional visualization: Cluster size rank plot (log scale)
        plt.figure(figsize=(10, 6))
        ranks = range(1, len(cluster_sizes) + 1)
        plt.loglog(ranks, sorted(cluster_sizes, reverse=True), 'b-', linewidth=2)
        plt.xlabel('Cluster Rank')
        plt.ylabel('Cluster Size')
        plt.title('Cluster Size vs Rank (Log-Log Scale)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add power law fit if applicable
        if len(cluster_sizes) > 10:
            from scipy import stats as scipy_stats
            log_ranks = np.log10(ranks[:20])  # Top 20 for fit
            log_sizes = np.log10(sorted(cluster_sizes, reverse=True)[:20])
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_ranks, log_sizes)
            plt.text(0.05, 0.95, f'Power law exponent: {slope:.2f}\nR²: {r_value**2:.3f}',
                    transform=plt.gca().transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(output_dir / "cluster_rank_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_all_results(self, dataset_name="climate_3k_static"):
        """Analyze all available results"""
        print(f"\nAnalyzing all results for {dataset_name}...")
        
        # Find all cluster files
        cluster_files = list(self.results_dir.glob(f"{dataset_name}_*_clusters.h5"))
        
        all_stats = []
        for cluster_file in cluster_files:
            # Parse filename
            parts = cluster_file.stem.split('_')
            if len(parts) >= 4:
                # Reconstruct dataset name (might have underscores)
                dataset_parts = []
                model_idx = -1
                
                # Find where the model name starts (known models)
                known_models = ['convnextv2', 'dinov2']
                for i, part in enumerate(parts):
                    if any(model in part.lower() for model in known_models):
                        model_idx = i
                        break
                    dataset_parts.append(part)
                
                if model_idx > 0:
                    dataset = '_'.join(dataset_parts)
                    model = parts[model_idx]
                    method = parts[model_idx + 1] if len(parts) > model_idx + 1 and parts[model_idx + 1] != 'clusters' else 'average'
                    
                    if dataset.startswith(dataset_name):
                        stats = self.analyze_single_result(dataset, model, method)
                        if stats:
                            all_stats.append(stats)
        
        if len(all_stats) == 0:
            print("No results found to analyze")
        
        return all_stats


def main():
    """Run climate video clustering analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze climate video clustering results')
    parser.add_argument('--dataset', type=str, default='climate_3k_static',
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='convnextv2',
                       help='Model name')
    parser.add_argument('--method', type=str, default=None,
                       help='Specific method to analyze. If not specified, analyzes all.')
    parser.add_argument('--project_root', type=str, default='/work/shixu/climate_project',
                       help='Project root directory')
    
    args = parser.parse_args()
    
    print("Starting analysis...")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Method: {args.method or 'all'}")
    
    analyzer = ClimateVideoAnalyzer(project_root=args.project_root)
    
    if args.method:
        # Analyze specific result
        analyzer.analyze_single_result(args.dataset, args.model, args.method)
    else:
        # Analyze all results
        analyzer.analyze_all_results(args.dataset)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()