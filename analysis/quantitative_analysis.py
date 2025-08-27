#!/usr/bin/env python3
"""
Quantitative analysis 
Calculates quantitative metrics and weighted overall scores for different configurations

"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import h5py
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
import warnings
warnings.filterwarnings('ignore')

class ClusteringMetricsAnalyzer:
    def __init__(self, project_root="/work/shixu/climate_project"):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "results"
        self.analysis_dir = self.project_root / "analysis" / "clustering_metrics"
        self.features_dir = self.project_root / "features"
        
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Define configurations
        self.frame_selections = ['static', 'diverse']
        self.models = {
            'convnextv2': ['average', 'weighted_diversity', 'max_confidence', 'weighted_confidence'],
            'dinov2': ['average', 'weighted_diversity', 'temporal_coherence']
        }
        
        # Scoring weights for overall score calculation
        self.scoring_weights = {
            'cluster_quality': 0.30,         # Silhouette score
            'cluster_separation': 0.20,      # Davies-Bouldin (inverted)
            'distribution_equality': 0.10,   # 1 - Gini
            'coverage_efficiency': 0.10,     # Coverage per cluster
            'fragmentation': 0.10,           # 1 - singleton ratio
            'calinski_harabasz': 0.20        # Calinski-Harabasz score
        }
        
        self.results_data = []
        
        print("Clustering Metrics Analyzer initialized")
        print(f"Output directory: {self.analysis_dir}")
        
    def compute_clustering_metrics(self, config_name, labels):
        """Compute quality metrics for clustering results"""
        print(f"  Computing metrics for {config_name}...")
        
        # Basic statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique_labels)
        n_videos = len(labels)
        
        # Load embeddings for quality metrics
        embedding_patterns = [
            f"{config_name}.torch",
            f"{config_name.replace('_clusters', '')}.torch",
            config_name.replace('1fps_dinov2_vitb14_lc', 'dinov2') + ".torch",
            config_name.replace('1fps_convnextv2', 'convnextv2') + ".torch"
        ]
        
        embeddings = None
        for pattern in embedding_patterns:
            embedding_file = self.features_dir / pattern
            if embedding_file.exists():
                try:
                    embeddings = torch.load(embedding_file, weights_only=True)
                    print(f"    Loaded embeddings from {embedding_file.name}")
                    break
                except:
                    continue
        
        # Compute quality metrics if embeddings available
        quality_metrics = {}
        
        if embeddings is not None:
            try:
                # Get video IDs that match the clustering
                video_ids = list(embeddings.keys())[:n_videos]
                
                if len(video_ids) == n_videos:
                    embeddings_list = []
                    for vid in video_ids:
                        emb = embeddings[vid]
                        if isinstance(emb, torch.Tensor):
                            emb = emb.cpu().numpy()
                        embeddings_list.append(emb.flatten())
                    
                    embeddings_matrix = np.array(embeddings_list)
                    
                    # Remove singletons for clustering metrics
                    non_singleton_mask = np.array([counts[labels[i]] > 1 for i in range(len(labels))])
                    
                    if np.sum(non_singleton_mask) > 1:
                        # Silhouette Score
                        try:
                            sample_size = min(5000, np.sum(non_singleton_mask))
                            quality_metrics['silhouette_score'] = float(
                                silhouette_score(
                                    embeddings_matrix[non_singleton_mask],
                                    labels[non_singleton_mask],
                                    sample_size=sample_size
                                )
                            )
                        except:
                            quality_metrics['silhouette_score'] = -1.0
                        
                        # Davies-Bouldin Score
                        try:
                            quality_metrics['davies_bouldin_score'] = float(
                                davies_bouldin_score(
                                    embeddings_matrix[non_singleton_mask],
                                    labels[non_singleton_mask]
                                )
                            )
                        except:
                            quality_metrics['davies_bouldin_score'] = 10.0
                        
                        # Calinski-Harabasz Score
                        try:
                            quality_metrics['calinski_harabasz_score'] = float(
                                calinski_harabasz_score(
                                    embeddings_matrix[non_singleton_mask],
                                    labels[non_singleton_mask]
                                )
                            )
                        except:
                            quality_metrics['calinski_harabasz_score'] = 0.0
                    else:
                        quality_metrics['silhouette_score'] = -1.0
                        quality_metrics['davies_bouldin_score'] = 10.0
                        quality_metrics['calinski_harabasz_score'] = 0.0
                else:
                    print(f"    Warning: Embedding count mismatch ({len(video_ids)} vs {n_videos})")
            except Exception as e:
                print(f"    Error computing quality metrics: {e}")
        else:
            print(f"    Warning: No embeddings found, using default quality scores")
            quality_metrics['silhouette_score'] = -1.0
            quality_metrics['davies_bouldin_score'] = 10.0
            quality_metrics['calinski_harabasz_score'] = 0.0
        
        # Size distribution metrics
        sorted_counts = np.sort(counts)[::-1]
        
        metrics = {
            'n_clusters': n_clusters,
            'n_videos': n_videos,
            'largest_cluster': int(np.max(counts)),
            'avg_cluster_size': float(np.mean(counts)),
            'median_cluster_size': float(np.median(counts)),
            'gini_coefficient': self._compute_gini(counts),
            'top1_coverage': float(sorted_counts[0] / n_videos * 100),
            'top5_coverage': float(np.sum(sorted_counts[:5]) / n_videos * 100) if n_clusters >= 5 else 100,
            'top10_coverage': float(np.sum(sorted_counts[:10]) / n_videos * 100) if n_clusters >= 10 else 100,
            'top20_coverage': float(np.sum(sorted_counts[:20]) / n_videos * 100) if n_clusters >= 20 else 100,
            'n_singletons': int(np.sum(counts == 1)),
            'singleton_ratio': float(np.sum(counts == 1) / n_clusters * 100),
            'mega_cluster_ratio': float(np.max(counts) / n_videos * 100),
            **quality_metrics
        }
        
        return metrics
    
    def _compute_gini(self, values):
        """Compute Gini coefficient for distribution inequality"""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return float((2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n)
    
    def calculate_overall_score(self, metrics):
        """Calculate weighted overall score from individual metrics"""
        
        # Normalize metrics to [0, 1]
        normalized = {}
        
        # Silhouette score: [-1, 1] → [0, 1]
        normalized['cluster_quality'] = (metrics.get('silhouette_score', -1) + 1) / 2
        
        # Davies-Bouldin: [0, ∞] → [1, 0] (inverted, capped at 5)
        db_score = metrics.get('davies_bouldin_score', 5)
        normalized['cluster_separation'] = max(0, 1 - db_score / 5)
        
        # Gini coefficient: [0, 1] → [1, 0] (inverted for equality)
        normalized['distribution_equality'] = 1 - metrics['gini_coefficient']
        
        # Coverage efficiency: coverage per cluster (capped at 10% per cluster)
        efficiency = metrics['top10_coverage'] / min(10, metrics['n_clusters'])
        normalized['coverage_efficiency'] = min(1, efficiency / 10)
        
        # Fragmentation: singleton ratio [0, 100] → [1, 0] (inverted)
        normalized['fragmentation'] = 1 - metrics['singleton_ratio'] / 100
        
        # Calinski-Harabasz Score: [0, ∞), normalize by log scale
        ch_score = metrics.get('calinski_harabasz_score', 0.0)
        normalized['calinski_harabasz'] = min(1, np.log1p(ch_score) / 10)
        
        # Calculate weighted score
        overall_score = sum(
            normalized[component] * weight
            for component, weight in self.scoring_weights.items()
        )
        
        # Add component scores to metrics
        metrics['score_components'] = normalized
        metrics['overall_score'] = overall_score
        
        return overall_score
    
    def load_and_analyze_results(self):
        """Load clustering results and compute metrics"""
        print("\nLoading and analyzing clustering results...")
        
        for selection in self.frame_selections:
            for model, methods in self.models.items():
                for method in methods:
                    # Construct filename
                    if model == 'dinov2':
                        pattern = f"climate_3k_{selection}_1fps_{model}_vitb14_lc_{method}"
                    else:
                        pattern = f"climate_3k_{selection}_1fps_{model}_{method}"
                    
                    cluster_file = self.results_dir / f"{pattern}_clusters.h5"
                    
                    if cluster_file.exists():
                        print(f"\n  Processing: {pattern}")
                        
                        # Load cluster labels
                        with h5py.File(cluster_file, 'r') as f:
                            labels = f['labels'][:]
                        
                        # Compute metrics
                        metrics = self.compute_clustering_metrics(pattern, labels)
                        
                        # Calculate overall score
                        overall_score = self.calculate_overall_score(metrics)
                        
                        # Store results
                        result = {
                            'selection': selection,
                            'model': model,
                            'method': method,
                            'config': pattern,
                            **metrics
                        }
                        
                        self.results_data.append(result)
                        
                        # Print summary
                        print(f"    Overall Score: {overall_score:.3f}")
                        print(f"    Silhouette: {metrics.get('silhouette_score', 'N/A'):.3f}")
                        print(f"    Clusters: {metrics['n_clusters']}, Gini: {metrics['gini_coefficient']:.3f}")
                    else:
                        print(f"  Warning: Not found: {cluster_file}")
        
        # Create DataFrame
        self.results_df = pd.DataFrame(self.results_data)
        print(f"\nAnalyzed {len(self.results_data)} configurations")
    
    def save_results_table(self):
        """Save results table with metrics and scores"""
        print("\nSaving results table...")
        
        # Sort by overall score
        sorted_df = self.results_df.sort_values('overall_score', ascending=False)
        
        # Select key columns for output
        output_columns = [
            'selection', 'model', 'method', 'overall_score',
            'silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score',
            'gini_coefficient', 'n_clusters', 'top10_coverage', 'singleton_ratio'
        ]
        
        # Save full results
        sorted_df[output_columns].to_csv(
            self.analysis_dir / 'clustering_metrics_results.csv', 
            index=False
        )
        
        # Save top 10 configurations
        top_10_df = sorted_df.head(10)[output_columns]
        top_10_df.to_csv(
            self.analysis_dir / 'top_10_configurations.csv',
            index=False
        )
        
        # Save scoring weights
        with open(self.analysis_dir / 'scoring_weights.json', 'w') as f:
            json.dump(self.scoring_weights, f, indent=2)
        
        print(f"Results saved to: {self.analysis_dir}")
        
        # Print summary table
        print("\nTop 5 Configurations:")
        print("-" * 80)
        print(f"{'Config':<40} {'Score':<8} {'Silhouette':<12} {'Clusters':<10}")
        print("-" * 80)
        
        for _, row in sorted_df.head(5).iterrows():
            config_short = f"{row['selection'][:4]}_{row['model'][:4]}_{row['method'][:8]}"
            print(f"{config_short:<40} {row['overall_score']:<8.3f} "
                  f"{row['silhouette_score']:<12.3f} {row['n_clusters']:<10}")

def main():
    """Run clustering metrics analysis"""
    print("Starting Climate Video Clustering Metrics Analysis")
    print("=" * 60)
    
    analyzer = ClusteringMetricsAnalyzer()
    
    # Load and analyze results
    analyzer.load_and_analyze_results()
    
    if len(analyzer.results_data) > 0:
        # Save results table
        analyzer.save_results_table()
        
        print("\nAnalysis complete!")
        print(f"Results saved in: {analyzer.analysis_dir}")
    else:
        print("No results found to analyze")

if __name__ == "__main__":
    main()