#!/usr/bin/env python3
"""
Creates UMAP visualizations for analyzing cluster separation
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
from collections import Counter
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import colorsys

# Set style for visualization
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class UMAPClusterVisualizer:
    def __init__(self, project_root="/work/shixu/climate_project"):
        self.project_root = Path(project_root)
        self.features_dir = self.project_root / "features"
        self.results_dir = self.project_root / "results"
        self.graph_prep_dir = self.project_root / "MP4VisualFrameDetection" / "graph_prep" / "results"
        
        # Create output directory
        self.output_dir = self.project_root / "analysis" / "umap_cluster_visualization"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"UMAP Cluster Visualizer initialized")
        print(f"Output directory: {self.output_dir}")
    
    def generate_distinct_colors(self, n_colors):
        """Generate perceptually distinct colors for cluster visualization"""
        if n_colors == 0:
            return []
        
        if n_colors == 1:
            return ['#1f77b4']  # Blue
        
        # For small numbers, use predefined distinct colors
        if n_colors <= 12:
            distinct_colors = [
                '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                '#ff7f00', '#ffff33', '#a65628', '#f781bf',
                '#999999', '#1f77b4', '#17becf', '#bcbd22'
            ]
            return distinct_colors[:n_colors]
        
        # For larger numbers, generate colors using golden angle
        colors = []
        golden_angle = 137.508  # Golden angle in degrees
        
        for i in range(n_colors):
            hue = (i * golden_angle) % 360
            saturation = 0.8 if i % 2 == 0 else 0.7
            value = 0.8 if i % 2 == 0 else 0.9
            
            rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
            )
            colors.append(hex_color)
        
        return colors
    
    def create_cluster_color_mapping(self, labels, min_cluster_size=10):
        """Create color mapping for clusters based on size"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort clusters by size
        sorted_indices = np.argsort(counts)[::-1]
        sorted_labels = unique_labels[sorted_indices]
        sorted_counts = counts[sorted_indices]
        
        # Categorize clusters by size
        large_clusters = sorted_labels[sorted_counts >= min_cluster_size]
        medium_clusters = sorted_labels[(sorted_counts >= 5) & (sorted_counts < min_cluster_size)]
        small_clusters = sorted_labels[sorted_counts < 5]
        
        print(f"Found {len(large_clusters)} large clusters (≥{min_cluster_size} videos)")
        print(f"Found {len(medium_clusters)} medium clusters (5-{min_cluster_size-1} videos)")
        print(f"Found {len(small_clusters)} small clusters (<5 videos)")
        
        # Create color mapping
        color_map = {}
        
        # Assign distinct colors to large clusters
        if len(large_clusters) > 0:
            large_colors = self.generate_distinct_colors(len(large_clusters))
            for i, cluster_id in enumerate(large_clusters):
                color_map[cluster_id] = large_colors[i]
        
        # Assign muted colors to medium clusters
        if len(medium_clusters) > 0:
            medium_colors = self.generate_distinct_colors(len(medium_clusters))
            for i, cluster_id in enumerate(medium_clusters):
                base_color = medium_colors[i]
                rgb = tuple(int(base_color[j:j+2], 16)/255 for j in (1, 3, 5))
                hsv = colorsys.rgb_to_hsv(*rgb)
                muted_rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1]*0.4, hsv[2]*0.7)
                muted_hex = '#{:02x}{:02x}{:02x}'.format(
                    int(muted_rgb[0]*255), int(muted_rgb[1]*255), int(muted_rgb[2]*255)
                )
                color_map[cluster_id] = muted_hex
        
        # Assign gray to small clusters
        for cluster_id in small_clusters:
            color_map[cluster_id] = '#d3d3d3'
        
        return color_map, large_clusters, medium_clusters, small_clusters
    
    def load_clustering_data(self, dataset, model, method):
        """Load embeddings, clustering results, and key mapping"""
        print(f"\nLoading data for {dataset}_{model}_{method}...")
        
        # Load embeddings
        embedding_file = self.features_dir / f"{dataset}_{model}_{method}.torch"
        if not embedding_file.exists():
            print(f"Embedding file not found: {embedding_file}")
            return None, None, None
        
        embeddings = torch.load(embedding_file, weights_only=True)
        print(f"Loaded {len(embeddings)} embeddings")
        
        # Load clustering results
        cluster_file = self.results_dir / f"{dataset}_{model}_{method}_clusters.h5"
        if not cluster_file.exists():
            print(f"Cluster file not found: {cluster_file}")
            return None, None, None
        
        with h5py.File(cluster_file, 'r') as f:
            labels = f['labels'][:]
        
        print(f"Loaded clustering with {len(np.unique(labels))} clusters")
        
        # Load key mapping with fallback
        key_mapping_file = self.graph_prep_dir / f"key_mapping_{dataset}_{model}_{method}.torch"
        if not key_mapping_file.exists():
            alternatives = [
                self.graph_prep_dir / f"key_mapping_{dataset}_{model}.torch",
                self.graph_prep_dir / f"key_mapping_{dataset}.torch"
            ]
            
            key_mapping = None
            for alt_file in alternatives:
                if alt_file.exists():
                    key_mapping = torch.load(alt_file, weights_only=True)
                    break
            
            if key_mapping is None:
                embedding_keys = list(embeddings.keys())
                key_mapping = {vid: i for i, vid in enumerate(embedding_keys)}
                print(f"Created mapping for {len(key_mapping)} videos")
        else:
            key_mapping = torch.load(key_mapping_file, weights_only=True)
            print(f"Loaded key mapping with {len(key_mapping)} entries")
        
        return embeddings, labels, key_mapping
    
    def prepare_embeddings_matrix(self, embeddings, labels, vid_to_node, sample_size=5000):
        """Prepare embeddings matrix for UMAP computation"""
        print(f"Preparing embeddings matrix...")
        
        available_video_ids = list(embeddings.keys())
        mapping_video_ids = set(vid_to_node.keys())
        common_video_ids = set(available_video_ids) & mapping_video_ids
        
        if len(common_video_ids) == 0:
            if len(available_video_ids) == len(labels):
                vid_to_node = {vid: i for i, vid in enumerate(available_video_ids)}
                common_video_ids = set(available_video_ids)
                print(f"Created order-based mapping with {len(common_video_ids)} videos")
            else:
                print(f"Cannot create mapping: {len(available_video_ids)} != {len(labels)}")
                return np.array([]), np.array([]), []
        
        # Sample if needed
        common_video_ids = list(common_video_ids)
        if len(common_video_ids) > sample_size:
            # Stratified sampling to ensure representation from different clusters
            video_to_cluster = {}
            for vid in common_video_ids:
                node_id = vid_to_node.get(vid)
                if node_id is not None and node_id < len(labels):
                    video_to_cluster[vid] = labels[node_id]
            
            # Sample from each cluster
            cluster_videos = {}
            for vid, cluster in video_to_cluster.items():
                if cluster not in cluster_videos:
                    cluster_videos[cluster] = []
                cluster_videos[cluster].append(vid)
            
            sampled_ids = []
            for cluster, vids in cluster_videos.items():
                n_sample = min(len(vids), max(1, sample_size // len(cluster_videos)))
                sampled_from_cluster = np.random.choice(vids, n_sample, replace=False)
                sampled_ids.extend(sampled_from_cluster)
            
            if len(sampled_ids) > sample_size:
                sampled_ids = np.random.choice(sampled_ids, sample_size, replace=False)
            
            print(f"Stratified sampling: {len(sampled_ids)} videos from {len(cluster_videos)} clusters")
        else:
            sampled_ids = common_video_ids
        
        # Process embeddings
        embeddings_list = []
        labels_list = []
        video_ids_list = []
        
        for vid in sampled_ids:
            emb = embeddings[vid]
            if isinstance(emb, torch.Tensor):
                emb_np = emb.cpu().numpy()
            else:
                emb_np = np.array(emb)
            
            if len(emb_np.shape) > 1:
                emb_flat = emb_np.flatten()
            else:
                emb_flat = emb_np
            
            node_id = vid_to_node.get(vid)
            if node_id is not None and node_id < len(labels):
                embeddings_list.append(emb_flat)
                labels_list.append(labels[node_id])
                video_ids_list.append(vid)
        
        if len(embeddings_list) == 0:
            return np.array([]), np.array([]), []
        
        embeddings_matrix = np.array(embeddings_list)
        labels_array = np.array(labels_list)
        
        # Normalize embeddings
        embeddings_matrix = embeddings_matrix / (np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-12)
        
        print(f"Prepared embedding matrix: {embeddings_matrix.shape}")
        print(f"Cluster distribution: {Counter(labels_array)}")
        
        return embeddings_matrix, labels_array, video_ids_list
    
    def compute_umap_embedding(self, embeddings_matrix, n_neighbors=15, min_dist=0.1, random_state=42):
        """Compute UMAP embedding"""
        print("Computing UMAP embedding...")
        
        if embeddings_matrix.size == 0 or len(embeddings_matrix.shape) != 2 or embeddings_matrix.shape[0] < 2:
            print("Invalid embeddings matrix for UMAP")
            return None
        
        n_samples = embeddings_matrix.shape[0]
        n_neighbors = min(n_neighbors, n_samples - 1) if n_neighbors >= n_samples else n_neighbors
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, n_neighbors),
            min_dist=min_dist,
            random_state=random_state,
            metric='cosine',
            spread=1.0,
            low_memory=False,
            n_epochs=500,
            init='spectral'
        )
        
        umap_embedding = reducer.fit_transform(embeddings_matrix)
        print(f"UMAP embedding computed: {umap_embedding.shape}")
        
        return umap_embedding
    
    def create_cluster_visualizations(self, umap_embedding, labels, video_ids, 
                                    dataset_name, model_name, method_name,
                                    min_cluster_size=10):
        """Create main cluster visualization plots"""
        print(f"Creating cluster visualizations...")
        
        # Create color mapping
        color_map, large_clusters, medium_clusters, small_clusters = self.create_cluster_color_mapping(
            labels, min_cluster_size
        )
        
        cluster_sizes = Counter(labels)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.8])
        
        # Plot 1: All clusters
        ax1 = fig.add_subplot(gs[0, 0])
        colors = [color_map[label] for label in labels]
        ax1.scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                   c=colors, s=30, alpha=0.7, edgecolors='white', linewidth=0.3)
        ax1.set_title('All Clusters', fontsize=12, fontweight='bold')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Large clusters only
        ax2 = fig.add_subplot(gs[0, 1])
        small_mask = ~np.isin(labels, large_clusters)
        if np.any(small_mask):
            ax2.scatter(umap_embedding[small_mask, 0], umap_embedding[small_mask, 1],
                       c='#f0f0f0', s=8, alpha=0.3, edgecolors='none', zorder=1)
        
        large_mask = np.isin(labels, large_clusters)
        if np.any(large_mask):
            colors = [color_map[label] for label in labels[large_mask]]
            ax2.scatter(umap_embedding[large_mask, 0], umap_embedding[large_mask, 1],
                       c=colors, s=40, alpha=0.8, edgecolors='white', 
                       linewidth=0.5, zorder=2)
            
            # Add labels for top clusters
            for cluster_id in sorted(large_clusters, key=lambda x: cluster_sizes[x], reverse=True)[:6]:
                cluster_mask = labels == cluster_id
                if np.any(cluster_mask):
                    cluster_points = umap_embedding[cluster_mask]
                    centroid = np.mean(cluster_points, axis=0)
                    ax2.annotate(f'C{cluster_id}\n({cluster_sizes[cluster_id]})', 
                               centroid, fontsize=9, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor='black', alpha=0.8),
                               zorder=3)
        
        ax2.set_title(f'Large Clusters Only (≥{min_cluster_size} videos)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.grid(True, alpha=0.2)
        
        # Plot 3: Statistics panel
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        # Statistics
        total_videos = sum(cluster_sizes.values())
        total_clusters = len(cluster_sizes)
        
        stats_text = f"""Cluster Statistics

Total Videos: {total_videos:,}
Total Clusters: {total_clusters}

Large Clusters (≥10): {len(large_clusters)}
Medium Clusters (5-9): {len(medium_clusters)}
Small Clusters (<5): {len(small_clusters)}

Largest Cluster: {max(cluster_sizes.values())} videos
"""
        
        ax3.text(0.05, 0.95, stats_text, ha='left', va='top', 
                fontsize=11, transform=ax3.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        
        # Top clusters legend
        ax3.text(0.05, 0.5, 'Top 8 Clusters:', ha='left', va='top',
                fontsize=12, fontweight='bold', transform=ax3.transAxes)
        
        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:8]
        
        y_pos = 0.45
        for i, (cluster_id, size) in enumerate(top_clusters):
            color = color_map.get(cluster_id, 'gray')
            percentage = (size / total_videos) * 100
            
            # Color square
            ax3.add_patch(patches.Rectangle((0.05, y_pos - 0.015), 0.03, 0.03, 
                                          facecolor=color, edgecolor='black',
                                          transform=ax3.transAxes))
            
            # Text
            ax3.text(0.1, y_pos, f'C{cluster_id}: {size} ({percentage:.1f}%)',
                    ha='left', va='center', fontsize=10, transform=ax3.transAxes)
            
            y_pos -= 0.05
        
        # Main title
        fig.suptitle(f'UMAP Cluster Analysis\n{dataset_name} - {model_name} - {method_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save with high quality
        output_file = self.output_dir / f"umap_clusters_{dataset_name}_{model_name}_{method_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        
        print(f"Saved UMAP visualization: {output_file}")
        
        return fig, (large_clusters, medium_clusters, small_clusters, cluster_sizes, color_map)
    
    def analyze_clustering(self, dataset, model, method, 
                          min_cluster_size=10, sample_size=5000,
                          umap_params=None):
        """Main analysis pipeline for clustering visualization"""
        if umap_params is None:
            umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 42}
        
        print(f"\nAnalyzing {dataset}_{model}_{method}")
        
        # Load data
        embeddings, labels, vid_to_node = self.load_clustering_data(dataset, model, method)
        if embeddings is None:
            return None
        
        # Prepare embeddings matrix
        embeddings_matrix, labels_array, video_ids = self.prepare_embeddings_matrix(
            embeddings, labels, vid_to_node, sample_size
        )
        
        if embeddings_matrix.size == 0 or len(labels_array) == 0:
            print("No valid data found for analysis")
            return None
        
        # Compute UMAP
        umap_embedding = self.compute_umap_embedding(embeddings_matrix, **umap_params)
        if umap_embedding is None:
            print("UMAP computation failed")
            return None
        
        # Create visualizations
        fig, cluster_stats = self.create_cluster_visualizations(
            umap_embedding, labels_array, video_ids,
            dataset, model, method, min_cluster_size
        )
        
        # Save summary statistics
        large_clusters, medium_clusters, small_clusters, cluster_sizes, color_map = cluster_stats
        summary_data = {
            'dataset': dataset,
            'model': model,
            'method': method,
            'total_clusters': len(cluster_sizes),
            'large_clusters': len(large_clusters),
            'medium_clusters': len(medium_clusters),
            'small_clusters': len(small_clusters),
            'cluster_sizes': dict(cluster_sizes),
            'umap_parameters': umap_params,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / f"clustering_summary_{dataset}_{model}_{method}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Saved summary statistics: {summary_file}")
        
        return {
            'embeddings_matrix': embeddings_matrix,
            'umap_embedding': umap_embedding,
            'labels': labels_array,
            'cluster_stats': cluster_stats,
            'color_map': color_map
        }

def main():
    """Run UMAP cluster visualization analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create UMAP visualizations for video clustering analysis')
    parser.add_argument('--dataset', type=str, default='climate_3k_static',
                       help='Dataset name')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['convnextv2', 'dinov2'],
                       help='Model names to analyze')
    parser.add_argument('--method', type=str, default='average',
                       help='Combination method')
    parser.add_argument('--min_cluster_size', type=int, default=10,
                       help='Minimum cluster size for color assignment')
    parser.add_argument('--sample_size', type=int, default=5000,
                       help='Number of videos to sample for visualization')
    parser.add_argument('--project_root', type=str, default='/work/shixu/climate_project',
                       help='Project root directory')
    
    args = parser.parse_args()
    
    print("Starting UMAP cluster visualization analysis...")
    print(f"Dataset: {args.dataset}")
    print(f"Models: {args.models}")
    print(f"Method: {args.method}")
    
    visualizer = UMAPClusterVisualizer(project_root=args.project_root)
    
    results = {}
    for model in args.models:
        print(f"\n{'='*60}")
        result = visualizer.analyze_clustering(
            args.dataset, model, args.method,
            min_cluster_size=args.min_cluster_size,
            sample_size=args.sample_size
        )
        if result:
            results[f"{model}_{args.method}"] = result
    
    print(f"\nUMAP cluster visualization complete!")
    print(f"Results saved to: {visualizer.output_dir}")

if __name__ == "__main__":
    main()