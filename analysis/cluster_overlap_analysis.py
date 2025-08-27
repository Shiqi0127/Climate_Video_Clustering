#!/usr/bin/env python3
"""
Analyzes overlap between DINOv2 and ConvNeXt V2 clustering results to examine how videos are distributed across clusters in both models.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

class ClusterOverlapAnalyzer:
    def __init__(self, dinov2_path, convnext_path, key_mapping_path=None):
        """
        Initialize overlap analyzer with paths to cluster files.
        
        Args:
            dinov2_path: Path to DINOv2 clusters h5 file
            convnext_path: Path to ConvNeXt V2 clusters h5 file
            key_mapping_path: Optional path to key mapping file
        """
        self.dinov2_path = Path(dinov2_path)
        self.convnext_path = Path(convnext_path)
        
        # Load cluster assignments
        self.dinov2_labels = self._load_clusters(self.dinov2_path)
        self.convnext_labels = self._load_clusters(self.convnext_path)
        
        # Load key mapping if available
        self.video_ids = None
        if key_mapping_path and Path(key_mapping_path).exists():
            key_mapping = torch.load(key_mapping_path, weights_only=True)
            self.video_ids = [key_mapping.get(i, f"video_{i}") for i in range(len(self.dinov2_labels))]
        
        print(f"Loaded DINOv2 clusters: {len(np.unique(self.dinov2_labels))} clusters")
        print(f"Loaded ConvNeXt V2 clusters: {len(np.unique(self.convnext_labels))} clusters")
        print(f"Total videos: {len(self.dinov2_labels)}")
    
    def _load_clusters(self, path):
        """Load cluster labels from h5 file"""
        with h5py.File(path, 'r') as f:
            return f['labels'][:]
    
    def analyze_overlap(self, top_k=10):
        """
        Analyze overlap between top K clusters from both models.
        
        Args:
            top_k: Number of top clusters to analyze
        """
        # Get top clusters by size
        dinov2_top = self._get_top_clusters(self.dinov2_labels, top_k)
        convnext_top = self._get_top_clusters(self.convnext_labels, top_k)
        
        # Create overlap matrix
        overlap_matrix = np.zeros((len(dinov2_top), len(convnext_top)))
        overlap_percentages = np.zeros((len(dinov2_top), len(convnext_top)))
        
        for i, dino_cluster in enumerate(dinov2_top):
            dino_mask = self.dinov2_labels == dino_cluster
            dino_size = np.sum(dino_mask)
            
            for j, conv_cluster in enumerate(convnext_top):
                conv_mask = self.convnext_labels == conv_cluster
                
                # Count videos in both clusters
                overlap = np.sum(dino_mask & conv_mask)
                overlap_matrix[i, j] = overlap
                overlap_percentages[i, j] = (overlap / dino_size) * 100 if dino_size > 0 else 0
        
        return dinov2_top, convnext_top, overlap_matrix, overlap_percentages
    
    def _get_top_clusters(self, labels, top_k):
        """Get top K clusters by size"""
        unique, counts = np.unique(labels, return_counts=True)
        top_indices = np.argsort(counts)[::-1][:top_k]
        return unique[top_indices]
    
    def find_near_complete_overlaps(self, threshold=80):
        """
        Find cluster pairs where one cluster is almost entirely contained in another.
        
        Args:
            threshold: Percentage threshold for considering near complete overlap
        """
        dinov2_unique = np.unique(self.dinov2_labels)
        convnext_unique = np.unique(self.convnext_labels)
        
        near_complete_overlaps = []
        
        # Check DINOv2 clusters contained in ConvNeXt clusters
        for dino_cluster in dinov2_unique:
            dino_mask = self.dinov2_labels == dino_cluster
            dino_size = np.sum(dino_mask)
            
            if dino_size < 5:  # Skip very small clusters
                continue
            
            for conv_cluster in convnext_unique:
                conv_mask = self.convnext_labels == conv_cluster
                overlap = np.sum(dino_mask & conv_mask)
                
                percentage = (overlap / dino_size) * 100
                if percentage >= threshold:
                    near_complete_overlaps.append({
                        'type': 'DINOv2_in_ConvNeXt',
                        'dinov2_cluster': int(dino_cluster),
                        'convnext_cluster': int(conv_cluster),
                        'dinov2_size': int(dino_size),
                        'overlap_count': int(overlap),
                        'overlap_percentage': percentage
                    })
        
        # Check ConvNeXt clusters contained in DINOv2 clusters
        for conv_cluster in convnext_unique:
            conv_mask = self.convnext_labels == conv_cluster
            conv_size = np.sum(conv_mask)
            
            if conv_size < 5:  # Skip very small clusters
                continue
            
            for dino_cluster in dinov2_unique:
                dino_mask = self.dinov2_labels == dino_cluster
                overlap = np.sum(dino_mask & conv_mask)
                
                percentage = (overlap / conv_size) * 100
                if percentage >= threshold:
                    near_complete_overlaps.append({
                        'type': 'ConvNeXt_in_DINOv2',
                        'convnext_cluster': int(conv_cluster),
                        'dinov2_cluster': int(dino_cluster),
                        'convnext_size': int(conv_size),
                        'overlap_count': int(overlap),
                        'overlap_percentage': percentage
                    })
        
        return near_complete_overlaps
    
    def visualize_overlap_heatmap(self, top_k=10, save_path=None):
        """Create heatmap visualization of cluster overlaps"""
        dinov2_top, convnext_top, overlap_matrix, overlap_percentages = self.analyze_overlap(top_k)
        
        # Calculate percentages from ConvNeXt perspective
        convnext_percentages = np.zeros_like(overlap_percentages)
        for j, conv_cluster in enumerate(convnext_top):
            conv_size = np.sum(self.convnext_labels == conv_cluster)
            if conv_size > 0:
                convnext_percentages[:, j] = (overlap_matrix[:, j] / conv_size) * 100
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Absolute counts heatmap
        sns.heatmap(overlap_matrix, 
                    xticklabels=[f"C{c}" for c in convnext_top],
                    yticklabels=[f"D{c}" for c in dinov2_top],
                    annot=True, fmt='g', cmap='YlOrRd', ax=ax1)
        ax1.set_xlabel('ConvNeXt V2 Clusters')
        ax1.set_ylabel('DINOv2 Clusters')
        ax1.set_title('Overlap Count Between Top Clusters')
        
        # Percentage heatmap - DINOv2 perspective
        sns.heatmap(overlap_percentages,
                    xticklabels=[f"C{c}" for c in convnext_top],
                    yticklabels=[f"D{c}" for c in dinov2_top],
                    annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': '% of DINOv2 cluster'})
        ax2.set_xlabel('ConvNeXt V2 Clusters')
        ax2.set_ylabel('DINOv2 Clusters')
        ax2.set_title('% of DINOv2 Cluster in ConvNeXt V2 Cluster')
        
        # Percentage heatmap - ConvNeXt perspective
        sns.heatmap(convnext_percentages,
                    xticklabels=[f"C{c}" for c in convnext_top],
                    yticklabels=[f"D{c}" for c in dinov2_top],
                    annot=True, fmt='.1f', cmap='Blues', ax=ax3, cbar_kws={'label': '% of ConvNeXt V2 cluster'})
        ax3.set_xlabel('ConvNeXt V2 Clusters')
        ax3.set_ylabel('DINOv2 Clusters')
        ax3.set_title('% of ConvNeXt V2 Cluster in DINOv2 Cluster')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def get_shared_videos(self, dinov2_cluster, convnext_cluster):
        """Get video IDs that appear in both specified clusters"""
        dino_mask = self.dinov2_labels == dinov2_cluster
        conv_mask = self.convnext_labels == convnext_cluster
        shared_mask = dino_mask & conv_mask
        
        shared_indices = np.where(shared_mask)[0]
        
        if self.video_ids:
            return [self.video_ids[i] for i in shared_indices]
        else:
            return [f"video_{i}" for i in shared_indices]
    
    def generate_overlap_report(self, save_path=None):
        """Generate cluster overlap analysis report"""
        report = ["# Cluster Overlap Analysis Report\n"]
        report.append(f"## Dataset Overview")
        report.append(f"- Total videos: {len(self.dinov2_labels):,}")
        report.append(f"- DINOv2 clusters: {len(np.unique(self.dinov2_labels))}")
        report.append(f"- ConvNeXt V2 clusters: {len(np.unique(self.convnext_labels))}\n")
        
        # Top cluster overlaps
        report.append("## Top 10 Cluster Overlaps\n")
        dinov2_top, convnext_top, overlap_matrix, overlap_percentages = self.analyze_overlap(10)
        
        # Find strongest overlaps
        max_overlaps = []
        for i, dino_cluster in enumerate(dinov2_top):
            for j, conv_cluster in enumerate(convnext_top):
                if overlap_matrix[i, j] > 10:  # At least 10 videos
                    max_overlaps.append({
                        'dinov2': dino_cluster,
                        'convnext': conv_cluster,
                        'count': overlap_matrix[i, j],
                        'percentage': overlap_percentages[i, j]
                    })
        
        max_overlaps.sort(key=lambda x: x['count'], reverse=True)
        
        for overlap in max_overlaps[:20]:
            report.append(f"- DINOv2 cluster {overlap['dinov2']} ∩ ConvNeXt cluster {overlap['convnext']}: "
                         f"{int(overlap['count'])} videos ({overlap['percentage']:.1f}% of DINOv2 cluster)")
        
        # Near-complete overlaps
        report.append("\n## Near-Complete Overlaps (≥80%)\n")
        near_complete = self.find_near_complete_overlaps(80)
        
        if near_complete:
            for item in near_complete[:20]:
                if item['type'] == 'DINOv2_in_ConvNeXt':
                    report.append(f"- DINOv2 cluster {item['dinov2_cluster']} (size: {item['dinov2_size']}) → "
                                 f"ConvNeXt cluster {item['convnext_cluster']}: {item['overlap_percentage']:.1f}%")
                else:
                    report.append(f"- ConvNeXt cluster {item['convnext_cluster']} (size: {item['convnext_size']}) → "
                                 f"DINOv2 cluster {item['dinov2_cluster']}: {item['overlap_percentage']:.1f}%")
        else:
            report.append("No near-complete overlaps found.")
        
        # Calculate variation of information
        report.append("\n## Cross-Model Metrics\n")
        vi = self.calculate_variation_of_information()
        report.append(f"- Variation of Information (VI): {vi:.3f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def calculate_variation_of_information(self):
        """Calculate Variation of Information between two clusterings"""
        # Calculate entropies
        _, counts1 = np.unique(self.dinov2_labels, return_counts=True)
        _, counts2 = np.unique(self.convnext_labels, return_counts=True)
        
        h1 = entropy(counts1 / counts1.sum())
        h2 = entropy(counts2 / counts2.sum())
        
        # Calculate mutual information
        mi = mutual_info_score(self.dinov2_labels, self.convnext_labels)
        
        # VI = H(X) + H(Y) - 2*MI(X,Y)
        vi = h1 + h2 - 2 * mi
        
        return vi


def main():
    """Main function to run overlap analysis"""
    # Paths to cluster files
    dinov2_path = "/work/shixu/climate_project/results/climate_3k_diverse_1fps_dinov2_vitb14_lc_average_clusters.h5"
    convnext_path = "/work/shixu/climate_project/results/climate_3k_diverse_1fps_convnextv2_average_clusters.h5"
    
    # Optional: path to key mapping file
    key_mapping_path = "/work/shixu/climate_project/graph_prep/key_mapping_climate_3k_diverse_1fps_dinov2_vitb14_lc_average.torch"
    
    # Initialize analyzer
    analyzer = ClusterOverlapAnalyzer(dinov2_path, convnext_path, key_mapping_path)
    
    # Generate report
    print("\n" + "="*60)
    report = analyzer.generate_overlap_report("cluster_overlap_report.txt")
    print(report)
    
    # Create visualizations
    print("\nCreating overlap heatmaps...")
    analyzer.visualize_overlap_heatmap(top_k=15, save_path="cluster_overlap_heatmap_full.png")
    
    # Find specific overlaps
    print("\nFinding near-complete overlaps...")
    near_complete = analyzer.find_near_complete_overlaps(threshold=80)
    
    if near_complete:
        print(f"\nFound {len(near_complete)} near-complete overlaps (≥80%):")
        
        # Separate by type
        dino_in_conv = [x for x in near_complete if x['type'] == 'DINOv2_in_ConvNeXt']
        conv_in_dino = [x for x in near_complete if x['type'] == 'ConvNeXt_in_DINOv2']
        
        if dino_in_conv:
            print(f"\nDINOv2 clusters contained in ConvNeXt clusters ({len(dino_in_conv)}):")
            for item in dino_in_conv[:5]:
                print(f"  - D{item['dinov2_cluster']} → C{item['convnext_cluster']}: "
                      f"{item['overlap_percentage']:.1f}% ({item['overlap_count']}/{item['dinov2_size']} videos)")
        
        if conv_in_dino:
            print(f"\nConvNeXt clusters contained in DINOv2 clusters ({len(conv_in_dino)}):")
            for item in conv_in_dino[:5]:
                print(f"  - C{item['convnext_cluster']} → D{item['dinov2_cluster']}: "
                      f"{item['overlap_percentage']:.1f}% ({item['overlap_count']}/{item['convnext_size']} videos)")
    

if __name__ == "__main__":
    main()