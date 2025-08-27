#!/usr/bin/env python3
"""
Extracts representative frames from clusters and creates visualizations

"""

import os
import numpy as np
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import h5py


class VideoFrameVisualizer:
    def __init__(self, project_root="/work/shixu/climate_project", 
                 dataset="climate_3k_static", model="convnextv2", method="average",
                 frames_dir_name="frames_static"):
        """
        Initialize the video frame visualizer.
        
        Args:
            project_root: Root directory of the project
            dataset: Dataset name
            model: Model name used for embeddings
            method: Combination method used
            frames_dir_name: Directory name containing extracted frames
        """
        self.project_root = Path(project_root)
        self.dataset = dataset
        self.model = model
        self.method = method
        
        # Set up directories
        self.analysis_dir = self.project_root / "analysis" / f"{dataset}_{model}_{method}"
        self.inspection_dir = self.analysis_dir / "cluster_inspection"
        self.frames_output_dir = self.inspection_dir / "frame_grids"
        self.results_dir = self.project_root / "results"
        self.graph_prep_dir = self.project_root / "MP4VisualFrameDetection" / "graph_prep" / "results"
        self.frames_source_dir = self.project_root / "data" / dataset / frames_dir_name
        
        # Create output directory
        self.frames_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Video Frame Visualizer initialized")
        print(f"Dataset: {dataset}")
        print(f"Model: {model}")
        print(f"Method: {method}")
        print(f"Output directory: {self.frames_output_dir}")
    
    def load_cluster_data(self):
        """Load clustering results and create video-to-cluster mappings."""
        print("\nLoading cluster data...")
        
        # Load clustering results
        cluster_file = self.results_dir / f"{self.dataset}_{self.model}_{self.method}_clusters.h5"
        if not cluster_file.exists():
            print(f"Cluster file not found: {cluster_file}")
            return None, None
        
        with h5py.File(cluster_file, 'r') as f:
            labels = f['labels'][:]
        
        # Load key mapping
        key_mapping_file = self.graph_prep_dir / f"key_mapping_{self.dataset}_{self.model}_{self.method}.torch"
        if not key_mapping_file.exists():
            print(f"Key mapping not found: {key_mapping_file}")
            return None, None
        
        key_mapping = torch.load(key_mapping_file, weights_only=True)
        reverse_mapping = {v: k for k, v in key_mapping.items()}
        
        # Create cluster to videos mapping
        cluster_to_videos = {}
        for node_id, cluster_id in enumerate(labels):
            cluster_id = int(cluster_id)
            if node_id in reverse_mapping:
                video_id = reverse_mapping[node_id]
                if cluster_id not in cluster_to_videos:
                    cluster_to_videos[cluster_id] = []
                cluster_to_videos[cluster_id].append(video_id)
        
        # Sort clusters by size
        cluster_sizes = [(cid, len(videos)) for cid, videos in cluster_to_videos.items()]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Loaded {len(cluster_to_videos)} clusters")
        print(f"Largest cluster contains {cluster_sizes[0][1]} videos")
        
        return cluster_to_videos, cluster_sizes
    
    def get_video_frames_path(self, video_id):
        """Get path to extracted frames for a video."""
        video_frames_path = self.frames_source_dir / video_id
        
        if video_frames_path.exists():
            return video_frames_path
        
        # Try with _frames suffix
        video_frames_path = self.frames_source_dir / f"{video_id}_frames"
        if video_frames_path.exists():
            return video_frames_path
            
        return None
    
    def load_frames_from_video(self, video_frames_path, num_frames=6):
        """
        Load frames from a video directory.
        
        Args:
            video_frames_path: Path to directory containing video frames
            num_frames: Number of frames to sample
            
        Returns:
            Dictionary containing frames and metadata, or None if loading fails
        """
        if not video_frames_path or not video_frames_path.exists():
            return None
        
        # Get all frame files
        frame_files = sorted([f for f in video_frames_path.iterdir() 
                            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not frame_files:
            return None
        
        # Select frames uniformly
        if len(frame_files) <= num_frames:
            selected_files = frame_files
        else:
            # Select uniformly distributed frames
            indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
            selected_files = [frame_files[i] for i in indices]
        
        frames = []
        frame_indices = []
        
        for i, frame_file in enumerate(selected_files):
            try:
                # Load frame
                frame = Image.open(frame_file)
                frame_rgb = np.array(frame.convert('RGB'))
                frames.append(frame_rgb)
                frame_indices.append(i)
            except Exception as e:
                print(f"Error loading frame {frame_file}: {e}")
                continue
        
        if not frames:
            return None
        
        return {
            'frames': frames,
            'frame_indices': frame_indices,
            'total_frames': len(frame_files),
            'video_path': str(video_frames_path)
        }
    
    def create_frame_grid(self, frames_data_list, cluster_id, cluster_size, max_videos=8, frames_per_video=4):
        """
        Create a grid visualization showing representative frames from multiple videos in a cluster.
        
        Args:
            frames_data_list: List of frame data dictionaries
            cluster_id: ID of the cluster
            cluster_size: Total number of videos in the cluster
            max_videos: Maximum number of videos to display
            frames_per_video: Number of frames to show per video
            
        Returns:
            Matplotlib figure object or None
        """
        if not frames_data_list:
            return None
        
        # Limit number of videos to display
        display_videos = frames_data_list[:max_videos]
        
        # Calculate figure size
        fig_width = frames_per_video * 3
        fig_height = len(display_videos) * 2.5 + 1
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Add title
        fig.suptitle(f'Cluster {cluster_id} - {cluster_size} videos\n'
                    f'Showing {len(display_videos)} sample videos with {frames_per_video} frames each', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = gridspec.GridSpec(len(display_videos), frames_per_video, 
                              figure=fig, hspace=0.3, wspace=0.1,
                              top=0.95, bottom=0.02, left=0.02, right=0.98)
        
        for video_idx, frames_data in enumerate(display_videos):
            video_id = Path(frames_data['video_path']).name
            frames = frames_data['frames'][:frames_per_video]
            
            for frame_idx, frame in enumerate(frames):
                ax = fig.add_subplot(gs[video_idx, frame_idx])
                
                # Display frame
                ax.imshow(frame)
                ax.axis('off')
                
                # Add video ID to first frame
                if frame_idx == 0:
                    # Extract date from video ID if present
                    try:
                        date_part = video_id.split('_')[-1]
                        display_text = f"{video_id[:30]}...\n{date_part}" if len(video_id) > 30 else f"{video_id}\n{date_part}"
                    except:
                        display_text = video_id[:40] + '...' if len(video_id) > 40 else video_id
                    
                    ax.text(0.5, -0.1, display_text, transform=ax.transAxes,
                           ha='center', va='top', fontsize=8, wrap=True)
        
        plt.tight_layout()
        return fig
    
    def sample_and_visualize_cluster(self, cluster_id, video_ids, cluster_size, max_videos=10):
        """
        Sample videos from a cluster and create visualization.
        
        Args:
            cluster_id: ID of the cluster
            video_ids: List of video IDs in the cluster
            cluster_size: Total size of the cluster
            max_videos: Maximum number of videos to sample
            
        Returns:
            Number of successfully processed videos
        """
        print(f"\nProcessing cluster {cluster_id}")
        print(f"Cluster contains {cluster_size} videos")
        print(f"Sampling {min(len(video_ids), max_videos)} videos...")
        
        # Random sampling
        if len(video_ids) > max_videos:
            sampled_videos = random.sample(video_ids, max_videos)
        else:
            sampled_videos = video_ids
        
        frames_data_list = []
        
        # Load frames for sampled videos
        for video_id in sampled_videos:
            video_frames_path = self.get_video_frames_path(video_id)
            
            if video_frames_path:
                frames_data = self.load_frames_from_video(video_frames_path, num_frames=6)
                
                if frames_data and frames_data['frames']:
                    frames_data_list.append(frames_data)
        
        print(f"Successfully loaded frames from {len(frames_data_list)} videos")
        
        # Create and save grid visualization
        if frames_data_list:
            grid_fig = self.create_frame_grid(frames_data_list, cluster_id, cluster_size,
                                            max_videos=8, frames_per_video=4)
            
            if grid_fig:
                output_path = self.frames_output_dir / f"cluster_{cluster_id}_grid.png"
                grid_fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(grid_fig)
                print(f"Saved grid visualization: {output_path}")
        
        return len(frames_data_list)
    
    def process_clusters(self, max_clusters=10, max_videos_per_cluster=10):
        """
        Process top clusters to create frame grid visualizations.
        
        Args:
            max_clusters: Maximum number of clusters to process
            max_videos_per_cluster: Maximum videos to sample per cluster
        """
        # Load cluster data
        cluster_to_videos, cluster_sizes = self.load_cluster_data()
        
        if not cluster_to_videos:
            print("Failed to load cluster data")
            return
        
        print(f"\nProcessing top {max_clusters} clusters...")
        
        # Process top clusters
        for i, (cluster_id, cluster_size) in enumerate(cluster_sizes[:max_clusters]):
            print(f"\n{'='*50}")
            print(f"Processing cluster {i+1}/{max_clusters}")
            
            video_ids = cluster_to_videos[cluster_id]
            self.sample_and_visualize_cluster(cluster_id, video_ids, cluster_size, max_videos_per_cluster)
        
        print(f"\nVisualization complete. Output saved to: {self.frames_output_dir}")


def main():
    """Main function to run video frame sampling and visualization."""
    import argparse

    parser = argparse.ArgumentParser(description='Sample and visualize frames from video clusters')
    parser.add_argument('--dataset', type=str, default='climate_3k_static',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='convnextv2',
                        help='Model name')
    parser.add_argument('--method', type=str, default='average',
                        help='Combination method')
    parser.add_argument('--max_clusters', type=int, default=10,
                        help='Maximum number of clusters to process')
    parser.add_argument('--max_videos', type=int, default=10,
                        help='Maximum videos per cluster to sample')
    parser.add_argument('--frames_dir', type=str, default='frames_static',
                        help='Directory name containing extracted frames')

    args = parser.parse_args()

    print(f"Starting video frame visualization...")
    print(f"Configuration: {args.dataset}_{args.model}_{args.method}")

    visualizer = VideoFrameVisualizer(
        dataset=args.dataset,
        model=args.model,
        method=args.method,
        frames_dir_name=args.frames_dir
    )

    # Process clusters and create visualizations
    visualizer.process_clusters(
        max_clusters=args.max_clusters,
        max_videos_per_cluster=args.max_videos
    )


if __name__ == "__main__":
    main()