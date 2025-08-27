#!/usr/bin/env python3
"""
Density-Based Frame Extraction

This script implements two extraction methods:
- Static: Uniformly spaced frame extraction
- Diverse: Visually diverse frame selection using clustering


"""

import os
import cv2
import argparse
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from collections import defaultdict
import hashlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Frame extraction with density-based sampling.
    
    This class handles the core frame extraction logic, supporting both
    uniform and diversity-based frame selection strategies.
    """
    
    def __init__(
        self, 
        min_quality_std: float = 10.0, 
        resize_dim: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the frame extractor.
        
        Args:
            min_quality_std: Minimum standard deviation for frame quality validation
            resize_dim: Target dimensions for frame resizing during processing
        """
        self.min_quality_std = min_quality_std
        self.resize_dim = resize_dim
        
        # Month mapping for directory structure
        self.month_names = {
            '01': '01_January', '02': '02_February', '03': '03_March',
            '04': '04_April', '05': '05_May', '06': '06_June',
            '07': '07_July', '08': '08_August', '09': '09_September',
            '10': '10_October', '11': '11_November', '12': '12_December'
        }
        
        # Extraction parameters
        self.default_fps_density = 1  # 1 frame per second
        self.min_frames = 4           # Minimum frames per video
        self.max_frames = 60          # Maximum frames per video
        
        logger.info(f"FrameExtractor initialized with min_quality_std={min_quality_std}")
    
    def calculate_density_based_frame_count(
        self, 
        video_duration_seconds: float, 
        fps_density: Optional[float] = None
    ) -> int:
        """
        Calculate number of frames to extract based on temporal density.
        
        Args:
            video_duration_seconds: Duration of video in seconds
            fps_density: Frames per second to extract (uses default if None)
            
        Returns:
            Number of frames to extract, bounded by min/max constraints
        """
        if fps_density is None:
            fps_density = self.default_fps_density
        
        # Calculate target frames
        target_frames = int(video_duration_seconds * fps_density)
        
        # Apply bounds
        target_frames = max(self.min_frames, min(self.max_frames, target_frames))
        
        return target_frames
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get video metadata including duration and frame count.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata or None if video cannot be opened
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }
    
    def is_valid_frame(self, frame: np.ndarray) -> bool:
        """
        Validate frame quality by checking for black/white frames and sufficient variance.
        
        Args:
            frame: Frame to validate
            
        Returns:
            True if frame passes quality checks
        """
        if frame is None or frame.size == 0:
            return False
        
        # Convert to grayscale and check standard deviation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std = gray.std()
        
        # Check for completely black or white frames
        mean = gray.mean()
        if mean < 5 or mean > 250:  # Nearly black or white
            return False
            
        return std > self.min_quality_std
    
    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """
        Calculate perceptual hash of frame for duplicate detection.
        
        Args:
            frame: Frame to hash
            
        Returns:
            Binary string hash representation
        """
        # Resize to small size for hashing
        small = cv2.resize(frame, (8, 8))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Calculate average
        avg = gray.mean()
        
        # Create binary hash
        hash_bits = (gray > avg).flatten()
        hash_str = ''.join(['1' if b else '0' for b in hash_bits])
        
        return hash_str
    
    def extract_static_frames(
        self, 
        video_path: str, 
        output_dir: str, 
        num_frames: Optional[int] = None, 
        fps_density: Optional[float] = None
    ) -> Dict:
        """
        Extract uniformly spaced frames from video.
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output frames
            num_frames: Number of frames to extract (overrides density calculation)
            fps_density: Frames per second to extract
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        # Get video info
        video_info = self.get_video_info(video_path)
        if not video_info:
            logger.error(f"Cannot get video info: {video_path}")
            return {'success': False, 'frames': 0}
        
        # Determine number of frames 
        if num_frames is None:
            num_frames = self.calculate_density_based_frame_count(
                video_info['duration'], 
                fps_density
            )
            actual_density = num_frames / video_info['duration'] if video_info['duration'] > 0 else 0
            logger.info(
                f"Video duration: {video_info['duration']:.1f}s, "
                f"extracting {num_frames} frames "
                f"(density: {actual_density:.2f} fps)"
            )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return {'success': False, 'frames': 0}
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count == 0:
            cap.release()
            return {'success': False, 'frames': 0}
        
        # Calculate uniform spacing
        frame_indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)
        
        saved_frames = []
        frame_metadata = []
        
        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            
            if ret and self.is_valid_frame(frame):
                frame_path = os.path.join(output_dir, f"frame_{len(saved_frames):04d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                saved_frames.append(frame_path)
                frame_metadata.append({
                    'index': int(target_idx),
                    'timestamp': target_idx / fps if fps > 0 else 0,
                    'hash': self.compute_frame_hash(frame)
                })
        
        cap.release()
        
        return {
            'success': True,
            'frames': len(saved_frames),
            'strategy': 'static',
            'sampling_method': 'density-based',
            'video_duration': video_info['duration'],
            'target_frames': num_frames,
            'actual_frames': len(saved_frames),
            'extraction_density': len(saved_frames) / video_info['duration'] if video_info['duration'] > 0 else 0,
            'configured_density': fps_density or self.default_fps_density,
            'frame_indices': [m['index'] for m in frame_metadata],
            'metadata': frame_metadata
        }
    
    def extract_diverse_frames(
        self, 
        video_path: str, 
        output_dir: str, 
        num_frames: Optional[int] = None, 
        fps_density: Optional[float] = None,
        sample_rate: int = 10
    ) -> Dict:
        """
        Extract visually diverse frames using clustering.
        
        Args:
            video_path: Path to input video
            output_dir: Directory for output frames
            num_frames: Number of frames to extract
            fps_density: Frames per second to extract
            sample_rate: Sample every Nth frame for analysis
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        # Get video info
        video_info = self.get_video_info(video_path)
        if not video_info:
            return {'success': False, 'frames': 0}
        
        # Determine number of frames using density-based approach
        if num_frames is None:
            num_frames = self.calculate_density_based_frame_count(
                video_info['duration'], 
                fps_density
            )
            actual_density = num_frames / video_info['duration'] if video_info['duration'] > 0 else 0
            logger.info(
                f"Video duration: {video_info['duration']:.1f}s, "
                f"targeting {num_frames} diverse frames "
                f"(density: {actual_density:.2f} fps)"
            )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'frames': 0}
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Adjust sample rate for sufficient coverage
        min_samples_needed = max(num_frames * 3, int(video_info['duration'] * 2))
        if frame_count < min_samples_needed * sample_rate:
            sample_rate = max(1, frame_count // min_samples_needed)
            logger.debug(f"Adjusted sample_rate to {sample_rate} for better coverage")
        
        # Sample frames at regular intervals
        sample_indices = list(range(0, frame_count, sample_rate))
        
        frames_data = []
        features = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret and self.is_valid_frame(frame):
                # Extract features for diversity analysis
                features_vec = self._extract_frame_features(frame)
                features.append(features_vec)
                frames_data.append((idx, frame))
        
        cap.release()
        
        if len(frames_data) <= num_frames:
            # If fewer frames than requested, save all
            return self._save_all_frames(frames_data, output_dir, fps, video_info, num_frames, fps_density)
        
        # Use K-means clustering to find diverse frames
        selected_indices = self._select_diverse_frames(features, num_frames)
        
        # Save selected frames
        saved_frames = []
        frame_metadata = []
        
        for i, idx in enumerate(selected_indices):
            frame_idx, frame = frames_data[idx]
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
            frame_metadata.append({
                'index': frame_idx,
                'timestamp': frame_idx / fps if fps > 0 else 0,
                'hash': self.compute_frame_hash(frame)
            })
        
        return {
            'success': True,
            'frames': len(saved_frames),
            'strategy': 'diverse',
            'sampling_method': 'density-based',
            'video_duration': video_info['duration'],
            'target_frames': num_frames,
            'actual_frames': len(saved_frames),
            'extraction_density': len(saved_frames) / video_info['duration'] if video_info['duration'] > 0 else 0,
            'configured_density': fps_density or self.default_fps_density,
            'frame_indices': [m['index'] for m in frame_metadata],
            'metadata': frame_metadata
        }
    
    def _extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract visual features from frame for diversity analysis.
        
        Args:
            frame: Input frame
            
        Returns:
            Feature vector
        """
        # Resize for efficiency
        small_frame = cv2.resize(frame, self.resize_dim)
        
        # 1. Color histogram features
        hist_b = cv2.calcHist([small_frame], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([small_frame], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([small_frame], [2], None, [32], [0, 256])
        color_feat = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        color_feat = color_feat / (color_feat.sum() + 1e-6)
        
        # 2. Edge density
        edges = cv2.Canny(cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        # 3. Texture features (variance in blocks)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        block_size = 16
        texture_feat = []
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                texture_feat.append(block.var())
        
        texture_feat = np.array(texture_feat) / (np.max(texture_feat) + 1e-6)
        
        # Combine features with weights
        combined_feat = np.concatenate([
            color_feat * 0.5,              # Color features
            np.array([edge_density]) * 0.3, # Edge density
            texture_feat * 0.2              # Texture
        ])
        
        return combined_feat
    
    def _select_diverse_frames(self, features: List[np.ndarray], num_frames: int) -> List[int]:
        """
        Select diverse frames using K-means clustering.
        
        Args:
            features: List of feature vectors
            num_frames: Number of frames to select
            
        Returns:
            Indices of selected frames
        """
        features_array = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)
        
        # Cluster to find diverse groups
        kmeans = KMeans(n_clusters=num_frames, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_normalized)
        
        # Select one frame from each cluster (closest to centroid)
        selected_indices = []
        for cluster_id in range(num_frames):
            cluster_mask = clusters == cluster_id
            cluster_features = features_normalized[cluster_mask]
            cluster_frame_indices = np.where(cluster_mask)[0]
            
            if len(cluster_features) > 0:
                # Find frame closest to cluster centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                best_idx = cluster_frame_indices[np.argmin(distances)]
                selected_indices.append(best_idx)
        
        # Sort by temporal order
        selected_indices.sort()
        
        return selected_indices
    
    def _save_all_frames(
        self, 
        frames_data: List[Tuple[int, np.ndarray]], 
        output_dir: str, 
        fps: float,
        video_info: Dict,
        num_frames: int,
        fps_density: Optional[float]
    ) -> Dict:
        """
        Save all frames when total is less than requested.
        
        Args:
            frames_data: List of (index, frame) tuples
            output_dir: Output directory
            fps: Video frames per second
            video_info: Video metadata
            num_frames: Target number of frames
            fps_density: Configured density
            
        Returns:
            Extraction results dictionary
        """
        saved_frames = []
        frame_metadata = []
        
        for i, (idx, frame) in enumerate(frames_data):
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
            frame_metadata.append({
                'index': idx,
                'timestamp': idx / fps if fps > 0 else 0,
                'hash': self.compute_frame_hash(frame)
            })
        
        return {
            'success': True,
            'frames': len(saved_frames),
            'strategy': 'diverse',
            'sampling_method': 'density-based',
            'video_duration': video_info['duration'],
            'target_frames': num_frames,
            'actual_frames': len(saved_frames),
            'extraction_density': len(saved_frames) / video_info['duration'] if video_info['duration'] > 0 else 0,
            'configured_density': fps_density or self.default_fps_density,
            'frame_indices': [m['index'] for m in frame_metadata],
            'metadata': frame_metadata
        }
    
    def extract_frames(
        self, 
        video_path: str, 
        output_dir: str, 
        strategy: str = 'static',
        num_frames: Optional[int] = None, 
        fps_density: Optional[float] = None, 
        **kwargs
    ) -> Dict:
        """
        Main extraction method.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            strategy: Extraction strategy ('static' or 'diverse')
            num_frames: Number of frames to extract (overrides density)
            fps_density: Frames per second to extract
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Dictionary containing extraction results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Select strategy
        if strategy == 'static':
            result = self.extract_static_frames(video_path, output_dir, num_frames, fps_density)
        elif strategy == 'diverse':
            sample_rate = kwargs.get('sample_rate', 10)
            result = self.extract_diverse_frames(video_path, output_dir, num_frames, fps_density, sample_rate)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose 'static' or 'diverse'")
        
        # Add metadata and save
        if result['success']:
            result['video_path'] = video_path
            result['output_dir'] = output_dir
            
            # Save metadata
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(result, f, indent=2)
        
        return result


class OrganizedExtractor:
    """
    Manages organized directory structure for frame extraction.
    
    This class handles the organization of extracted frames into a structured
    directory hierarchy and maintains dataset-level statistics.
    """
    
    def __init__(
        self, 
        base_output_dir: str, 
        dataset_name: str, 
        fps_density: float = 1, 
        min_quality: float = 10.0
    ):
        """
        Initialize the organized extractor.
        
        Args:
            base_output_dir: Base directory for output
            dataset_name: Name of the dataset
            fps_density: Default frames per second to extract
            min_quality: Minimum quality threshold
        """
        self.base_output_dir = Path(base_output_dir)
        self.dataset_name = dataset_name
        self.fps_density = fps_density
        self.extractor = FrameExtractor(min_quality_std=min_quality)
        
        # Create base directory structure
        self.dataset_dir = self.base_output_dir / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategy directories will be created as needed
        self.strategy_dirs = {}
        
        # Track extraction statistics
        self.extraction_stats = defaultdict(list)
        
        logger.info(f"OrganizedExtractor initialized with min_quality={min_quality}")
    
    def get_strategy_dir(self, strategy: str) -> Path:
        """
        Get or create strategy directory.
        
        Args:
            strategy: Extraction strategy name
            
        Returns:
            Path to strategy directory
        """
        if strategy not in self.strategy_dirs:
            strategy_dir = self.dataset_dir / f"frames_{strategy}"
            strategy_dir.mkdir(exist_ok=True)
            self.strategy_dirs[strategy] = strategy_dir
            
            # Create info file for strategy
            info_path = strategy_dir / "strategy_info.json"
            if not info_path.exists():
                info = {
                    'strategy': strategy,
                    'description': self._get_strategy_description(strategy),
                    'sampling_method': 'density-based',
                    'fps_density': self.fps_density,
                    'created': datetime.now().isoformat()
                }
                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)
        
        return self.strategy_dirs[strategy]
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get description for each strategy."""
        descriptions = {
            'static': f'Uniformly spaced frames at {self.fps_density} fps density',
            'diverse': f'Visually diverse frames selected at {self.fps_density} fps density'
        }
        return descriptions.get(strategy, 'Unknown strategy')
    
    def process_video(
        self, 
        video_id: str, 
        video_path: str, 
        strategy: str, 
        num_frames: Optional[int] = None, 
        fps_density: Optional[float] = None, 
        **kwargs
    ) -> Dict:
        """
        Process single video with organized output.
        
        Args:
            video_id: Unique video identifier
            video_path: Path to video file
            strategy: Extraction strategy
            num_frames: Number of frames to extract
            fps_density: Frames per second to extract
            **kwargs: Additional strategy parameters
            
        Returns:
            Extraction results dictionary
        """
        # Get strategy directory
        strategy_dir = self.get_strategy_dir(strategy)
        
        # Create video output directory
        video_output_dir = strategy_dir / video_id
        
        # Use configured density if not specified
        if fps_density is None:
            fps_density = self.fps_density
        
        # Extract frames
        result = self.extractor.extract_frames(
            video_path, str(video_output_dir), strategy, num_frames, fps_density, **kwargs
        )
        
        # Track statistics
        if result['success']:
            self.extraction_stats['durations'].append(result.get('video_duration', 0))
            self.extraction_stats['frame_counts'].append(result.get('actual_frames', 0))
            self.extraction_stats['target_frames'].append(result.get('target_frames', 0))
            self.extraction_stats['extraction_densities'].append(result.get('extraction_density', 0))
        
        return result
    
    def create_summary_report(self, stats: Dict) -> None:
        """
        Create summary report with extraction statistics.
        
        Args:
            stats: Statistics dictionary to save
        """
        # Dataset summary directory
        summary_dir = self.dataset_dir / "summaries"
        summary_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate density-based statistics
        if self.extraction_stats['durations']:
            density_stats = {
                'mean_duration': float(np.mean(self.extraction_stats['durations'])),
                'median_duration': float(np.median(self.extraction_stats['durations'])),
                'min_duration': float(np.min(self.extraction_stats['durations'])),
                'max_duration': float(np.max(self.extraction_stats['durations'])),
                'mean_frames_extracted': float(np.mean(self.extraction_stats['frame_counts'])),
                'median_frames_extracted': float(np.median(self.extraction_stats['frame_counts'])),
                'mean_extraction_density': float(np.mean(self.extraction_stats['extraction_densities'])),
                'configured_density': self.fps_density
            }
            stats['density_sampling_stats'] = density_stats
        
        # Save detailed stats
        stats_path = summary_dir / f"extraction_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


def process_video_batch_organized(
    video_list_path: str, 
    base_output_dir: str, 
    dataset_name: str, 
    strategy: str = 'static',
    num_frames: Optional[int] = None, 
    fps_density: float = 1, 
    min_quality: float = 10.0, 
    **kwargs
) -> None:
    """
    Process batch of videos with organized extraction.
    
    Args:
        video_list_path: Path to file containing video IDs
        base_output_dir: Base output directory
        dataset_name: Name of dataset
        strategy: Extraction strategy
        num_frames: Fixed number of frames to extract
        fps_density: Frames per second to extract
        min_quality: Minimum quality threshold
        **kwargs: Additional parameters
    """
    base_video_dir = "/ceph/lprasse/ClimateVisions/Videos"
    
    # Initialize extractor
    org_extractor = OrganizedExtractor(base_output_dir, dataset_name, fps_density, min_quality)
    
    # Read video IDs
    video_ids = []
    with open(video_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                video_ids.append(line.split('\t')[0])
    
    # Print configuration
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing Climate Videos - {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total videos: {len(video_ids)}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Sampling method: Density-based")
    logger.info(f"Target density: {fps_density} fps (1 frame every {1/fps_density:.1f} seconds)")
    logger.info(f"Quality threshold: {min_quality}")
    if num_frames:
        logger.info(f"Fixed frame override: {num_frames} frames")
    logger.info(f"Output structure: {base_output_dir}/{dataset_name}/frames_{strategy}/")
    logger.info(f"{'='*60}\n")
    
    # Initialize statistics
    stats = {
        'dataset': dataset_name,
        'total': len(video_ids),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'start_time': datetime.now().isoformat(),
        'strategy': strategy,
        'sampling_method': 'density-based',
        'fps_density': fps_density,
        'min_quality': min_quality,
        'fixed_frame_count': num_frames,
        'parameters': kwargs
    }
    
    failed_videos = []
    
    # Process each video
    for i, video_id in enumerate(video_ids):
        try:
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {i+1}/{len(video_ids)} "
                    f"(Success: {stats['success']}, Failed: {stats['failed']}, "
                    f"Skipped: {stats['skipped']})"
                )
            
            # Parse video path
            date = video_id.split('_')[-1]
            year, month = date.split('-')[:2]
            month_folder = org_extractor.extractor.month_names.get(month, month)
            video_path = f"{base_video_dir}/{year}/{month_folder}/{video_id}.mp4"
            
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_id}")
                stats['failed'] += 1
                failed_videos.append({'id': video_id, 'reason': 'not_found'})
                continue
            
            # Check if already processed
            strategy_dir = org_extractor.get_strategy_dir(strategy)
            video_output_dir = strategy_dir / video_id
            metadata_path = video_output_dir / 'metadata.json'
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        existing_meta = json.load(f)
                    # Skip if already processed with density-based sampling
                    if existing_meta.get('sampling_method') == 'density-based':
                        stats['skipped'] += 1
                        continue
                except:
                    pass
            
            # Process video
            result = org_extractor.process_video(
                video_id, video_path, strategy, num_frames, fps_density, **kwargs
            )
            
            if result['success']:
                stats['success'] += 1
            else:
                stats['failed'] += 1
                failed_videos.append({'id': video_id, 'reason': 'extraction_failed'})
                
        except Exception as e:
            logger.error(f"Error processing {video_id}: {e}")
            stats['failed'] += 1
            failed_videos.append({'id': video_id, 'reason': str(e)})
    
    # Final statistics
    stats['end_time'] = datetime.now().isoformat()
    stats['failed_videos'] = failed_videos
    
    # Create summary report
    org_extractor.create_summary_report(stats)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction Complete - {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Density: {fps_density} fps")
    logger.info(f"Quality threshold: {min_quality}")
    logger.info(f"Total: {stats['total']}")
    logger.info(f"Success: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    logger.info(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    logger.info(f"Skipped: {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    logger.info(f"\nOutput directory: {org_extractor.dataset_dir}")
    logger.info(f"{'='*60}\n")


def main():
    """Main entry point for the frame extraction tool."""
    parser = argparse.ArgumentParser(
        description='Density-based frame extraction for climate video analysis'
    )
    parser.add_argument(
        '--video_list', 
        type=str, 
        required=True, 
        help='Path to video ID list file (.txt)'
    )
    parser.add_argument(
        '--output_root', 
        type=str, 
        required=True,
        help='Base output directory'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Dataset name (e.g., climate_3k_density)'
    )
    parser.add_argument(
        '--strategy', 
        type=str, 
        choices=['static', 'diverse'],
        default='static',
        help='Frame selection strategy: static (uniform) or diverse (visual diversity)'
    )
    parser.add_argument(
        '--fps_density', 
        type=float, 
        default=1,
        help='Frames per second to extract (default: 1)'
    )
    parser.add_argument(
        '--num_frames', 
        type=int, 
        default=None,
        help='Fixed number of frames (overrides density-based calculation)'
    )
    parser.add_argument(
        '--sample_rate', 
        type=int, 
        default=10,
        help='Sample rate for diverse strategy (every Nth frame)'
    )
    parser.add_argument(
        '--min_quality', 
        type=float, 
        default=10.0,
        help='Minimum standard deviation for frame quality validation'
    )
    
    args = parser.parse_args()
    
    # Process videos with density-based extraction
    process_video_batch_organized(
        args.video_list,
        args.output_root,
        args.dataset,
        args.strategy,
        args.num_frames,
        args.fps_density,
        args.min_quality, 
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()