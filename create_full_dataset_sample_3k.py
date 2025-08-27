#!/usr/bin/env python3
"""
Create Full Dataset Sample
Samples ~3000 videos across all years/months with length filtering
"""

import os
import pandas as pd
import random
import cv2
from collections import defaultdict
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import multiprocessing
from functools import partial

class VideoSampler:
    def __init__(self, video_base_path="/ceph/lprasse/ClimateVisions/Videos", 
                 min_duration=5.0, target_size=3000):
        self.video_base_path = Path(video_base_path)
        self.min_duration = min_duration
        self.target_size = target_size
        self.duration_cache = {}
        
    def get_video_path(self, video_id):
        """Construct full path to video file"""
        try:
            date_part = video_id.split('_')[-1]  # YYYY-MM-DD
            year, month, day = date_part.split('-')
            
            month_names = {
                '01': '01_January', '02': '02_February', '03': '03_March',
                '04': '04_April', '05': '05_May', '06': '06_June',
                '07': '07_July', '08': '08_August', '09': '09_September',
                '10': '10_October', '11': '11_November', '12': '12_December'
            }
            
            month_dir = month_names.get(month, f"{month}_Unknown")
            video_path = self.video_base_path / year / month_dir / f"{video_id}.mp4"
            
            return video_path if video_path.exists() else None
        except:
            return None
    
    def get_video_duration(self, video_id):
        """Get video duration with caching"""
        if video_id in self.duration_cache:
            return self.duration_cache[video_id]
        
        video_path = self.get_video_path(video_id)
        if not video_path:
            self.duration_cache[video_id] = None
            return None
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                self.duration_cache[video_id] = duration
                return duration
        except Exception as e:
            self.duration_cache[video_id] = None
            return None
    
    def check_video_batch(self, video_ids):
        """Check a batch of videos for duration (for parallel processing)"""
        results = []
        for video_id in video_ids:
            duration = self.get_video_duration(video_id)
            if duration and duration >= self.min_duration:
                results.append((video_id, duration))
        return results

def load_unique_video_ids():
    """Load unique (non-duplicate) video IDs"""
    duplicates_file = "/ceph/lprasse/ClimateVisions/video_classification/Duplicates_and_HashValues/duplicates.csv"
    all_videos_file = "/ceph/lprasse/ClimateVisions/video_classification/Duplicates_and_HashValues/video_hash_values.csv"
    
    print("Loading deduplication data...")
    all_videos_df = pd.read_csv(all_videos_file)
    duplicates_df = pd.read_csv(duplicates_file)
    
    all_video_ids = set(all_videos_df['id'].tolist())
    duplicate_ids = set(duplicates_df['id'].tolist())
    unique_video_ids = all_video_ids - duplicate_ids
    
    print(f"Total videos: {len(all_video_ids):,}")
    print(f"Duplicates: {len(duplicate_ids):,}")
    print(f"Unique videos: {len(unique_video_ids):,}")
    
    return unique_video_ids

def load_duration_cache(cache_file="video_durations_cache.json"):
    """Load pre-computed durations if available"""
    if os.path.exists(cache_file):
        print(f"Loading duration cache from {cache_file}...")
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_duration_cache(duration_cache, cache_file="video_durations_cache.json"):
    """Save duration cache for future use"""
    with open(cache_file, 'w') as f:
        json.dump(duration_cache, f)
    print(f"Saved duration cache to {cache_file}")

def create_full_dataset_sample(unique_video_ids, min_duration=5.0, 
                                      target_size=3000, check_durations=True,
                                      use_parallel=True, n_workers=4):
    """
    Create a sample with duration filtering
    
    Parameters:
    - min_duration: Minimum video duration in seconds
    - target_size: Target number of videos in sample
    - check_durations: Whether to check video durations
    - use_parallel: Use parallel processing for duration checks
    - n_workers: Number of parallel workers
    """
    
    sampler = VideoSampler(min_duration=min_duration)
    
    # Load cached durations if available
    sampler.duration_cache = load_duration_cache()
    
    # Group videos by year/month with optional duration filtering
    video_groups = defaultdict(list)
    video_durations = {}
    filtered_count = 0
    no_file_count = 0
    
    if check_durations:
        print(f"\n🔍 Filtering videos with minimum duration: {min_duration}s")
        print(f"Using {'parallel' if use_parallel else 'sequential'} processing...")
        
        # Group videos by year/month first
        temp_groups = defaultdict(list)
        for video_id in unique_video_ids:
            try:
                date_part = video_id.split('_')[-1]
                year, month = date_part.split('-')[:2]
                year, month = int(year), int(month)
                temp_groups[(year, month)].append(video_id)
            except:
                continue
        
        # Process each month
        total_videos_to_check = len(unique_video_ids)
        videos_checked = 0
        
        for (year, month), month_videos in sorted(temp_groups.items()):
            print(f"\nProcessing {year}-{month:02d} ({len(month_videos)} videos)...")
            
            valid_videos = []
            
            if use_parallel and len(month_videos) > 100:
                # Split into batches for parallel processing
                batch_size = max(50, len(month_videos) // n_workers)
                batches = [month_videos[i:i+batch_size] 
                          for i in range(0, len(month_videos), batch_size)]
                
                with multiprocessing.Pool(n_workers) as pool:
                    results = pool.map(sampler.check_video_batch, batches)
                
                for batch_results in results:
                    for video_id, duration in batch_results:
                        valid_videos.append(video_id)
                        video_durations[video_id] = duration
            else:
                # sequential processing
                for video_id in tqdm(month_videos, desc=f"{year}-{month:02d}"):
                    duration = sampler.get_video_duration(video_id)
                    
                    if duration is None:
                        no_file_count += 1
                    elif duration < min_duration:
                        filtered_count += 1
                    else:
                        valid_videos.append(video_id)
                        video_durations[video_id] = duration
            
            video_groups[(year, month)] = valid_videos
            videos_checked += len(month_videos)
            
            # progress update
            print(f"  Valid videos: {len(valid_videos)}/{len(month_videos)} "
                  f"({len(valid_videos)/len(month_videos)*100:.1f}%)")
        
        # Save updated duration cache
        save_duration_cache(sampler.duration_cache)
        
        print(f"\nDuration filtering complete:")
        print(f"  Videos checked: {videos_checked:,}")
        print(f"  Videos filtered (too short): {filtered_count:,}")
        print(f"  Videos not found: {no_file_count:,}")
        print(f"  Valid videos: {sum(len(v) for v in video_groups.values()):,}")
        
    else:
        # No duration filtering
        print("\n Skipping duration filtering")
        for video_id in unique_video_ids:
            try:
                date_part = video_id.split('_')[-1]
                year, month = date_part.split('-')[:2]
                year, month = int(year), int(month)
                video_groups[(year, month)].append(video_id)
            except:
                continue
    
    # Calculate sampling strategy
    total_valid_videos = sum(len(videos) for videos in video_groups.values())
    
    if total_valid_videos < target_size:
        print(f"\n Warning: Only {total_valid_videos:,} valid videos available "
              f"(target: {target_size:,})")
        target_size = total_valid_videos
    
    print(f"\n Sampling Strategy:")
    print(f"  Total valid videos: {total_valid_videos:,}")
    print(f"  Target sample size: {target_size:,}")
    print(f"  Periods with videos: {len(video_groups)}")
    
    # Sampling parameters
    min_per_period = max(5, target_size // (len(video_groups) * 2))
    
    # Ensure minimum doesn't exceed target
    total_min_possible = sum(min(min_per_period, len(videos)) 
                           for videos in video_groups.values())
    
    if total_min_possible > target_size:
        min_per_period = max(1, target_size // len(video_groups))
    
    reserved_for_min = sum(min(min_per_period, len(videos)) 
                          for videos in video_groups.values())
    remaining_budget = target_size - reserved_for_min
    
    print(f"  Minimum per period: {min_per_period}")
    print(f"  Reserved for minimums: {reserved_for_min}")
    print(f"  Remaining budget: {remaining_budget}")
    
    # Sample from each period
    selected_videos = []
    period_samples = {}
    
    for (year, month), videos in sorted(video_groups.items()):
        if not videos:
            continue
            
        # Base allocation
        base_allocation = min(min_per_period, len(videos))
        
        # Additional allocation based on proportion
        if remaining_budget > 0 and len(videos) > min_per_period:
            proportion = len(videos) / total_valid_videos
            additional = int(remaining_budget * proportion)
            total_allocation = min(base_allocation + additional, len(videos))
        else:
            total_allocation = base_allocation
        
        # Sample videos (prefer longer videos if we have duration info)
        if video_durations and len(videos) > total_allocation:
            # Sort by duration and sample preferring longer videos
            videos_with_duration = [(v, video_durations.get(v, min_duration)) 
                                  for v in videos]
            videos_with_duration.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 80% by duration, then random sample from those
            top_videos = [v for v, d in videos_with_duration[:int(len(videos)*0.8)]]
            if len(top_videos) >= total_allocation:
                period_sample = random.sample(top_videos, total_allocation)
            else:
                period_sample = top_videos + random.sample(
                    [v for v in videos if v not in top_videos],
                    total_allocation - len(top_videos)
                )
        else:
            # random sampling
            period_sample = random.sample(videos, total_allocation) if total_allocation < len(videos) else videos
        
        selected_videos.extend(period_sample)
        period_samples[(year, month)] = period_sample
    
    # Final adjustment to exact target size
    if len(selected_videos) > target_size:
        excess = len(selected_videos) - target_size
        
        # Remove from largest samples first
        for (year, month), samples in sorted(period_samples.items(), 
                                           key=lambda x: len(x[1]), reverse=True):
            if excess <= 0:
                break
            
            if len(samples) > min_per_period:
                remove_count = min(excess, len(samples) - min_per_period)
                # Remove shortest videos if we have duration info
                if video_durations:
                    samples_with_duration = [(v, video_durations.get(v, min_duration)) 
                                           for v in samples]
                    samples_with_duration.sort(key=lambda x: x[1])
                    to_remove = [v for v, d in samples_with_duration[:remove_count]]
                else:
                    to_remove = random.sample(samples, remove_count)
                
                for video in to_remove:
                    selected_videos.remove(video)
                    period_samples[(year, month)].remove(video)
                excess -= remove_count
    
    print(f"\n Final sample: {len(selected_videos):,} videos")
    
    # Calculate duration statistics if available
    if video_durations:
        sample_durations = [video_durations.get(v, 0) for v in selected_videos 
                          if v in video_durations]
        if sample_durations:
            print(f"\n📊 Sample Duration Statistics:")
            print(f"  Mean duration: {np.mean(sample_durations):.1f}s")
            print(f"  Median duration: {np.median(sample_durations):.1f}s")
            print(f"  Min duration: {np.min(sample_durations):.1f}s")
            print(f"  Max duration: {np.max(sample_durations):.1f}s")
            print(f"  Total duration: {sum(sample_durations)/3600:.1f} hours")
    
    return selected_videos, period_samples, video_durations

def save_metadata(selected_videos, period_samples, video_durations, 
                         min_duration, output_dir="samples"):
    """Save sample information with duration data"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save video list with durations
    video_list_file = f"{output_dir}/full_dataset_sample_3k_{timestamp}.txt"
    with open(video_list_file, 'w') as f:
        f.write(f"# Full Dataset Sample - {len(selected_videos)} videos\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Minimum duration: {min_duration}s\n")
        f.write(f"# Random seed: 42\n\n")
        
        for video_id in sorted(selected_videos):
            duration = video_durations.get(video_id, 'unknown')
            f.write(f"{video_id}\t{duration}\n")
    
    print(f"\n💾 Saved video list: {video_list_file}")
    
    # Create detailed CSV
    csv_data = []
    for video_id in selected_videos:
        try:
            date_part = video_id.split('_')[-1]
            year, month, day = date_part.split('-')
            csv_data.append({
                'video_id': video_id,
                'year': int(year),
                'month': int(month),
                'day': int(day),
                'date': date_part,
                'duration_seconds': video_durations.get(video_id, -1),
                'expected_frames': int(video_durations.get(video_id, 0)) if video_id in video_durations else -1
            })
        except:
            csv_data.append({
                'video_id': video_id,
                'year': 'unknown',
                'month': 'unknown',
                'day': 'unknown',
                'date': 'unknown',
                'duration_seconds': video_durations.get(video_id, -1),
                'expected_frames': -1
            })
    
    csv_df = pd.DataFrame(csv_data)
    csv_file = f"{output_dir}/full_dataset_sample_3k_{timestamp}.csv"
    csv_df.to_csv(csv_file, index=False)
    print(f"💾 Saved CSV: {csv_file}")
    
    # metadata
    metadata = {
        'sample_info': {
            'total_videos': len(selected_videos),
            'generation_timestamp': datetime.now().isoformat(),
            'random_seed': 42,
            'target_size': 3000,
            'min_duration_seconds': min_duration,
            'duration_filtering_applied': len(video_durations) > 0
        },
        'duration_statistics': {},
        'temporal_distribution': {},
        'quality_metrics': {}
    }
    
    # Add duration statistics
    if video_durations:
        sample_durations = [video_durations[v] for v in selected_videos if v in video_durations]
        metadata['duration_statistics'] = {
            'mean_seconds': float(np.mean(sample_durations)),
            'median_seconds': float(np.median(sample_durations)),
            'std_seconds': float(np.std(sample_durations)),
            'min_seconds': float(np.min(sample_durations)),
            'max_seconds': float(np.max(sample_durations)),
            'total_hours': sum(sample_durations) / 3600,
            'duration_distribution': {
                '5-10s': len([d for d in sample_durations if 5 <= d < 10]),
                '10-30s': len([d for d in sample_durations if 10 <= d < 30]),
                '30-60s': len([d for d in sample_durations if 30 <= d < 60]),
                '>60s': len([d for d in sample_durations if d >= 60])
            }
        }
    
    # Add temporal distribution with quality metrics
    for (year, month), samples in sorted(period_samples.items()):
        key = f"{year}-{month:02d}"
        
        period_durations = [video_durations.get(v, 0) for v in samples if v in video_durations]
        
        metadata['temporal_distribution'][key] = {
            'count': len(samples),
            'mean_duration': float(np.mean(period_durations)) if period_durations else 0,
            'total_duration_hours': sum(period_durations) / 3600 if period_durations else 0,
            'sample_ids': samples[:3] + ['...'] if len(samples) > 3 else samples
        }
    
    # Quality metrics
    metadata['quality_metrics'] = {
        'expected_total_frames': sum(int(d) for d in video_durations.values() if d) if video_durations else 'unknown',
        'videos_per_period_std': float(np.std([len(v) for v in period_samples.values()])),
        'temporal_coverage': f"{len(period_samples)} months",
        'average_videos_per_month': len(selected_videos) / len(period_samples)
    }
    
    metadata_file = f"{output_dir}/full_dataset_sample_3k_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata_file}")
    
    return video_list_file, metadata_file, csv_file

def main():
    """Create sample"""
   
    
    MIN_DURATION = 5.0  # Minimum video duration in seconds
    TARGET_SIZE = 3000
    CHECK_DURATIONS = True  # duration check
    USE_PARALLEL = True  # parallel processing for duration checks
    N_WORKERS = 4  # number of parallel workers
    

    random.seed(42)
    
    # Load unique videos

    unique_video_ids = load_unique_video_ids()
    
    # Create sample with duration filtering
    selected_videos, period_samples, video_durations = create_full_dataset_sample(
        unique_video_ids,
        min_duration=MIN_DURATION,
        target_size=TARGET_SIZE,
        check_durations=CHECK_DURATIONS,
        use_parallel=USE_PARALLEL,
        n_workers=N_WORKERS
    )
    
    # Save sample information

    video_list_file, metadata_file, csv_file = save_metadata(
        selected_videos,
        period_samples,
        video_durations,
        MIN_DURATION
    )
    
   

if __name__ == "__main__":
    main()