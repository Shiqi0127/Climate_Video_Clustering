#!/usr/bin/env python3
"""
DINOv2 embedding generation for video frames
Processes frames organized by video and creates video-level embeddings
Includes average, weighted_diversity, and temporal_coherence combination methods


Acknowledgments:
- Adapted from Prasse et al. (2025)'s DINOv2 implementation (DINOv2.py)
  Original script: Single image embedding generation for dataset processing
  This adaptation: Extended for video frame processing with multiple combination methods

"""

## Dataset paths - modify to match your setup
frames_root = "/work/shixu/climate_project/data/climate_3k_static/frames_static"
dataset = "climate_3k_static"
model_name = "dinov2_vitb14_lc"
# Base path without extension - will add method name and .torch extension
output_base = f"/work/shixu/climate_project/features/{dataset}_dinov2"

############################################################################################################

# load packages
import torch
from PIL import Image, ImageFile
import tqdm
import os
import torch.nn as nn
from dinov2.data.transforms import make_classification_eval_transform
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

# allow to load images that exceed max size
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
def setup_logging(dataset_name, combination_method):
    """Set up logging with both console and file handlers"""
    # Create logs directory
    log_dir = Path("/work/shixu/climate_project/logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    general_log = log_dir / f"{dataset_name}_dinov2_{combination_method}_{timestamp}.log"
    error_log = log_dir / f"{dataset_name}_dinov2_{combination_method}_{timestamp}_errors.log"
    
    logger = logging.getLogger('dinov2_video_embedding')
    logger.setLevel(logging.DEBUG)

    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    file_handler = logging.FileHandler(general_log)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    error_handler = logging.FileHandler(error_log)
    error_handler.setLevel(logging.ERROR)
    error_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s\n%(exc_info)s')
    error_handler.setFormatter(error_format)
    

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger, str(general_log), str(error_log)

# Initialize logger (will be properly configured in main)
logger = logging.getLogger('dinov2_video_embedding')

# Load DINOv2 
try:
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading DINOv2 model: {model_name}")
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    

    model.linear_head = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    transform = make_classification_eval_transform()
    
    logger.info(f"Model loaded successfully: {model_name}")
    logger.info(f"Using device: {device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    raise



def compute_frame_diversity(embeddings):
    """
    Diversity calculate using standard deviation of similarities
    Frames that are uniquely different will get higher weights
    """
    try:
        if len(embeddings) == 1:
            return torch.ones(1)
        
        # Edge case: very few frames
        if len(embeddings) < 3:
            logger.warning(f"Only {len(embeddings)} frames available for diversity weighting, using uniform weights")
            return torch.ones(len(embeddings)) / len(embeddings)
        

        embeddings_np = embeddings.cpu().numpy()
        
        # pairwise similarities
        similarities = cosine_similarity(embeddings_np)
        
        # For each frame, compute its uniqueness score
        diversity_scores = []
        for i in range(len(embeddings)):
            # Get similarities to other frames
            other_sims = np.concatenate([similarities[i, :i], similarities[i, i+1:]])
            if len(other_sims) > 0:
                # Use standard deviation as uniqueness measure , higher std = frame is very similar to some and very different to others
                uniqueness = np.std(other_sims)
                diversity_scores.append(uniqueness)
            else:
                diversity_scores.append(0)
        
        diversity_scores = np.array(diversity_scores)
        
        # apply temperature-scaled softmax for better weight distribution
        temperature = 2.0  
        diversity_scores = diversity_scores / temperature
        diversity_scores = np.exp(diversity_scores) / np.sum(np.exp(diversity_scores))
        

        assert abs(diversity_scores.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {diversity_scores.sum()}"
        
        logger.debug(f"Diversity weights - min: {diversity_scores.min():.4f}, "
                    f"max: {diversity_scores.max():.4f}, std: {diversity_scores.std():.4f}")
        
        return torch.tensor(diversity_scores, dtype=torch.float32)
        
    except Exception as e:
        logger.error(f"Error computing frame diversity: {e}", exc_info=True)
        # Return uniform weights as fallback
        return torch.ones(len(embeddings)) / len(embeddings)

def compute_temporal_coherence_weights(frame_embeddings, window_size=3):
    """
    Weight frames based on their temporal coherence
    Frames that maintain semantic consistency with neighbors get higher weights
    
    Args:
        frame_embeddings: Tensor of frame embeddings
        window_size: Size of temporal window for coherence calculation
    
    Returns:
        Tensor of weights summing to 1
    """
    try:
        if len(frame_embeddings) <= window_size:
            logger.warning(f"Too few frames ({len(frame_embeddings)}) for temporal coherence, using uniform weights")
            return torch.ones(len(frame_embeddings)) / len(frame_embeddings)
        
        coherence_scores = []
        embeddings_np = frame_embeddings.cpu().numpy()
        
        for i in range(len(embeddings_np)):
            # Get temporal window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(embeddings_np), i + window_size // 2 + 1)
            window = embeddings_np[start_idx:end_idx]
            
            # Calculate coherence as mean similarity to neighbors
            if len(window) > 1:
                similarities = cosine_similarity([embeddings_np[i]], window)[0]
                # exclude self-similarity
                similarities = [s for j, s in enumerate(similarities) if start_idx + j != i]
                coherence = np.mean(similarities) if similarities else 0
            else:
                coherence = 1.0
            
            coherence_scores.append(coherence)
        
        coherence_scores = np.array(coherence_scores)
        
        # Convert to weights using temperature-scaled softmax , higher coherence = higher weight (frames that fit well temporally)
        temperature = 2.0  
        coherence_scores = coherence_scores * temperature
        weights = np.exp(coherence_scores) / np.sum(np.exp(coherence_scores))
        

        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {weights.sum()}"
        
        logger.debug(f"Temporal coherence weights - min: {weights.min():.4f}, "
                    f"max: {weights.max():.4f}, std: {weights.std():.4f}")
        logger.debug(f"Window size: {window_size}")
        
        return torch.tensor(weights, dtype=torch.float32)
        
    except Exception as e:
        logger.error(f"Error computing temporal coherence weights: {e}", exc_info=True)
        # Return uniform weights as fallback
        return torch.ones(len(frame_embeddings)) / len(frame_embeddings)

def process_single_frame(frame_path, transform):
    """Process a single frame and return the preprocessed tensor"""
    try:
        image = Image.open(frame_path)
        
        # images with transparency
        if image.mode in ('RGBA', "P") and len(image.getbands()) == 4:
            image_org = image.copy()
            image = Image.new("RGB", image_org.size, (255, 255, 255))
            image.paste(image_org, mask=image_org.split()[3])
        else:
            image = image.convert("RGB")
        
 
        transformed = transform(image)
        return transformed, None
        
    except Exception as e:
        logger.warning(f"Error processing frame {frame_path}: {e}")
        return None, str(e)

def compute_video_embeddings(frames_root, model, transform, output_file, 
                           combination_method='average', batch_size=16, window_size=3):
    """
    Compute embeddings for videos from extracted frames
    
    Args:
        frames_root: Root directory containing video folders with frames
        model: The DINOv2 model
        transform: The image transform
        output_file: Where to save embeddings
        combination_method: 'average', 'weighted_diversity', or 'temporal_coherence'
        batch_size: Batch size for processing frames
        window_size: Window size for temporal coherence method
    """
    
    # Validate combination method
    valid_methods = ['average', 'weighted_diversity', 'temporal_coherence']
    if combination_method not in valid_methods:
        raise ValueError(f"Invalid combination method. Choose from: {valid_methods}")
    
    # output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    video_embeddings = {}
    processing_errors = {}
    stats = {
        'total_videos': 0,
        'successful_videos': 0,
        'failed_videos': 0,
        'total_frames_processed': 0,
        'failed_frames': 0,
    
    }
    
    # Get all video directories
    video_dirs = [d for d in os.listdir(frames_root) 
                  if os.path.isdir(os.path.join(frames_root, d))]
    
    # Sort for consistent processing
    video_dirs.sort()
    stats['total_videos'] = len(video_dirs)
    
    logger.info(f"Found {len(video_dirs)} video directories")
    logger.info(f"Using combination method: {combination_method}")
    if combination_method == 'temporal_coherence':
        logger.info(f"Temporal window size: {window_size}")
    
    # Process each video
    for video_idx, video_dir in enumerate(tqdm.tqdm(video_dirs, desc="Processing videos")):
        video_path = os.path.join(frames_root, video_dir)
        
        # Get video ID (remove _frames suffix if present)
        video_id = video_dir.replace('_frames', '')
        
        try:
            # Get all frame files
            frame_files = sorted([f for f in os.listdir(video_path) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            if not frame_files:
                logger.warning(f"No frames found in {video_dir}")
                processing_errors[video_id] = "No frames found"
                stats['failed_videos'] += 1
                continue
            
            frame_paths = [os.path.join(video_path, f) for f in frame_files]
            
            # Process frames in batches
            frame_embeddings = []
            frame_errors = []
            
            for i in range(0, len(frame_files), batch_size):
                batch_files = frame_files[i:i+batch_size]
                batch_images = []
                batch_indices = []
                
                for j, frame_file in enumerate(batch_files):
                    frame_path = os.path.join(video_path, frame_file)
                    tensor, error = process_single_frame(frame_path, transform)
                    
                    if tensor is not None:
                        batch_images.append(tensor)
                        batch_indices.append(i + j)
                    else:
                        frame_errors.append((frame_file, error))
                        stats['failed_frames'] += 1
                
                if batch_images:
                    # Stack batch
                    batch_tensor = torch.stack(batch_images).to(device)
                    
                    # Get embeddings
                    with torch.no_grad():
                        batch_embeddings = model(batch_tensor)
                        frame_embeddings.extend(batch_embeddings.cpu())
                    
                    stats['total_frames_processed'] += len(batch_images)
            
            if not frame_embeddings:
                logger.error(f"No valid frames for video {video_id}")
                processing_errors[video_id] = f"All {len(frame_files)} frames failed"
                stats['failed_videos'] += 1
                continue
            
            # Log frame processing issues if any
            if frame_errors:
                logger.warning(f"Video {video_id}: {len(frame_errors)}/{len(frame_files)} frames failed")
                if len(frame_errors) <= 3:
                    processing_errors[video_id] = f"{len(frame_errors)} frames failed: {frame_errors}"
                else:
                    processing_errors[video_id] = f"{len(frame_errors)} frames failed"
            
            # Combine frame embeddings
            frame_embeddings = torch.stack(frame_embeddings)
            
            # Apply combination method
            if combination_method == 'average':
                video_embedding = torch.mean(frame_embeddings, dim=0)
                
            elif combination_method == 'weighted_diversity':
                diversity_weights = compute_frame_diversity(frame_embeddings)
                video_embedding = torch.sum(frame_embeddings * diversity_weights.unsqueeze(1), dim=0)
           
                
            elif combination_method == 'temporal_coherence':
                coherence_weights = compute_temporal_coherence_weights(frame_embeddings, window_size)
                video_embedding = torch.sum(frame_embeddings * coherence_weights.unsqueeze(1), dim=0)
              
               
            
            video_embeddings[video_id] = video_embedding
            stats['successful_videos'] += 1
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
            processing_errors[video_id] = f"Processing error: {str(e)}"
            stats['failed_videos'] += 1
            continue
        
        # Log progress
        if (video_idx + 1) % 100 == 0:
            logger.info(f"Progress: {video_idx + 1}/{len(video_dirs)} videos processed")
    
    # Save embeddings
    logger.info(f"Saving embeddings for {len(video_embeddings)} videos...")
    torch.save(video_embeddings, output_file)
    logger.info(f"Embeddings saved to: {output_file}")
    
    # Save metadata
    metadata = {
        'dataset': dataset,
        'model': model_name,
        'combination_method': combination_method,
        'window_size': window_size if combination_method == 'temporal_coherence' else None,
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'embedding_dim': video_embeddings[next(iter(video_embeddings))].shape[0] if video_embeddings else 0
    }
    
    metadata_file = output_file.replace('.torch', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_file}")
    
    # Save error log
    if processing_errors:
        error_file = output_file.replace('.torch', '_processing_errors.json')
        with open(error_file, 'w') as f:
            json.dump(processing_errors, f, indent=2)
        logger.warning(f"Processing errors saved to: {error_file}")
    
   
    # Final statistics
    logger.info("="*60)
    logger.info("Processing Statistics:")
    logger.info(f"Total videos: {stats['total_videos']}")
    logger.info(f"Successful: {stats['successful_videos']} ({stats['successful_videos']/stats['total_videos']*100:.1f}%)")
    logger.info(f"Failed: {stats['failed_videos']} ({stats['failed_videos']/stats['total_videos']*100:.1f}%)")
    logger.info(f"Total frames processed: {stats['total_frames_processed']}")
    logger.info(f"Failed frames: {stats['failed_frames']}")
    logger.info("="*60)
    
    return video_embeddings

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate DINOv2 embeddings for video frames')
    parser.add_argument('--frames_root', type=str, default=frames_root,
                       help='Root directory containing video frame folders')
    parser.add_argument('--dataset', type=str, default=dataset,
                       help='Dataset name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (if not specified, auto-generated based on method)')
    parser.add_argument('--combination', type=str, default='average',
                       choices=['average', 'weighted_diversity', 'temporal_coherence'],
                       help='How to combine frame embeddings')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--window_size', type=int, default=3,
                       help='Window size for temporal coherence method (default: 3)')
    
    args = parser.parse_args()
    
    # Agenerate output filename based on combination method if not specified
    if args.output is None:
   
        if args.dataset != dataset:
            output_base_updated = f"/work/shixu/climate_project/features/{args.dataset}_dinov2"
        else:
            output_base_updated = output_base
        
        #  append combination method to filename for consistency
        args.output = f"{output_base_updated}_{args.combination}.torch"
    
    # Set up logging
    logger, log_file, error_log_file = setup_logging(args.dataset, args.combination)
    
    logger.info("="*60)
    logger.info("DINOv2 Video Embedding Generation")
    logger.info("="*60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Frames directory: {args.frames_root}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Combination method: {args.combination}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.combination == 'temporal_coherence':
        logger.info(f"Window size: {args.window_size}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Error log: {error_log_file}")
    logger.info("="*60)
    
    try:
        video_embeddings = compute_video_embeddings(
            args.frames_root,
            model,
            transform,
            args.output,
            combination_method=args.combination,
            batch_size=args.batch_size,
            window_size=args.window_size
        )
        
        # Print statistics
        if video_embeddings:
            sample_embedding = next(iter(video_embeddings.values()))
            logger.info(f"Final Statistics:")
            logger.info(f"Total videos processed: {len(video_embeddings)}")
            logger.info(f"Embedding dimension: {sample_embedding.shape}")
            logger.info(f"Sample video IDs: {list(video_embeddings.keys())[:5]}")
            
            # Print method-specific info
            if args.combination == 'temporal_coherence':
                logger.info(f"Note: temporal_coherence uses window_size={args.window_size} for temporal consistency")
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"error during processing: {e}", exc_info=True)
        raise