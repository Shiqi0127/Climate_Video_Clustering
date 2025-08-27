#!/usr/bin/env python3
"""
ConvNeXt V2 embedding generation for video frames 
Processes frames organized by video and creates video-level embeddings
Includes average, weighted_diversity, max_confidence, and weighted_confidence combination methods
weighted_frequency is implemented but not compared in the thesis

Acknowledgments:
- Adapted from Prasse et al. (2025)'s ConvNeXt V2 implementation (convnextv2.py)
  Original script: Single image embedding generation for dataset processing
  This adaptation: Extended for video frame processing with multiple combination methods
"""

## Dataset paths - modify to match your setup
frames_root = "/work/shixu/climate_project/data/climate_3k_static/frames_static"
dataset = "climate_3k_static"
model_name = "facebook/convnextv2-tiny-1k-224"
# Base path without extension - will add method name and .torch extension
output_base = f"/work/shixu/climate_project/features/{dataset}_convnextv2"

############################################################################################################

## load packages
import torch
import glob
from torchvision import transforms
from PIL import Image, ImageFile
import tqdm
import os
import sys
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch.nn as nn
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import logging
from datetime import datetime
import traceback

# allow to load images that exceed max size
ImageFile.LOAD_TRUNCATED_IMAGES = True

# set cache directory for Hugging Face
os.environ['TRANSFORMERS_CACHE'] = '/work/shixu/climate_project/.cache/huggingface'
os.environ['HF_HOME'] = '/work/shixu/climate_project/.cache/huggingface'

# Set up logging
def setup_logging(dataset_name, combination_method):
   
    log_dir = Path("/work/shixu/climate_project/logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    general_log = log_dir / f"{dataset_name}_{combination_method}_{timestamp}.log"
    error_log = log_dir / f"{dataset_name}_{combination_method}_{timestamp}_errors.log"
    
  
    logger = logging.getLogger('video_embedding')
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

# Initialize logger (will be configured in main)
logger = logging.getLogger('video_embedding')

# Global variable for full model (used in confidence methods)
FULL_MODEL_FOR_CONFIDENCE = None

# Load the main model
try:
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=os.environ['TRANSFORMERS_CACHE'])
    model = ConvNextV2ForImageClassification.from_pretrained(model_name, cache_dir=os.environ['TRANSFORMERS_CACHE'])
    model.classifier = nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully: {model_name}")
    logger.info(f"Using device: {device}")
except Exception as e:
    logger.error(f"Failed to load model: {e}", exc_info=True)
    raise

def load_full_model_for_confidence():
    """Load the full model with classification head for confidence-based methods"""
    global FULL_MODEL_FOR_CONFIDENCE
    
    try:
        logger.info("Loading full ConvNeXt V2 model with classification head for confidence scoring...")
      
        
        full_model = ConvNextV2ForImageClassification.from_pretrained(
            model_name,
            cache_dir=os.environ['TRANSFORMERS_CACHE']
        )
        full_model.to(device)
        full_model.eval()
        
        FULL_MODEL_FOR_CONFIDENCE = full_model
        logger.info("Full model loaded successfully and cached in memory")
        
    except Exception as e:
        logger.error(f"Failed to load full model for confidence scoring: {e}", exc_info=True)
        raise



def compute_frame_diversity(embeddings):
    """
    diversity computation using standard deviation of similarities
    Frames that are uniquely different get higher weights
    """
    try:
        if len(embeddings) == 1:
            return torch.ones(1)
        
        # Edge case: very few frames
        if len(embeddings) < 3:
            logger.warning(f"Only {len(embeddings)} frames available for diversity weighting, using uniform weights")
            return torch.ones(len(embeddings)) / len(embeddings)
      
        embeddings_np = embeddings.cpu().numpy()
        
        #  pairwise similarities
        similarities = cosine_similarity(embeddings_np)
        
        # For each frame, clauculate its uniqueness score
        diversity_scores = []
        for i in range(len(embeddings)):
            # Get similarities to other frames
            other_sims = np.concatenate([similarities[i, :i], similarities[i, i+1:]])
            if len(other_sims) > 0:
                # Use standard deviation as uniqueness measure
                # Higher std = frame is very similar to some and very different to others
                uniqueness = np.std(other_sims)
                # Alternative: use negative mean similarity.....uniqueness = 1 - np.mean(other_sims)
                diversity_scores.append(uniqueness)
            else:
                diversity_scores.append(0)
        
        diversity_scores = np.array(diversity_scores)
        
        # Apply temperature-scaled softmax for better weight distribution
        temperature = 2.0  
        diversity_scores = diversity_scores / temperature
        diversity_scores = np.exp(diversity_scores) / np.sum(np.exp(diversity_scores))
        
        # validate weights
        assert abs(diversity_scores.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {diversity_scores.sum()}"
        
        logger.debug(f"Diversity weights - min: {diversity_scores.min():.4f}, "
                    f"max: {diversity_scores.max():.4f}, std: {diversity_scores.std():.4f}")
        
        return torch.tensor(diversity_scores, dtype=torch.float32)
        
    except Exception as e:
        logger.error(f"Error computing frame diversity: {e}", exc_info=True)
        # Return uniform weights as fallback
        return torch.ones(len(embeddings)) / len(embeddings)

def compute_frame_frequency_weights(frame_embeddings, threshold=0.95):
    """
    frequency weighting - unique frames get higher weights
    """
    try:
        if len(frame_embeddings) == 1:
            return torch.ones(1)
        
        # Edge case: very few frames
        if len(frame_embeddings) < 3:
            logger.warning(f"Only {len(frame_embeddings)} frames available for frequency weighting, using uniform weights")
            return torch.ones(len(frame_embeddings)) / len(frame_embeddings)
        
        embeddings_np = frame_embeddings.cpu().numpy()
        similarities = cosine_similarity(embeddings_np)
        
        # Count how many frames each frame is similar to
        similarity_counts = np.sum(similarities > threshold, axis=1)
        
        # Inverse frequency weighting: rare frames get higher weight....avoid division by zero
        weights = 1.0 / (similarity_counts + 1.0)
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        # Validate
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {weights.sum()}"
        
        logger.debug(f"Frequency weights - min: {weights.min():.4f}, "
                    f"max: {weights.max():.4f}, unique frames: {np.sum(similarity_counts == 1)}")
        
        return torch.tensor(weights, dtype=torch.float32)
        
    except Exception as e:
        logger.error(f"Error computing frame frequency weights: {e}", exc_info=True)
        # Return uniform weights as fallback
        return torch.ones(len(frame_embeddings)) / len(frame_embeddings)

def get_max_confidence_frame(frame_embeddings, model, processor, frame_paths):
    """
    Select single frame with maximum confidence from pre-trained classifier
    Returns the embedding of the single most confident frame
    """
    try:
        if FULL_MODEL_FOR_CONFIDENCE is None:
            raise ValueError("Full model not loaded for confidence scoring")
        
        full_model = FULL_MODEL_FOR_CONFIDENCE
        
        max_confidence = -float('inf')
        best_embedding = None
        best_frame_idx = 0
        confidence_scores = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                image = Image.open(frame_path)
                
                if image.mode in ('RGBA', "P") and len(image.getbands()) == 4:
                    image_org = image.copy()
                    image = Image.new("RGB", image_org.size, (255, 255, 255))
                    image.paste(image_org, mask=image_org.split()[3])
                else:
                    image = image.convert("RGB")
                
                inputs = processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = full_model(**inputs.to(device))
                    logits = outputs.logits
                    
                    # Get max confidence (softmax of logits)
                    probs = torch.softmax(logits, dim=-1)
                    max_prob = torch.max(probs).item()
                    confidence_scores.append(max_prob)
                    
                    if max_prob > max_confidence:
                        max_confidence = max_prob
                        best_embedding = frame_embeddings[i]
                        best_frame_idx = i
            
            except Exception as e:
                logger.warning(f"Error processing frame {frame_path} for confidence: {e}")
                confidence_scores.append(0)
                continue
        
        logger.debug(f"Max confidence selection - selected frame {best_frame_idx} with confidence {max_confidence:.4f}")
        logger.debug(f"Confidence distribution - mean: {np.mean(confidence_scores):.4f}, "
                    f"std: {np.std(confidence_scores):.4f}")
        
        return best_embedding if best_embedding is not None else frame_embeddings[0]
        
    except Exception as e:
        logger.error(f"Error in max confidence selection: {e}", exc_info=True)
        # Return first frame as fallback
        return frame_embeddings[0]

def get_weighted_confidence_combination(frame_embeddings, model, processor, frame_paths, temperature=2.0):
    """
    Compute weighted combination of frame embeddings based on classifier confidence scores
    
    Args:
        frame_embeddings: Tensor of frame embeddings
        model: Full model (not used, used the global one)
        processor: Image processor
        frame_paths: List of paths to frame images
        temperature: Temperature for softmax (higher = more uniform weights)
    
    Returns:
        Weighted combination of frame embeddings
    """
    try:
        if FULL_MODEL_FOR_CONFIDENCE is None:
            raise ValueError("Full model not loaded for confidence scoring.")
        
        full_model = FULL_MODEL_FOR_CONFIDENCE
        
        # Edge case: single frame
        if len(frame_embeddings) == 1:
            return frame_embeddings[0]
        
        confidence_scores = []
        
        # Get confidence score for each frame
        for i, frame_path in enumerate(frame_paths):
            try:
                image = Image.open(frame_path)
                
                if image.mode in ('RGBA', "P") and len(image.getbands()) == 4:
                    image_org = image.copy()
                    image = Image.new("RGB", image_org.size, (255, 255, 255))
                    image.paste(image_org, mask=image_org.split()[3])
                else:
                    image = image.convert("RGB")
                
                inputs = processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = full_model(**inputs.to(device))
                    logits = outputs.logits
                    
                    # Get max confidence (softmax of logits)
                    probs = torch.softmax(logits, dim=-1)
                    max_prob = torch.max(probs).item()
                    confidence_scores.append(max_prob)
            
            except Exception as e:
                logger.warning(f"Error processing frame {frame_path} for confidence: {e}")
                # Use a low confidence score for failed frames
                confidence_scores.append(0.1)
                continue
        
        # Convert confidence scores to weights using temperature-scaled softmax
        confidence_scores = np.array(confidence_scores)
        
        # Apply temperature scaling
        scaled_scores = confidence_scores / temperature
        
        # Compute softmax weights
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Subtract max for numerical stability
        weights = exp_scores / np.sum(exp_scores)
        
        # Validate weights
        assert abs(weights.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {weights.sum()}"
        
        # Log statistics
        best_frame_idx = np.argmax(confidence_scores)
        logger.debug(f"Weighted confidence - temperature: {temperature}")
        logger.debug(f"Confidence scores - min: {confidence_scores.min():.4f}, "
                    f"max: {confidence_scores.max():.4f}, mean: {confidence_scores.mean():.4f}")
        logger.debug(f"Confidence weights - min: {weights.min():.4f}, "
                    f"max: {weights.max():.4f}, best frame: {best_frame_idx}")
        
        # Create weighted combination
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(frame_embeddings.device)
        video_embedding = torch.sum(frame_embeddings * weights_tensor.unsqueeze(1), dim=0)
        
        return video_embedding
        
    except Exception as e:
        logger.error(f"Error in weighted confidence combination: {e}", exc_info=True)
        # Return simple average as fallback
        return torch.mean(frame_embeddings, dim=0)

def get_uniform_temporal_frames(frame_embeddings, n_frames=4):
    """
    Select n frames uniformly distributed across the video timeline
    """
    try:
        total_frames = len(frame_embeddings)
        
        if total_frames <= n_frames:
            # If have fewer frames than requested, use all
            return torch.mean(frame_embeddings, dim=0)
        
        # select indices uniformly
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        selected_embeddings = frame_embeddings[indices]
        
        logger.debug(f"Selected {n_frames} frames from {total_frames} total frames at indices: {indices}")
        
        return torch.mean(selected_embeddings, dim=0)
        
    except Exception as e:
        logger.error(f"Error in uniform temporal selection: {e}", exc_info=True)
        return torch.mean(frame_embeddings, dim=0)

def process_single_frame(frame_path, processor):
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
        
        # Process image
        processed = processor(images=image, return_tensors="pt")
        return processed['pixel_values'].squeeze(0), None
        
    except Exception as e:
        logger.warning(f"Error processing frame {frame_path}: {e}")
        return None, str(e)

def compute_video_embeddings(frames_root, model, processor, output_file, 
                           combination_method='average', batch_size=16, temperature=2.0):
    """
    Compute embeddings for videos from extracted frames
    
    Args:
        frames_root: Root directory containing video folders with frames
        model: The ConvNeXt model
        processor: The image processor
        output_file: Where to save embeddings
        combination_method: 'average', 'weighted_diversity', 'weighted_frequency', 
                          'max_confidence', 'weighted_confidence', or 'uniform_temporal'
        batch_size: Batch size for processing frames
        temperature: Temperature for weighted_confidence method
    """
    
    # Validate combination method
    valid_methods = ['average', 'weighted_diversity', 'weighted_frequency', 
                     'max_confidence', 'weighted_confidence', ]
    if combination_method not in valid_methods:
        raise ValueError(f"Invalid combination method. Choose from: {valid_methods}")
    
    # Create output directory
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
    
    # sort for consistent processing
    video_dirs.sort()
    stats['total_videos'] = len(video_dirs)
    
    logger.info(f"Found {len(video_dirs)} video directories")
    logger.info(f"Using combination method: {combination_method}")
    if combination_method == 'weighted_confidence':
        logger.info(f"Temperature for weighted confidence: {temperature}")
    
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
            valid_frame_paths = []
            
            for i in range(0, len(frame_files), batch_size):
                batch_files = frame_files[i:i+batch_size]
                batch_images = []
                batch_indices = []
                
                for j, frame_file in enumerate(batch_files):
                    frame_path = os.path.join(video_path, frame_file)
                    tensor, error = process_single_frame(frame_path, processor)
                    
                    if tensor is not None:
                        batch_images.append(tensor)
                        batch_indices.append(i + j)
                        valid_frame_paths.append(frame_path)
                    else:
                        frame_errors.append((frame_file, error))
                        stats['failed_frames'] += 1
                
                if batch_images:
                    # stack batch
                    batch_tensor = torch.stack(batch_images).to(device)
                    
                    # Get embeddings
                    with torch.no_grad():
                        batch_output = model(pixel_values=batch_tensor)
                        batch_embeddings = batch_output.logits
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
               
                
            elif combination_method == 'weighted_frequency':
                frequency_weights = compute_frame_frequency_weights(frame_embeddings)
                video_embedding = torch.sum(frame_embeddings * frequency_weights.unsqueeze(1), dim=0)
               
                
            elif combination_method == 'max_confidence':
             
                video_embedding = get_max_confidence_frame(
                    frame_embeddings, model, processor, valid_frame_paths
                )
                
            elif combination_method == 'weighted_confidence':
           
                video_embedding = get_weighted_confidence_combination(
                    frame_embeddings, model, processor, valid_frame_paths, temperature
                )
                
           
            
            else:
                raise ValueError(f"Unknown combination method: {combination_method}")
            
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
        'temperature': temperature if combination_method == 'weighted_confidence' else None,
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
    
    parser = argparse.ArgumentParser(description='Generate ConvNeXt V2 embeddings for video frames')
    parser.add_argument('--frames_root', type=str, default=frames_root,
                       help='Root directory containing video frame folders')
    parser.add_argument('--dataset', type=str, default=dataset,
                       help='Dataset name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path ')
    parser.add_argument('--combination', type=str, default='average',
                       choices=['average', 'weighted_diversity', 'weighted_frequency', 
                               'max_confidence', 'weighted_confidence', 'uniform_temporal'],
                       help='How to combine frame embeddings')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature for weighted_confidence method (higher = more uniform weights)')
    
    args = parser.parse_args()
    

  
    if args.output is None:
   
        if args.dataset != dataset:
            output_base_updated = f"/work/shixu/climate_project/features/{args.dataset}_convnextv2"
        else:
            output_base_updated = output_base
    
        
        args.output = f"{output_base_updated}_{args.combination}.torch"

    
    # Set up logging
    logger, log_file, error_log_file = setup_logging(args.dataset, args.combination)
    
    logger.info("="*60)
    logger.info("ConvNeXt V2 Video Embedding Generation - Final Version")
    logger.info("="*60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Frames directory: {args.frames_root}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Combination method: {args.combination}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.combination == 'weighted_confidence':
        logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Error log: {error_log_file}")
    logger.info("="*60)
    
    # Load full model if using confidence-based methods
    if args.combination in ['max_confidence', 'weighted_confidence']:
        load_full_model_for_confidence()
    
    try:
        video_embeddings = compute_video_embeddings(
            args.frames_root,
            model,
            processor,
            args.output,
            combination_method=args.combination,
            batch_size=args.batch_size,
            temperature=args.temperature
        )
        
        # Print statistics
        if video_embeddings:
            sample_embedding = next(iter(video_embeddings.values()))
            logger.info(f"Final Statistics:")
            logger.info(f"Total videos processed: {len(video_embeddings)}")
            logger.info(f"Embedding dimension: {sample_embedding.shape}")
            logger.info(f"Sample video IDs: {list(video_embeddings.keys())[:5]}")
            
        
        logger.info("Processing completed successfully")
        
        # Clean up full model if loaded
        if FULL_MODEL_FOR_CONFIDENCE is not None:
            del FULL_MODEL_FOR_CONFIDENCE
            torch.cuda.empty_cache()
            logger.info("Cleaned up full model")
        
    except Exception as e:
        logger.error(f"error during processing: {e}", exc_info=True)
        raise