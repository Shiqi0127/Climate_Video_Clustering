# Climate Video Clustering Pipeline

A frame-based embedding approach for clustering climate-related videos using Minimum Cost Multicut.

## Overview

This repository contains the implementation of a video clustering pipeline, which adapts the Minimum Cost Multicut Clustering method from [Prasse et al. (2025)](https://github.com/KathPra/MP4VisualFrameDetection) to climate-related videos. The pipeline extracts frames from videos, generates embeddings using foundation models (DINOv2 and ConvNeXt V2), and clusters them to identify recurring visual frames in climate discourse.

## Features

- **Frame Extraction Methods**:
  - Static: Uniformly spaced frame extraction
  - Diverse: Visually diverse frame selection 

- **Multiple Frame Combination Methods**:
  - Average: Simple mean of frame embeddings
  - Weighted Diversity: Frames weighted by visual uniqueness
  - Max Confidence: Single most confident frame selection
  - Weighted Confidence: Confidence-weighted combination
  - Temporal Coherence (DINOv2 only): Temporally consistent frame weighting

- **Embedding Models**:
  - DINOv2 (ViT-B/14)
  - ConvNeXt V2 (Tiny)

## Pipeline Architecture

```
1. Frame Extraction (extract_frames_density.py)
   ├── Static extraction (uniform sampling)
   └── Diverse extraction (clustering-based)
   
2. Embedding Generation
   ├── DINOv2 embeddings (dinov2_video.py)
   └── ConvNeXt V2 embeddings (convnextv2_video.py)
   
3. Similarity Computation (cossim.py)
   └── Pairwise cosine similarities
   
4. Graph Construction (graph_mapping_video.py)
   └── Convert similarities to graph format
   
5. Multicut Clustering (external solver)
   └── Solve minimum cost multicut problem
```

## Requirements

### Python Dependencies
```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
PIL>=9.0.0
tqdm>=4.62.0

# Model-specific
transformers>=4.30.0  # For ConvNeXt V2
dinov2  # Facebook's DINOv2 (install from GitHub)

# Additional
matplotlib>=3.5.0
pandas>=1.3.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/climate-video-clustering.git
cd climate-video-clustering
```

2. Create a conda environment:
```bash
conda create -n climate-clustering python=3.9
conda activate climate-clustering
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install DINOv2:
```bash
pip install git+https://github.com/facebookresearch/dinov2.git
```

5. Set up cache directories (modify paths as needed):
```bash
export TRANSFORMERS_CACHE=/path/to/cache/huggingface
export HF_HOME=/path/to/cache/huggingface
```

## Usage

### 1. Frame Extraction

Extract frames from videos using density-based sampling:

```bash
# Static extraction (uniform sampling)
python extract_frames_density.py \
    --video_list video_ids.txt \
    --output_root /path/to/output \
    --dataset climate_3k_static \
    --strategy static \
    --fps_density 1.0

# Diverse extraction (clustering-based)
python extract_frames_density.py \
    --video_list video_ids.txt \
    --output_root /path/to/output \
    --dataset climate_3k_diverse \
    --strategy diverse \
    --fps_density 1.0 \
    --sample_rate 10
```

### 2. Generate Embeddings

#### DINOv2 Embeddings
```bash
# Average combination
python dinov2_video.py \
    --frames_root /path/to/frames \
    --dataset climate_3k_static \
    --combination average

# Weighted diversity
python dinov2_video.py \
    --frames_root /path/to/frames \
    --dataset climate_3k_static \
    --combination weighted_diversity

# Temporal coherence
python dinov2_video.py \
    --frames_root /path/to/frames \
    --dataset climate_3k_static \
    --combination temporal_coherence \
    --window_size 3
```

#### ConvNeXt V2 Embeddings
```bash
# Average combination
python convnextv2_video.py \
    --frames_root /path/to/frames \
    --dataset climate_3k_static \
    --combination average

# Max confidence
python convnextv2_video.py \
    --frames_root /path/to/frames \
    --dataset climate_3k_static \
    --combination max_confidence

# Weighted confidence
python convnextv2_video.py \
    --frames_root /path/to/frames \
    --dataset climate_3k_static \
    --combination weighted_confidence \
    --temperature 2.0
```

### 3. Calculate Similarities

```bash
python cossim.py \
    --dataset climate_3k_static \
    --model_config dinov2_average \
    --embs /path/to/embeddings \
    --setting evaluation
```

### 4. Create Graph Input

```bash
python graph_mapping_video.py \
    --dataset climate_3k_static \
    --model_config dinov2_average \
    --embs /path/to/embeddings
```

### 5. Run Multicut Clustering

The graph files are formatted for the multicut solver. Follow the instructions in [Prasse et al.'s repository](https://github.com/KathPra/MP4VisualFrameDetection) to run the actual clustering.

## Output Structure

```
output_directory/
├── climate_3k_static/
│   ├── frames_static/          # Uniformly sampled frames
│   │   ├── video_id_001/
│   │   │   ├── frame_0000.jpg
│   │   │   ├── frame_0001.jpg
│   │   │   └── metadata.json
│   │   └── ...
│   ├── frames_diverse/         # Diverse frames
│   │   └── ...
│   └── summaries/             # Extraction statistics
│
├── features/
│   ├── climate_3k_static_dinov2_average.torch
│   ├── climate_3k_static_dinov2_weighted_diversity.torch
│   ├── climate_3k_static_convnextv2_average.torch
│   └── ...
│
├── results/
│   ├── emb_dist/
│   │   └── cossim_climate_3k_static_dinov2_average.torch
│   └── key_mapping_climate_3k_static_dinov2_average.torch
│
└── multicut/
    └── cossim_climate_3k_static_dinov2_average_eval/
        └── input.txt
```

## Configuration Options

### Frame Extraction
- `--fps_density`: Frames per second to extract (default: 1.0)
- `--num_frames`: Fixed number of frames (overrides density calculation)
- `--min_quality`: Minimum standard deviation for frame quality (default: 10.0)
- `--sample_rate`: Sample every Nth frame for diversity analysis (default: 10)

### Embedding Generation
- `--batch_size`: Batch size for processing frames (default: 16)
- `--temperature`: Temperature for weighted confidence method (default: 2.0)
- `--window_size`: Window size for temporal coherence (default: 3)

## Logging

All scripts output detailed logs in the `logs/` directory:
- General logs: `{dataset}_{method}_{timestamp}.log`
- Error logs: `{dataset}_{method}_{timestamp}_errors.log`


## Acknowledgments

This implementation builds upon the work of [Prasse et al. (2025)](https://github.com/KathPra/MP4VisualFrameDetection) on visual frame detection using minimum cost multicut clustering. The original implementation for image data has been adapted for video processing with additional frame selection and combination strategies.


## Troubleshooting

### Common Issues

1. **Frame extraction produces too few frames**
   - Check video duration and adjust `--fps_density`
   - Verify video file

2. **DINOv2 import errors**
   - Ensure DINOv2 is properly installed from GitHub
   - Check CUDA compatibility

3. **Confidence methods fail**
   - These methods require the full model with classification head
   - Ensure sufficient GPU memory for loading two models

### Performance Tips

- For large datasets, process videos in parallel using multiple GPUs
- Use `--combination average` for fastest processing
- Static frame extraction is faster than diverse extraction
- Consider using lower `--fps_density` for very long videos
