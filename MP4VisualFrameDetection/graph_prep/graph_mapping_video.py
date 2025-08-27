#!/usr/bin/env python3
"""
Adapted from Prasse et al. (2025)'s implementation graph_mapping.py for image data
This version processes video embeddings and creates graph input for multicut solver.

Changes from original include:
- Removed ablation/split functionality
- Fixed undefined args.setting reference from original
- Reorganized file paths (added 'results/' directory structure)
- Changed from img_ids to video_ids for videos
"""



## conda env: env_prep

# import packages
import torch
import numpy as np
import argparse
import tqdm
import os

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create graph input from cosine similarities for video data.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model_config', type=str, required=True, help='Model configuration name')
    parser.add_argument('--embs', type=str, required=True, help='Path to the embeddings file')
    args = parser.parse_args()

    # Load precomputed cosine similarities
    dist = torch.load(f"results/emb_dist/cossim_{args.dataset}_{args.model_config}.torch")
    
    # Load original embeddings to get video IDs
    embs = torch.load(f"{args.embs}/{args.dataset}_{args.model_config}.torch")
    video_ids = list(embs.keys())
    
    # Create mapping from video_ids to node_ids
    node_ids = [i for i in range(len(video_ids))]
    key_mapping = dict(zip(video_ids, node_ids))
    
    # Save key mapping for later use
    os.makedirs("results", exist_ok=True)
    torch.save(key_mapping, f"results/key_mapping_{args.dataset}_{args.model_config}.torch")
    print(f"Key mapping created for {len(key_mapping)} videos")
    
    # Create edges between nodes (videos) based on cosine similarities
    edge_array = []
    weight_array = []
    
    print("Creating edges from similarity matrix...")
    for k, v in tqdm.tqdm(dist.items()):
        for t, similarity in v.items():
            edge_array.append(np.array([int(key_mapping[k]), int(key_mapping[t])], dtype=int))
            weight_array.append(similarity)
    
    edge_array = np.asarray(edge_array, dtype=int)
    weight_array = np.asarray(weight_array, dtype=float)
    
    # Scale similarities to the interval [0,1]
    weight_array_norm = (weight_array - np.min(weight_array)) / (np.max(weight_array) - np.min(weight_array))
    
    print(f"Edge array shape: {edge_array.shape}")
    print(f"Weight array shape: {weight_array.shape}")
    print(f"Weight range: [{np.min(weight_array_norm):.6f}, {np.max(weight_array_norm):.6f}]")
    
    # Combine edges and weights
    final_array = np.concatenate((edge_array, weight_array_norm[:, None]), axis=1)
    print(f"Final array shape: {final_array.shape}")
    
    # Create output directory
    output_dir = f"multicut/cossim_{args.dataset}_{args.model_config}_eval"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write input file for multicut solver
    input_file = os.path.join(output_dir, "input.txt")
    with open(input_file, "w") as f:
        # First line: number of nodes, number of edges
        f.write(f"{len(key_mapping)} {len(edge_array)}\n")
        # Following lines: node1 node2 weight
        np.savetxt(f, final_array, fmt='%d %d %.6f')
    
    print(f"Graph input file created: {input_file}")
    print(f"Nodes: {len(key_mapping)}, Edges: {len(edge_array)}")
    
    # Print weight distribution stats
    print("\nWeight distribution:")
    histo, bin_edges = np.histogram(weight_array_norm, bins=5)
    for i in range(len(histo)):
        print(f"  [{bin_edges[i]:.3f} - {bin_edges[i+1]:.3f}]: {histo[i]} edges")