#!/usr/bin/env python3
"""
Diagnostic script to analyze instance sizes before and after merging.
"""

import os
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


def analyze_laz_instances(laz_path: str, name: str = None):
    """Load LAZ file and return instance size statistics."""
    if not os.path.exists(laz_path):
        print(f"File not found: {laz_path}")
        return None, None
    
    print(f"\nLoading: {laz_path}")
    las = laspy.read(laz_path, laz_backend=laspy.LazBackend.LazrsParallel)
    
    # Get instance IDs
    if hasattr(las, 'PredInstance'):
        instances = np.array(las.PredInstance)
    else:
        print(f"  No PredInstance attribute found!")
        return None, None
    
    # Count points per instance
    unique, counts = np.unique(instances, return_counts=True)
    instance_sizes = dict(zip(unique, counts))
    
    # Filter out background (0) and negative values
    tree_instances = {k: v for k, v in instance_sizes.items() if k > 0}
    
    print(f"  Total points: {len(instances)}")
    print(f"  Background points (instance=0): {instance_sizes.get(0, 0)}")
    print(f"  Negative instance points: {sum(v for k, v in instance_sizes.items() if k < 0)}")
    print(f"  Tree instances: {len(tree_instances)}")
    
    if tree_instances:
        sizes = list(tree_instances.values())
        print(f"  Instance size stats:")
        print(f"    Min: {min(sizes)}")
        print(f"    Max: {max(sizes)}")
        print(f"    Mean: {np.mean(sizes):.1f}")
        print(f"    Median: {np.median(sizes):.1f}")
        
        # Count small instances
        small_100 = sum(1 for s in sizes if s < 100)
        small_300 = sum(1 for s in sizes if s < 300)
        small_1000 = sum(1 for s in sizes if s < 1000)
        print(f"    Instances < 100 points: {small_100}")
        print(f"    Instances < 300 points: {small_300}")
        print(f"    Instances < 1000 points: {small_1000}")
    
    return tree_instances, name or Path(laz_path).stem


def check_buffer_logic(tile_folder: str, buffer: float = 0.2):
    """Check which instances have points in the buffer zone."""
    laz_path = os.path.join(tile_folder, "pc_with_species.laz")
    if not os.path.exists(laz_path):
        print(f"File not found: {laz_path}")
        return
    
    print(f"\n=== Checking buffer logic for {tile_folder} ===")
    las = laspy.read(laz_path, laz_backend=laspy.LazBackend.LazrsParallel)
    points = np.vstack((las.x, las.y, las.z)).T
    instances = np.array(las.PredInstance)
    
    # Compute tile boundary
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    print(f"Tile boundary: X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]")
    print(f"Buffer zone: {buffer}m from edges")
    
    # Check which points are in buffer zone
    in_buffer_x = (points[:, 0] < min_x + buffer) | (points[:, 0] > max_x - buffer)
    in_buffer_y = (points[:, 1] < min_y + buffer) | (points[:, 1] > max_y - buffer)
    in_buffer = in_buffer_x | in_buffer_y
    
    # Find instances with points in buffer
    unique_instances = np.unique(instances)
    instances_in_buffer = set()
    instances_fully_inside = set()
    
    for inst_id in unique_instances:
        if inst_id <= 0:
            continue
        mask = instances == inst_id
        if np.any(in_buffer[mask]):
            instances_in_buffer.add(inst_id)
        else:
            instances_fully_inside.add(inst_id)
    
    print(f"\nTotal tree instances: {len(unique_instances) - 1}")  # -1 for background
    print(f"Instances fully inside (will be kept): {len(instances_fully_inside)}")
    print(f"Instances touching buffer (will be removed): {len(instances_in_buffer)}")
    
    # Show some examples
    if instances_in_buffer:
        print(f"\nExample instances touching buffer (first 10):")
        for inst_id in list(instances_in_buffer)[:10]:
            mask = instances == inst_id
            inst_points = points[mask]
            min_x_i, max_x_i = np.min(inst_points[:, 0]), np.max(inst_points[:, 0])
            min_y_i, max_y_i = np.min(inst_points[:, 1]), np.max(inst_points[:, 1])
            print(f"  Instance {inst_id}: {np.sum(mask)} points, X[{min_x_i:.2f}, {max_x_i:.2f}], Y[{min_y_i:.2f}, {max_y_i:.2f}]")


def plot_instance_distributions(instance_data_list: list, output_path: str):
    """Plot instance size distributions."""
    fig, axes = plt.subplots(1, len(instance_data_list), figsize=(6*len(instance_data_list), 5))
    if len(instance_data_list) == 1:
        axes = [axes]
    
    for ax, (instances, name) in zip(axes, instance_data_list):
        if instances is None:
            continue
        
        sizes = list(instances.values())
        
        # Histogram with log scale
        ax.hist(sizes, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Instance Size (points)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\n({len(sizes)} instances)')
        ax.set_yscale('log')
        
        # Add vertical lines for thresholds
        ax.axvline(x=100, color='red', linestyle='--', label='100 pts')
        ax.axvline(x=300, color='orange', linestyle='--', label='300 pts')
        ax.axvline(x=1000, color='green', linestyle='--', label='1000 pts')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose instance sizes")
    parser.add_argument("--tile_folder", type=str, help="Path to tile folder (detailview_predictions subfolder)")
    parser.add_argument("--merged_laz", type=str, help="Path to merged LAZ file")
    parser.add_argument("--output_plot", type=str, default="instance_size_comparison.png")
    parser.add_argument("--buffer", type=float, default=0.2, help="Buffer distance (default 0.2m)")
    
    args = parser.parse_args()
    
    data_list = []
    
    # Analyze tile if provided
    if args.tile_folder:
        laz_path = os.path.join(args.tile_folder, "pc_with_species.laz")
        instances, name = analyze_laz_instances(laz_path, f"Tile: {Path(args.tile_folder).name}")
        if instances:
            data_list.append((instances, name))
        
        # Check buffer logic
        check_buffer_logic(args.tile_folder, args.buffer)
    
    # Analyze merged file if provided
    if args.merged_laz:
        instances, name = analyze_laz_instances(args.merged_laz, "Merged")
        if instances:
            data_list.append((instances, name))
    
    # Plot comparison
    if data_list:
        plot_instance_distributions(data_list, args.output_plot)


if __name__ == "__main__":
    main()


