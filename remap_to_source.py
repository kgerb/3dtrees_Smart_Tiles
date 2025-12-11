#!/usr/bin/env python3
"""
Remap merged predictions back to original source point clouds.

Takes the merged LAZ file with global instance IDs and species predictions,
and maps them to the original source LAZ files using nearest neighbor matching.
"""

import argparse
import os
import numpy as np
import laspy
from scipy.spatial import KDTree
from pathlib import Path


def remap_to_source(
    merged_laz_path: str,
    source_files: list,
    output_folder: str,
    num_threads: int = 8,
    max_distance: float = 0.1,
):
    """
    Remap predictions from merged LAZ to original source files.
    
    Args:
        merged_laz_path: Path to merged LAZ with PredInstance and species_id
        source_files: List of paths to original source LAZ files
        output_folder: Output folder for remapped files
        num_threads: Number of threads for KDTree queries
        max_distance: Maximum distance for point matching. Points farther than
                      this from any merged point are assigned to background (0).
    """
    print("=" * 60)
    print("Remap Merged Predictions to Source Files")
    print("=" * 60)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load merged point cloud
    print(f"\nLoading merged LAZ: {merged_laz_path}")
    merged_las = laspy.read(merged_laz_path, laz_backend=laspy.LazBackend.LazrsParallel)
    merged_points = np.vstack((merged_las.x, merged_las.y, merged_las.z)).T
    
    print(f"  Merged points: {len(merged_points)}")
    
    # Get attributes from merged file
    merged_pred_instance = np.array(merged_las.PredInstance)
    merged_species_id = np.array(merged_las.species_id)
    
    # Build KDTree from merged points
    print("\nBuilding KD-tree for nearest neighbor search...")
    tree = KDTree(merged_points)
    
    # Process each source file
    for source_file in source_files:
        if not os.path.exists(source_file):
            print(f"\nWarning: Source file not found: {source_file}")
            continue
        
        source_name = Path(source_file).stem
        output_file = os.path.join(output_folder, f"{source_name}_with_predictions.laz")
        
        print(f"\n--- Processing: {source_name} ---")
        
        # Load source point cloud
        print(f"  Loading source file...")
        source_las = laspy.read(source_file, laz_backend=laspy.LazBackend.LazrsParallel)
        source_points = np.vstack((source_las.x, source_las.y, source_las.z)).T
        
        print(f"  Source points: {len(source_points)}")
        
        # Find nearest neighbors in merged cloud
        print(f"  Finding nearest neighbors (using {num_threads} workers)...")
        distances, indices = tree.query(source_points, workers=num_threads)
        
        # Report matching statistics
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        within_threshold = np.sum(distances <= max_distance)
        outside_threshold = len(distances) - within_threshold
        print(f"  Match distances: mean={mean_dist:.4f}m, max={max_dist:.4f}m")
        print(f"  Points within {max_distance}m threshold: {within_threshold} ({100*within_threshold/len(distances):.1f}%)")
        print(f"  Points outside threshold (→ background): {outside_threshold} ({100*outside_threshold/len(distances):.1f}%)")
        
        # Add extra dimensions if needed
        extra_dims = {dim.name for dim in source_las.point_format.extra_dimensions}
        
        if "PredInstance" not in extra_dims:
            source_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
            )
        if "species_id" not in extra_dims:
            source_las.add_extra_dim(
                laspy.ExtraBytesParams(name="species_id", type=np.int32)
            )
        
        # Map attributes from merged to source
        print(f"  Mapping attributes...")
        pred_instance = merged_pred_instance[indices].astype(np.int32)
        species_id = merged_species_id[indices].astype(np.int32)
        
        # Set points outside distance threshold to background (0)
        too_far_mask = distances > max_distance
        pred_instance[too_far_mask] = 0
        species_id[too_far_mask] = 0
        
        source_las.PredInstance = pred_instance
        source_las.species_id = species_id
        
        # Renumber instances to start from 1 for this file
        print(f"  Renumbering instances to start from 1...")
        unique_instances = np.unique(source_las.PredInstance)
        unique_instances = unique_instances[unique_instances > 0]  # Exclude 0 and negatives
        
        old_to_new = {0: 0, -1: -1}
        for new_id, old_id in enumerate(sorted(unique_instances), start=1):
            old_to_new[old_id] = new_id
        
        # Apply renumbering
        source_las.PredInstance = np.array([old_to_new.get(x, 0) for x in source_las.PredInstance], dtype=np.int32)
        
        # Count unique instances after renumbering
        num_instances = len(unique_instances)
        print(f"  Unique tree instances in source: {num_instances} (numbered 1 to {num_instances})")
        
        # Save output
        print(f"  Saving: {output_file}")
        with open(output_file, "wb") as f:
            source_las.write(f, do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel)
            f.flush()
            os.fsync(f.fileno())
        
        print(f"  Done!")
    
    print("\n" + "=" * 60)
    print("Remapping complete!")
    print(f"Output folder: {output_folder}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Remap merged predictions to original source point clouds"
    )
    parser.add_argument(
        "--merged_laz",
        type=str,
        required=True,
        help="Path to merged LAZ file with predictions"
    )
    parser.add_argument(
        "--source_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to original source LAZ files"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder for remapped files"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of threads for KDTree queries (default: 8)"
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=0.5,
        help="Maximum distance for matching. Points farther are set to background (default: 0.5m)"
    )
    
    args = parser.parse_args()
    
    remap_to_source(
        merged_laz_path=args.merged_laz,
        source_files=args.source_files,
        output_folder=args.output_folder,
        num_threads=args.num_threads,
        max_distance=args.max_distance,
    )


if __name__ == "__main__":
    main()

