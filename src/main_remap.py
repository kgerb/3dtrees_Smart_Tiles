#!/usr/bin/env python3
"""
Main remap script: Remap predictions from 10cm to target resolution (default: 2cm).

This script handles remapping of segmented predictions from coarse resolution
back to finer resolution using KDTree nearest neighbor lookup.

Usage:
    python main_remap.py --subsampled_10cm_folder /path/to/10cm --target_resolution 2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import laspy
    from scipy.spatial import KDTree
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install laspy scipy numpy")
    sys.exit(1)

# Import parameters
try:
    from parameters import REMAP_PARAMS
except ImportError:
    REMAP_PARAMS = {
        'target_resolution_cm': 2,
        'workers': 4,
    }


def remap_single_tile(
    segmented_file: Path,
    target_file: Path,
    output_file: Path,
    workers: int = -1
) -> Tuple[str, bool, str, int]:
    """
    Remap predictions from segmented file to target resolution file.
    
    Uses KDTree nearest neighbor search to transfer attributes from
    the segmented (coarse) file to the target (fine) file.
    
    Args:
        segmented_file: Path to segmented LAZ file (e.g., 10cm with predictions)
        target_file: Path to target resolution LAZ file (e.g., 2cm)
        output_file: Path for output LAZ file
        workers: Number of workers for KDTree queries (-1 = all CPUs)
    
    Returns:
        Tuple of (tile_id, success, message, point_count)
    """
    tile_id = segmented_file.stem.replace('_segmented', '').replace('_results', '')
    
    try:
        # Load segmented point cloud (source of predictions)
        segmented_las = laspy.read(
            str(segmented_file), 
            laz_backend=laspy.LazBackend.LazrsParallel
        )
        segmented_points = np.vstack((
            segmented_las.x, 
            segmented_las.y, 
            segmented_las.z
        )).T
        
        # Load target resolution point cloud
        target_las = laspy.read(
            str(target_file), 
            laz_backend=laspy.LazBackend.LazrsParallel
        )
        target_points = np.vstack((
            target_las.x, 
            target_las.y, 
            target_las.z
        )).T
        
        # Check for required attributes in segmented file
        extra_dims = {dim.name for dim in segmented_las.point_format.extra_dimensions}
        has_pred_instance = 'PredInstance' in extra_dims
        has_pred_semantic = 'PredSemantic' in extra_dims
        has_species_id = 'species_id' in extra_dims
        
        if not has_pred_instance:
            return (tile_id, False, "No PredInstance attribute in segmented file", 0)
        
        # Create KDTree from segmented points
        tree = KDTree(segmented_points)
        
        # Query nearest neighbors
        distances, indices = tree.query(target_points, workers=-1)
        
        # Create output with target resolution points
        # Add extra dimensions if they don't exist
        target_extra_dims = {dim.name for dim in target_las.point_format.extra_dimensions}
        
        if "PredInstance" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
            )
        
        if has_pred_semantic and "PredSemantic" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredSemantic", type=np.int32)
            )
        
        if has_species_id and "species_id" not in target_extra_dims:
            target_las.add_extra_dim(
                laspy.ExtraBytesParams(name="species_id", type=np.int32)
            )
        
        # Transfer attributes using nearest neighbor indices
        target_las.PredInstance = segmented_las.PredInstance[indices]
        
        if has_pred_semantic:
            target_las.PredSemantic = segmented_las.PredSemantic[indices]
        
        if has_species_id:
            target_las.species_id = segmented_las.species_id[indices]
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save output
        with open(str(output_file), "wb") as f:
            target_las.write(
                f, 
                do_compress=True, 
                laz_backend=laspy.LazBackend.LazrsParallel
            )
            f.flush()
            os.fsync(f.fileno())
        
        return (tile_id, True, "Success", len(target_points))
        
    except Exception as e:
        return (tile_id, False, str(e), 0)


def find_matching_files(
    results_dir: Path,
    target_folder: Path,
    target_resolution_m: float
) -> List[Tuple[Path, Path, str]]:
    """
    Find matching segmented and target resolution files.
    
    Args:
        results_dir: Directory containing *_results folders with segmented_pc.laz
        target_folder: Directory containing target resolution files
        target_resolution_m: Target resolution in meters
    
    Returns:
        List of (segmented_file, target_file, tile_id) tuples
    """
    matches = []
    
    # Find all results directories
    results_dirs = sorted(results_dir.glob("*_results"))
    
    for result_dir in results_dirs:
        # Get tile ID from directory name (e.g., c00_r00)
        dir_name = result_dir.name
        match = re.search(r'(c\d+_r\d+)', dir_name)
        if not match:
            continue
        
        tile_id = match.group(1)
        
        # Path to segmented file
        segmented_file = result_dir / "segmented_pc.laz"
        if not segmented_file.exists():
            continue
        
        # Find matching target resolution file
        # Try multiple naming patterns
        patterns = [
            f"{tile_id}.copc_subsampled{target_resolution_m}m.laz",
            f"{tile_id}*_subsampled{target_resolution_m}m.laz",
            f"*{tile_id}*.laz",
        ]
        
        target_file = None
        for pattern in patterns:
            matches_found = list(target_folder.glob(pattern))
            if matches_found:
                target_file = matches_found[0]
                break
        
        if target_file and target_file.exists():
            matches.append((segmented_file, target_file, tile_id))
    
    return matches


def remap_all_tiles(
    subsampled_10cm_dir: Path,
    target_resolution_cm: int = 2,
    subsampled_target_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    num_threads: int = 4
) -> Path:
    """
    Remap predictions from 10cm to target resolution for all tiles.
    
    Args:
        subsampled_10cm_dir: Path to folder containing *_results directories
        target_resolution_cm: Target resolution in cm (default: 2)
        subsampled_target_folder: Path to target resolution folder (auto-derived if None)
        output_folder: Output folder for remapped files (auto-derived if None)
        num_threads: Number of workers for KDTree queries
    
    Returns:
        Path to output folder
    """
    print("=" * 60)
    print("3DTrees Remap Pipeline")
    print("=" * 60)
    
    # Auto-derive paths if not provided
    if subsampled_target_folder is None:
        # Replace "subsampled_10cm" with "subsampled_{target}cm"
        folder_name = subsampled_10cm_dir.name
        new_name = folder_name.replace("10cm", f"{target_resolution_cm}cm")
        subsampled_target_folder = subsampled_10cm_dir.parent / new_name
    
    if output_folder is None:
        # Create segmented_remapped folder at same level
        output_folder = subsampled_10cm_dir.parent / "segmented_remapped"
    
    print(f"Input (10cm): {subsampled_10cm_dir}")
    print(f"Target ({target_resolution_cm}cm): {subsampled_target_folder}")
    print(f"Output: {output_folder}")
    print(f"Workers: {num_threads}")
    print()
    
    # Validate directories exist
    if not subsampled_10cm_dir.exists():
        raise ValueError(f"Input directory not found: {subsampled_10cm_dir}")
    
    if not subsampled_target_folder.exists():
        raise ValueError(f"Target directory not found: {subsampled_target_folder}")
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Convert resolution to meters
    target_resolution_m = target_resolution_cm / 100.0
    
    # Find matching files
    matches = find_matching_files(
        subsampled_10cm_dir, 
        subsampled_target_folder, 
        target_resolution_m
    )
    
    if not matches:
        # Try alternative: direct LAZ files in the 10cm folder
        print("  Looking for direct LAZ files...")
        segmented_files = list(subsampled_10cm_dir.glob("*_segmented*.laz"))
        
        for seg_file in segmented_files:
            tile_id_match = re.search(r'(c\d+_r\d+)', seg_file.name)
            if not tile_id_match:
                continue
            tile_id = tile_id_match.group(1)
            
            # Find target file
            target_patterns = [
                f"*{tile_id}*.laz",
            ]
            
            for pattern in target_patterns:
                target_files = list(subsampled_target_folder.glob(pattern))
                if target_files:
                    matches.append((seg_file, target_files[0], tile_id))
                    break
    
    if not matches:
        raise ValueError(f"No matching segmented/target file pairs found")
    
    print(f"Found {len(matches)} tiles to remap")
    print()
    
    # Process each tile
    successful = 0
    failed = 0
    total_points = 0
    
    for i, (segmented_file, target_file, tile_id) in enumerate(matches, 1):
        print(f"[{i}/{len(matches)}] Processing tile {tile_id}...")
        
        output_file = output_folder / f"{tile_id}_segmented_remapped.laz"
        
        # Skip if already exists
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"  Skipping (already exists)")
            successful += 1
            continue
        
        tile_id_result, success, message, point_count = remap_single_tile(
            segmented_file,
            target_file,
            output_file,
            workers=num_threads
        )
        
        if success:
            successful += 1
            total_points += point_count
            print(f"  ✓ {point_count:,} points")
        else:
            failed += 1
            print(f"  ✗ {message}")
    
    # Summary
    print()
    print("=" * 60)
    print("Remap Pipeline Complete")
    print("=" * 60)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total points: {total_points:,}")
    print(f"  Output: {output_folder}")
    
    return output_folder


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Remap Pipeline - Remap predictions to target resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--subsampled_10cm_folder",
        type=Path,
        required=True,
        help="Path to folder containing *_results directories or segmented LAZ files"
    )
    
    parser.add_argument(
        "--target_resolution",
        type=int,
        default=REMAP_PARAMS.get('target_resolution_cm', 2),
        help=f"Target resolution in cm (default: {REMAP_PARAMS.get('target_resolution_cm', 2)})"
    )
    
    parser.add_argument(
        "--subsampled_target_folder",
        type=Path,
        default=None,
        help="Path to target resolution folder (auto-derived if not specified)"
    )
    
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=None,
        help="Output folder for remapped files (auto-derived if not specified)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=REMAP_PARAMS.get('workers', 4),
        help=f"Number of workers for KDTree queries (default: {REMAP_PARAMS.get('workers', 4)})"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        output_folder = remap_all_tiles(
            subsampled_10cm_dir=args.subsampled_10cm_folder,
            target_resolution_cm=args.target_resolution,
            subsampled_target_folder=args.subsampled_target_folder,
            output_folder=args.output_folder,
            num_threads=args.workers
        )
        print(f"\nRemapped files ready: {output_folder}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

