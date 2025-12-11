#!/usr/bin/env python3
"""
Check point density in overlapping areas between adjacent tiles.
This helps verify that the merge filter is working correctly.
"""

import sys
import json
import laspy
import numpy as np
from pathlib import Path

def get_tile_bounds(tile_jobs_file):
    """Read tile bounds from tile_jobs file."""
    tiles = {}
    with open(tile_jobs_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                label = parts[0]
                bounds_str = parts[1]
                # Parse bounds: ([xmin, xmax], [ymin, ymax])
                bounds_str = bounds_str.strip('()')
                x_bounds, y_bounds = bounds_str.split('],[')
                xmin, xmax = map(float, x_bounds.strip('[').split(','))
                ymin, ymax = map(float, y_bounds.strip(']').split(','))
                tiles[label] = {
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax
                }
    return tiles

def count_points_in_region(copc_file, xmin, xmax, ymin, ymax):
    """Count points in a specific region using PDAL info or laspy."""
    try:
        import pdal
        import json as json_lib
        
        # Create PDAL pipeline to read bounds
        pipeline_json = {
            "pipeline": [
                {
                    "type": "readers.copc",
                    "filename": str(copc_file),
                    "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])"
                }
            ]
        }
        
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        
        arrays = pipeline.arrays
        if len(arrays) > 0:
            return len(arrays[0])
        return 0
    except ImportError:
        # Fallback: use laspy (less efficient for COPC, but works)
        try:
            las = laspy.read(str(copc_file))
            mask = ((las.x >= xmin) & (las.x <= xmax) & 
                   (las.y >= ymin) & (las.y <= ymax))
            return np.sum(mask)
        except Exception as e:
            print(f"  Error reading {copc_file}: {e}", file=sys.stderr)
            return None

def analyze_overlaps(tiles_dir, tile_jobs_file):
    """Analyze point density in overlapping regions."""
    tiles_dir = Path(tiles_dir)
    tiles = get_tile_bounds(tile_jobs_file)
    
    # Get sorted tile labels
    tile_labels = sorted(tiles.keys())
    
    print("=" * 80)
    print("Point Density Analysis in Overlapping Areas")
    print("=" * 80)
    print()
    
    for i in range(len(tile_labels) - 1):
        tile1_label = tile_labels[i]
        tile2_label = tile_labels[i + 1]
        
        tile1_bounds = tiles[tile1_label]
        tile2_bounds = tiles[tile2_label]
        
        # Find overlap region
        overlap_xmin = max(tile1_bounds['xmin'], tile2_bounds['xmin'])
        overlap_xmax = min(tile1_bounds['xmax'], tile2_bounds['xmax'])
        overlap_ymin = max(tile1_bounds['ymin'], tile2_bounds['ymin'])
        overlap_ymax = min(tile1_bounds['ymax'], tile2_bounds['ymax'])
        
        if overlap_xmin >= overlap_xmax or overlap_ymin >= overlap_ymax:
            print(f"No overlap between {tile1_label} and {tile2_label}")
            continue
        
        tile1_file = tiles_dir / f"{tile1_label}.copc.laz"
        tile2_file = tiles_dir / f"{tile2_label}.copc.laz"
        
        if not tile1_file.exists():
            print(f"Warning: {tile1_file} not found")
            continue
        if not tile2_file.exists():
            print(f"Warning: {tile2_file} not found")
            continue
        
        print(f"Analyzing overlap between {tile1_label} and {tile2_label}:")
        print(f"  Overlap bounds: X=[{overlap_xmin:.2f}, {overlap_xmax:.2f}], Y=[{overlap_ymin:.2f}, {overlap_ymax:.2f}]")
        overlap_area = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
        print(f"  Overlap area: {overlap_area:.2f} m²")
        
        # Count points in overlap region for each tile
        print(f"  Counting points in {tile1_label}...")
        count1 = count_points_in_region(tile1_file, overlap_xmin, overlap_xmax, overlap_ymin, overlap_ymax)
        
        print(f"  Counting points in {tile2_label}...")
        count2 = count_points_in_region(tile2_file, overlap_xmin, overlap_xmax, overlap_ymin, overlap_ymax)
        
        if count1 is not None and count2 is not None:
            density1 = count1 / overlap_area if overlap_area > 0 else 0
            density2 = count2 / overlap_area if overlap_area > 0 else 0
            
            print(f"  Points in {tile1_label}: {count1:,} (density: {density1:.2f} pts/m²)")
            print(f"  Points in {tile2_label}: {count2:,} (density: {density2:.2f} pts/m²)")
            
            # Expected: if merge worked, both tiles should have similar point counts in overlap
            # If only last file contributed, one tile would have significantly fewer points
            ratio = max(count1, count2) / min(count1, count2) if min(count1, count2) > 0 else float('inf')
            
            if ratio > 2.0:
                print(f"  ⚠️  WARNING: Large difference (ratio: {ratio:.2f}x) - merge may not be working correctly")
            elif ratio > 1.5:
                print(f"  ⚠️  CAUTION: Moderate difference (ratio: {ratio:.2f}x) - check merge filter")
            else:
                print(f"  ✓ Similar point counts (ratio: {ratio:.2f}x) - merge appears to be working")
            
            # Also check core regions for comparison
            tile1_core_area = (tile1_bounds['xmax'] - tile1_bounds['xmin']) * (tile1_bounds['ymax'] - tile1_bounds['ymin'])
            tile2_core_area = (tile2_bounds['xmax'] - tile2_bounds['xmin']) * (tile2_bounds['ymax'] - tile2_bounds['ymin'])
            
            print(f"  Core area {tile1_label}: {tile1_core_area:.2f} m²")
            print(f"  Core area {tile2_label}: {tile2_core_area:.2f} m²")
            print()
        else:
            print(f"  Error counting points")
            print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: check_overlap_density.py <tiles_dir> <tile_jobs_file>")
        sys.exit(1)
    
    tiles_dir = sys.argv[1]
    tile_jobs_file = sys.argv[2]
    
    analyze_overlaps(tiles_dir, tile_jobs_file)






