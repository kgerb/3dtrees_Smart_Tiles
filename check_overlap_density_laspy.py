#!/usr/bin/env python3
"""
Check point density in overlapping areas using laspy.
"""

import sys
import laspy
import numpy as np
from pathlib import Path

def parse_bounds(bounds_str):
    """Parse bounds string: ([xmin, xmax], [ymin, ymax])"""
    bounds_str = bounds_str.strip('()')
    parts = bounds_str.split('],[')
    x_part = parts[0].strip('[')
    y_part = parts[1].strip(']')
    
    xmin, xmax = map(float, x_part.split(','))
    ymin, ymax = map(float, y_part.split(','))
    
    return xmin, xmax, ymin, ymax

def count_points_in_bounds(copc_file, xmin, xmax, ymin, ymax):
    """Count points in a bounded region using laspy."""
    try:
        las = laspy.read(str(copc_file))
        
        # Filter points within bounds
        mask = ((las.x >= xmin) & (las.x <= xmax) & 
               (las.y >= ymin) & (las.y <= ymax))
        
        count = np.sum(mask)
        return int(count)
    except Exception as e:
        print(f"    Error reading {copc_file}: {e}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: check_overlap_density_laspy.py <tiles_dir> <tile_jobs_file>")
        sys.exit(1)
    
    tiles_dir = Path(sys.argv[1])
    tile_jobs_file = sys.argv[2]
    
    # Read tile jobs
    tiles = []
    with open(tile_jobs_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                label = parts[0].strip()
                proj_bounds = parts[1].strip()
                xmin, xmax, ymin, ymax = parse_bounds(proj_bounds)
                tiles.append({
                    'label': label,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax
                })
    
    print("=" * 80)
    print("Point Density Analysis in Overlapping Areas")
    print("=" * 80)
    print()
    
    for i in range(len(tiles) - 1):
        tile1 = tiles[i]
        tile2 = tiles[i + 1]
        
        # Find overlap region
        overlap_xmin = max(tile1['xmin'], tile2['xmin'])
        overlap_xmax = min(tile1['xmax'], tile2['xmax'])
        overlap_ymin = max(tile1['ymin'], tile2['ymin'])
        overlap_ymax = min(tile1['ymax'], tile2['ymax'])
        
        if overlap_xmin >= overlap_xmax or overlap_ymin >= overlap_ymax:
            print(f"No overlap between {tile1['label']} and {tile2['label']}")
            continue
        
        overlap_area = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
        
        tile1_file = tiles_dir / f"{tile1['label']}.copc.laz"
        tile2_file = tiles_dir / f"{tile2['label']}.copc.laz"
        
        if not tile1_file.exists():
            print(f"Warning: {tile1_file} not found")
            continue
        if not tile2_file.exists():
            print(f"Warning: {tile2_file} not found")
            continue
        
        print(f"Analyzing overlap between {tile1['label']} and {tile2['label']}:")
        print(f"  Overlap bounds: X=[{overlap_xmin:.2f}, {overlap_xmax:.2f}], Y=[{overlap_ymin:.2f}, {overlap_ymax:.2f}]")
        print(f"  Overlap area: {overlap_area:.2f} m²")
        
        # Count points
        print(f"  Counting points in {tile1['label']}...")
        count1 = count_points_in_bounds(tile1_file, overlap_xmin, overlap_xmax, overlap_ymin, overlap_ymax)
        
        print(f"  Counting points in {tile2['label']}...")
        count2 = count_points_in_bounds(tile2_file, overlap_xmin, overlap_xmax, overlap_ymin, overlap_ymax)
        
        if count1 is not None and count2 is not None:
            density1 = count1 / overlap_area if overlap_area > 0 else 0
            density2 = count2 / overlap_area if overlap_area > 0 else 0
            
            print(f"  Points in {tile1['label']}: {count1:,} (density: {density1:.2f} pts/m²)")
            print(f"  Points in {tile2['label']}: {count2:,} (density: {density2:.2f} pts/m²)")
            
            # Calculate ratio
            if count1 > 0 and count2 > 0:
                ratio = max(count1, count2) / min(count1, count2)
                
                if ratio > 2.0:
                    print(f"  ⚠️  WARNING: Large difference (ratio: {ratio:.2f}x) - merge may not be working correctly")
                    print(f"      This suggests only one tile's points are present in the overlap")
                elif ratio > 1.5:
                    print(f"  ⚠️  CAUTION: Moderate difference (ratio: {ratio:.2f}x) - check merge filter")
                else:
                    print(f"  ✓ Similar point counts (ratio: {ratio:.2f}x) - merge appears to be working")
                    print(f"      Both tiles contribute points to the overlap region")
            else:
                print(f"  ⚠️  One or both counts are zero")
        else:
            print(f"  Error counting points")
        
        print()

if __name__ == "__main__":
    main()






