#!/usr/bin/env python3
"""
Simple script to check point density in overlapping areas using pdal info.
"""

import sys
import subprocess
import json
import tempfile
import os
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
    """Count points in a bounded region using pdal pipeline."""
    # Create temporary pipeline JSON file with a null writer to count points
    pipeline = [
        {
            "type": "readers.copc",
            "filename": str(copc_file),
            "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])"
        },
        {
            "type": "writers.null"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(pipeline, f, indent=2)
        pipeline_file = f.name
    
    try:
        # Run pdal pipeline
        result = subprocess.run(
            ['pdal', 'pipeline', '--metadata', pipeline_file],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy()
        )
        
        if result.returncode != 0:
            # Try with info instead
            result = subprocess.run(
                ['pdal', 'info', pipeline_file],
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy()
            )
        
        if result.returncode != 0:
            return None
        
        # Parse JSON output
        data = json.loads(result.stdout)
        
        # Try to get point count from metadata or stats
        metadata = data.get('metadata', {})
        readers = metadata.get('readers.copc', {})
        if isinstance(readers, dict):
            count = readers.get('num_points', 0)
            if count > 0:
                return int(count)
        
        # Try stats path
        stats = data.get('stats', {})
        bbox = stats.get('bbox', {})
        native = bbox.get('native', {})
        count = native.get('points', 0)
        if count > 0:
            return int(count)
        
        # Try metadata path
        if 'metadata' in data:
            for key in data['metadata']:
                if 'num_points' in str(data['metadata'][key]):
                    try:
                        count = data['metadata'][key].get('num_points', 0)
                        if count > 0:
                            return int(count)
                    except:
                        pass
        
        return None
    except json.JSONDecodeError as e:
        return None
    except Exception as e:
        return None
    finally:
        if os.path.exists(pipeline_file):
            os.unlink(pipeline_file)

def main():
    if len(sys.argv) != 3:
        print("Usage: check_overlap_density_simple.py <tiles_dir> <tile_jobs_file>")
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
                elif ratio > 1.5:
                    print(f"  ⚠️  CAUTION: Moderate difference (ratio: {ratio:.2f}x) - check merge filter")
                else:
                    print(f"  ✓ Similar point counts (ratio: {ratio:.2f}x) - merge appears to be working")
            else:
                print(f"  ⚠️  One or both counts are zero")
        else:
            print(f"  Error counting points")
        
        print()

if __name__ == "__main__":
    main()

